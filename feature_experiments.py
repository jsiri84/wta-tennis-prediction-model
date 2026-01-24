"""
Feature Experiments - Systematically test adding new variables
"""

import json
import numpy as np
from itertools import combinations
from collections import defaultdict
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def load_data():
    with open('all_players_matches.json', 'r') as f:
        return json.load(f)


def calculate_rolling_stats(matches, lookback=10):
    """Calculate rolling stats"""
    if not matches or len(matches) == 0:
        return None
    
    recent = matches[:lookback]
    wins = sum(1 for m in recent if m['result'] == 'W')
    win_pct = wins / len(recent) if recent else 0.5
    
    streak = 0
    if recent:
        streak_type = recent[0]['result']
        for m in recent:
            if m['result'] == streak_type:
                streak += 1 if streak_type == 'W' else -1
            else:
                break
    
    stats = {
        'first_in_pct': [], 'first_won_pct': [], 'second_won_pct': [],
        'ace_pct': [], 'df_pct': [], 'bp_saved_pct': [],
        'rpw_pct': [], 'bp_conv_pct': [], 'v_first_won_pct': []
    }
    
    dr_values = []
    match_times = []
    three_setters = 0
    
    for m in recent:
        serve = m.get('serve', {})
        ret = m.get('return', {})
        
        for key in ['first_in_pct', 'first_won_pct', 'second_won_pct', 
                    'ace_pct', 'df_pct', 'bp_saved_pct']:
            if serve.get(key) is not None:
                stats[key].append(serve[key])
        
        for key in ['rpw_pct', 'bp_conv_pct', 'v_first_won_pct']:
            if ret.get(key) is not None:
                stats[key].append(ret[key])
        
        spw = (serve.get('first_won_pct') or 65) * 0.6 + (serve.get('second_won_pct') or 50) * 0.4
        rpw = ret.get('rpw_pct') or 35
        if (100 - rpw) > 0:
            dr_values.append(spw / (100 - rpw))
        
        try:
            match_times.append(int(m.get('time_mins', 0)))
        except:
            pass
        
        # Count 3-setters (check score for 3 sets)
        score = m.get('score', '')
        set_count = score.count('-')
        if set_count == 3:
            three_setters += 1
    
    return {
        'win_pct': win_pct,
        'wins': wins,
        'streak': streak,
        'match_count': len(recent),
        'avg_dr': np.mean(dr_values) if dr_values else 1.0,
        'avg_match_time': np.mean(match_times) if match_times else 90,
        'three_setter_pct': three_setters / len(recent) if recent else 0,
        **{k: np.mean(v) if v else None for k, v in stats.items()}
    }


def calculate_blended_stats(matches, surface, lookback=10, surface_weight=0.6):
    """Calculate surface-weighted blended stats"""
    if not matches:
        return None
    
    surface_matches = [m for m in matches if m.get('surface', '').lower() == surface.lower()]
    other_matches = [m for m in matches if m.get('surface', '').lower() != surface.lower()]
    
    surf_stats = calculate_rolling_stats(surface_matches[:lookback], lookback)
    other_stats = calculate_rolling_stats(other_matches[:lookback], lookback)
    
    if surf_stats and not other_stats:
        surf_stats['surface_match_count'] = len(surface_matches[:lookback])
        return surf_stats
    if other_stats and not surf_stats:
        other_stats['surface_match_count'] = 0
        return other_stats
    if not surf_stats and not other_stats:
        return None
    
    surf_count = len(surface_matches[:lookback])
    surface_confidence = min(surf_count / 10, 1.0)
    adaptive_weight = surface_weight * surface_confidence + (1 - surface_weight) * (1 - surface_confidence)
    
    blended = {}
    for key in ['win_pct', 'avg_dr', 'first_in_pct', 'first_won_pct', 'second_won_pct',
                'ace_pct', 'df_pct', 'bp_saved_pct', 'rpw_pct', 'bp_conv_pct',
                'v_first_won_pct', 'avg_match_time', 'three_setter_pct']:
        s_val = surf_stats.get(key) if surf_stats.get(key) is not None else 0
        o_val = other_stats.get(key) if other_stats.get(key) is not None else 0
        blended[key] = adaptive_weight * s_val + (1 - adaptive_weight) * o_val
    
    blended['surface_match_count'] = surf_count
    blended['streak'] = surf_stats.get('streak', 0)
    return blended


def round_to_numeric(round_str):
    round_map = {'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4, 'QF': 5, 'SF': 6, 'F': 7, 'RR': 2}
    return round_map.get(round_str, 3)


def tournament_level(tourn_name):
    """Extract tournament level from name"""
    tourn = tourn_name.lower()
    if 'australian open' in tourn or 'french open' in tourn or 'wimbledon' in tourn or 'us open' in tourn:
        return 4  # Grand Slam
    if '1000' in tourn or 'wta 1000' in tourn:
        return 3
    if '500' in tourn or 'wta 500' in tourn:
        return 2
    if '250' in tourn or 'wta 250' in tourn:
        return 1
    return 2  # Default


def days_between(date1, date2):
    """Calculate days between two YYYYMMDD date strings"""
    try:
        d1 = datetime.strptime(date1, '%Y%m%d')
        d2 = datetime.strptime(date2, '%Y%m%d')
        return abs((d2 - d1).days)
    except:
        return 7  # Default


def build_training_data(data, include_h2h=False, include_rest=False, 
                        include_level=False, include_3set=False):
    """Build training data with optional new features"""
    
    player_data = {}
    for name, info in data['players'].items():
        player_data[name] = {
            'elo_overall': info.get('elo_overall', 1500),
            'elo_hard': info.get('elo_hard', 1500),
            'elo_clay': info.get('elo_clay', 1500),
            'elo_grass': info.get('elo_grass', 1500),
            'wta_rank': info.get('wta_rank', 100),
            'matches': sorted(info.get('matches', []), key=lambda x: x.get('date', ''), reverse=True)
        }
    
    name_lookup = {}
    for name in player_data:
        name_lookup[name.lower()] = name
        name_lookup[name.lower().replace(' ', '')] = name
        parts = name.split()
        if len(parts) > 1:
            name_lookup[parts[-1].lower()] = name
    
    X, y = [], []
    
    for player_name, pdata in player_data.items():
        matches = pdata['matches']
        
        for i, match in enumerate(matches):
            historical = matches[i+1:] if i+1 < len(matches) else []
            if len(historical) < 5:
                continue
            
            opp_name = match.get('opponent', '')
            opp_key = None
            for key in [opp_name.lower(), opp_name.lower().replace(' ', '')]:
                if key in name_lookup:
                    opp_key = name_lookup[key]
                    break
            
            if not opp_key or opp_key not in player_data:
                continue
            
            opp_data = player_data[opp_key]
            match_date = match.get('date', '')
            opp_historical = [m for m in opp_data['matches'] if m.get('date', '') < match_date]
            
            if len(opp_historical) < 5:
                continue
            
            surface = match.get('surface', 'Hard').lower()
            surface_elo_key = f'elo_{surface}' if surface in ['hard', 'clay', 'grass'] else 'elo_hard'
            
            p1_all = calculate_rolling_stats(historical, 10)
            p2_all = calculate_rolling_stats(opp_historical, 10)
            p1_blend = calculate_blended_stats(historical, surface, 10, 0.6)
            p2_blend = calculate_blended_stats(opp_historical, surface, 10, 0.6)
            
            if not all([p1_all, p2_all, p1_blend, p2_blend]):
                continue
            
            try:
                p1_rank = int(match.get('rank', 100))
            except:
                p1_rank = 100
            try:
                p2_rank = int(match.get('opp_rank', 100))
            except:
                p2_rank = 100
            
            round_depth = round_to_numeric(match.get('round', 'R32'))
            
            def safe(stats, key, default=0):
                val = stats.get(key)
                return val if val is not None else default
            
            # BASE FEATURES (35)
            features = [
                # ELO (4)
                pdata['elo_overall'] - opp_data['elo_overall'],
                pdata.get(surface_elo_key, 1500) - opp_data.get(surface_elo_key, 1500),
                pdata['elo_overall'],
                opp_data['elo_overall'],
                # Rank (3)
                p2_rank - p1_rank,
                np.log1p(p2_rank) - np.log1p(p1_rank),
                p1_rank,
                # Form (4)
                safe(p1_all, 'win_pct', 0.5) - safe(p2_all, 'win_pct', 0.5),
                safe(p1_all, 'avg_dr', 1) - safe(p2_all, 'avg_dr', 1),
                safe(p1_all, 'streak', 0),
                safe(p2_all, 'streak', 0),
                # Serve (5)
                safe(p1_all, 'first_in_pct', 60) - safe(p2_all, 'first_in_pct', 60),
                safe(p1_all, 'first_won_pct', 65) - safe(p2_all, 'first_won_pct', 65),
                safe(p1_all, 'second_won_pct', 50) - safe(p2_all, 'second_won_pct', 50),
                safe(p1_all, 'ace_pct', 5) - safe(p2_all, 'ace_pct', 5),
                safe(p1_all, 'df_pct', 3) - safe(p2_all, 'df_pct', 3),
                # Return (4)
                safe(p1_all, 'rpw_pct', 35) - safe(p2_all, 'rpw_pct', 35),
                safe(p1_all, 'bp_conv_pct', 40) - safe(p2_all, 'bp_conv_pct', 40),
                safe(p1_all, 'bp_saved_pct', 60) - safe(p2_all, 'bp_saved_pct', 60),
                safe(p1_all, 'v_first_won_pct', 35) - safe(p2_all, 'v_first_won_pct', 35),
                # Blended serve (4)
                safe(p1_blend, 'first_won_pct', 65) - safe(p2_blend, 'first_won_pct', 65),
                safe(p1_blend, 'second_won_pct', 50) - safe(p2_blend, 'second_won_pct', 50),
                safe(p1_blend, 'ace_pct', 5) - safe(p2_blend, 'ace_pct', 5),
                safe(p1_blend, 'bp_saved_pct', 60) - safe(p2_blend, 'bp_saved_pct', 60),
                # Blended return (3)
                safe(p1_blend, 'rpw_pct', 35) - safe(p2_blend, 'rpw_pct', 35),
                safe(p1_blend, 'bp_conv_pct', 40) - safe(p2_blend, 'bp_conv_pct', 40),
                safe(p1_blend, 'v_first_won_pct', 35) - safe(p2_blend, 'v_first_won_pct', 35),
                # Blended form (3)
                safe(p1_blend, 'win_pct', 0.5) - safe(p2_blend, 'win_pct', 0.5),
                safe(p1_blend, 'avg_dr', 1) - safe(p2_blend, 'avg_dr', 1),
                safe(p1_blend, 'streak', 0) - safe(p2_blend, 'streak', 0),
                # Surface exp (3)
                safe(p1_blend, 'surface_match_count', 0) - safe(p2_blend, 'surface_match_count', 0),
                safe(p1_blend, 'surface_match_count', 0),
                safe(p2_blend, 'surface_match_count', 0),
                # Context (2)
                round_depth,
                safe(p1_blend, 'avg_match_time', 90) - safe(p2_blend, 'avg_match_time', 90),
            ]
            
            # === NEW OPTIONAL FEATURES ===
            
            # HEAD-TO-HEAD
            if include_h2h:
                h2h_wins = 0
                h2h_losses = 0
                for m in historical:
                    if m.get('opponent', '').lower() == opp_name.lower():
                        if m['result'] == 'W':
                            h2h_wins += 1
                        else:
                            h2h_losses += 1
                
                h2h_total = h2h_wins + h2h_losses
                h2h_win_pct = h2h_wins / h2h_total if h2h_total > 0 else 0.5
                
                features.extend([
                    h2h_wins - h2h_losses,  # H2H record diff
                    h2h_win_pct,            # H2H win %
                    h2h_total,              # Number of previous meetings
                ])
            
            # DAYS SINCE LAST MATCH (rest/rust)
            if include_rest:
                p1_last_match = historical[0].get('date', match_date) if historical else match_date
                p2_last_match = opp_historical[0].get('date', match_date) if opp_historical else match_date
                
                p1_rest = days_between(p1_last_match, match_date)
                p2_rest = days_between(p2_last_match, match_date)
                
                # Optimal rest is ~3-5 days. Too little = fatigue, too much = rust
                p1_rest_optimal = 1 if 2 <= p1_rest <= 7 else 0
                p2_rest_optimal = 1 if 2 <= p2_rest <= 7 else 0
                
                features.extend([
                    p2_rest - p1_rest,      # Rest difference (positive = opponent more rested)
                    p1_rest,                # P1 days rest
                    p2_rest,                # P2 days rest
                    p1_rest_optimal - p2_rest_optimal,  # Optimal rest advantage
                ])
            
            # TOURNAMENT LEVEL
            if include_level:
                level = tournament_level(match.get('tournament', ''))
                features.extend([
                    level,                  # Tournament importance (1-4)
                    level * (pdata['elo_overall'] - opp_data['elo_overall']) / 1000,  # ELO diff weighted by level
                ])
            
            # 3-SET MATCH HISTORY (fitness/clutch indicator)
            if include_3set:
                p1_3set = safe(p1_blend, 'three_setter_pct', 0.3)
                p2_3set = safe(p2_blend, 'three_setter_pct', 0.3)
                
                # Also check win rate in 3-setters from history
                p1_3set_wins = sum(1 for m in historical[:20] 
                                   if m['result'] == 'W' and m.get('score', '').count('-') == 3)
                p1_3set_total = sum(1 for m in historical[:20] 
                                    if m.get('score', '').count('-') == 3)
                p2_3set_wins = sum(1 for m in opp_historical[:20] 
                                   if m['result'] == 'W' and m.get('score', '').count('-') == 3)
                p2_3set_total = sum(1 for m in opp_historical[:20] 
                                    if m.get('score', '').count('-') == 3)
                
                p1_3set_winrate = p1_3set_wins / p1_3set_total if p1_3set_total > 0 else 0.5
                p2_3set_winrate = p2_3set_wins / p2_3set_total if p2_3set_total > 0 else 0.5
                
                features.extend([
                    p1_3set - p2_3set,           # 3-set frequency diff
                    p1_3set_winrate - p2_3set_winrate,  # 3-set win rate diff
                    p1_3set_winrate,             # P1 clutch ability
                ])
            
            X.append(features)
            y.append(1 if match.get('result') == 'W' else 0)
    
    return np.array(X), np.array(y)


def evaluate_model(X, y):
    """Evaluate model and return metrics"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, C=0.1)
    model.fit(X_train_s, y_train)
    
    acc = model.score(X_test_s, y_test)
    auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])
    cv = cross_val_score(model, X_train_s, y_train, cv=5).mean()
    
    return acc, auc, cv


def run_experiments():
    """Run all feature combination experiments"""
    data = load_data()
    print(f"Loaded {data['player_count']} players with {data['total_matches']} matches\n")
    
    # Feature flags
    features = ['h2h', 'rest', 'level', '3set']
    feature_names = {
        'h2h': 'Head-to-Head',
        'rest': 'Days Rest',
        'level': 'Tournament Level',
        '3set': '3-Set History'
    }
    
    results = []
    
    # Baseline (no new features)
    print("="*70)
    print("BASELINE (Current Model)")
    print("="*70)
    X, y = build_training_data(data, False, False, False, False)
    acc, auc, cv = evaluate_model(X, y)
    results.append(('Baseline', acc, auc, cv))
    print(f"  Accuracy: {acc:.1%}  |  AUC: {auc:.3f}  |  CV: {cv:.1%}\n")
    
    # Individual features
    print("="*70)
    print("INDIVIDUAL FEATURES (+1)")
    print("="*70)
    for f in features:
        flags = {k: (k == f) for k in features}
        X, y = build_training_data(data, flags['h2h'], flags['rest'], flags['level'], flags['3set'])
        acc, auc, cv = evaluate_model(X, y)
        name = f"+{feature_names[f]}"
        results.append((name, acc, auc, cv))
        print(f"{name:25} |  Acc: {acc:.1%}  |  AUC: {auc:.3f}  |  CV: {cv:.1%}")
    
    # Pairs of features
    print("\n" + "="*70)
    print("FEATURE PAIRS (+2)")
    print("="*70)
    for combo in combinations(features, 2):
        flags = {k: (k in combo) for k in features}
        X, y = build_training_data(data, flags['h2h'], flags['rest'], flags['level'], flags['3set'])
        acc, auc, cv = evaluate_model(X, y)
        name = f"+{feature_names[combo[0]]} +{feature_names[combo[1]]}"
        results.append((name, acc, auc, cv))
        print(f"{name:40} |  Acc: {acc:.1%}  |  AUC: {auc:.3f}  |  CV: {cv:.1%}")
    
    # Triples of features
    print("\n" + "="*70)
    print("FEATURE TRIPLES (+3)")
    print("="*70)
    for combo in combinations(features, 3):
        flags = {k: (k in combo) for k in features}
        X, y = build_training_data(data, flags['h2h'], flags['rest'], flags['level'], flags['3set'])
        acc, auc, cv = evaluate_model(X, y)
        names = [feature_names[c] for c in combo]
        name = f"+{' +'.join(names)}"
        results.append((name, acc, auc, cv))
        print(f"{name:55} |  Acc: {acc:.1%}  |  AUC: {auc:.3f}  |  CV: {cv:.1%}")
    
    # All features
    print("\n" + "="*70)
    print("ALL NEW FEATURES (+4)")
    print("="*70)
    X, y = build_training_data(data, True, True, True, True)
    acc, auc, cv = evaluate_model(X, y)
    name = "+ALL (H2H, Rest, Level, 3-Set)"
    results.append((name, acc, auc, cv))
    print(f"{name:55} |  Acc: {acc:.1%}  |  AUC: {auc:.3f}  |  CV: {cv:.1%}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY - RANKED BY AUC")
    print("="*70)
    results.sort(key=lambda x: x[2], reverse=True)
    
    baseline_auc = [r for r in results if r[0] == 'Baseline'][0][2]
    
    print(f"\n{'Configuration':<55} | {'Acc':>6} | {'AUC':>6} | {'CV':>6} | {'vs Base':>8}")
    print("-"*95)
    for name, acc, auc, cv in results:
        diff = auc - baseline_auc
        diff_str = f"{diff:+.3f}" if name != 'Baseline' else "---"
        marker = " ***" if diff > 0.003 else " **" if diff > 0.001 else " *" if diff > 0 else ""
        print(f"{name:<55} | {acc:>5.1%} | {auc:>6.3f} | {cv:>5.1%} | {diff_str:>8}{marker}")
    
    # Best configuration
    best = results[0]
    print(f"\n{'='*70}")
    print(f"BEST CONFIGURATION: {best[0]}")
    print(f"  Accuracy: {best[1]:.1%}")
    print(f"  AUC-ROC:  {best[2]:.3f} ({best[2] - baseline_auc:+.3f} vs baseline)")
    print(f"  CV:       {best[3]:.1%}")
    print(f"{'='*70}")


if __name__ == '__main__':
    run_experiments()
