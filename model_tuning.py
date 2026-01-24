"""
Model Tuning - Test additional variables and optimize parameters
"""

import json
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def load_data():
    with open('all_players_matches.json', 'r') as f:
        return json.load(f)


def calculate_rolling_stats(matches, lookback=10):
    """Calculate rolling stats with all available metrics"""
    if not matches or len(matches) == 0:
        return None
    
    recent = matches[:lookback]
    
    # Record & streaks
    wins = sum(1 for m in recent if m['result'] == 'W')
    win_pct = wins / len(recent) if recent else 0.5
    
    # Current streak
    streak = 0
    if recent:
        streak_type = recent[0]['result']
        for m in recent:
            if m['result'] == streak_type:
                streak += 1 if streak_type == 'W' else -1
            else:
                break
    
    # Stats collection
    stats = {
        'first_in_pct': [], 'first_won_pct': [], 'second_won_pct': [],
        'ace_pct': [], 'df_pct': [], 'bp_saved_pct': [],
        'rpw_pct': [], 'bp_conv_pct': [],
        'v_ace_pct': [], 'v_first_won_pct': [], 'v_second_won_pct': []
    }
    
    dr_values = []
    match_times = []
    
    for m in recent:
        serve = m.get('serve', {})
        ret = m.get('return', {})
        
        for key in ['first_in_pct', 'first_won_pct', 'second_won_pct', 
                    'ace_pct', 'df_pct', 'bp_saved_pct']:
            if serve.get(key) is not None:
                stats[key].append(serve[key])
        
        for key in ['rpw_pct', 'bp_conv_pct', 'v_ace_pct', 'v_first_won_pct', 'v_second_won_pct']:
            if ret.get(key) is not None:
                stats[key].append(ret[key])
        
        # DR calculation
        spw = (serve.get('first_won_pct') or 65) * 0.6 + (serve.get('second_won_pct') or 50) * 0.4
        rpw = ret.get('rpw_pct') or 35
        if (100 - rpw) > 0:
            dr_values.append(spw / (100 - rpw))
        
        # Match time
        try:
            match_times.append(int(m.get('time_mins', 0)))
        except:
            pass
    
    return {
        'win_pct': win_pct,
        'wins': wins,
        'losses': len(recent) - wins,
        'streak': streak,
        'match_count': len(recent),
        'avg_dr': np.mean(dr_values) if dr_values else 1.0,
        'avg_match_time': np.mean(match_times) if match_times else 90,
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
    
    # Blend numeric stats
    blended = {}
    for key in ['win_pct', 'avg_dr', 'first_in_pct', 'first_won_pct', 'second_won_pct',
                'ace_pct', 'df_pct', 'bp_saved_pct', 'rpw_pct', 'bp_conv_pct',
                'v_ace_pct', 'v_first_won_pct', 'v_second_won_pct', 'avg_match_time']:
        s_val = surf_stats.get(key) if surf_stats.get(key) is not None else 0
        o_val = other_stats.get(key) if other_stats.get(key) is not None else 0
        blended[key] = adaptive_weight * s_val + (1 - adaptive_weight) * o_val
    
    blended['surface_match_count'] = surf_count
    blended['streak'] = surf_stats.get('streak', 0)  # Use surface streak
    return blended


def round_to_numeric(round_str):
    """Convert round string to numeric depth (higher = later round)"""
    round_map = {
        'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4,
        'QF': 5, 'SF': 6, 'F': 7,
        'RR': 2,  # Round robin
    }
    return round_map.get(round_str, 3)


def build_enhanced_training_data(data, lookback=10, surface_weight=0.6):
    """Build training data with enhanced features"""
    
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
    
    X, y, info = [], [], []
    
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
            
            # Calculate stats
            p1_all = calculate_rolling_stats(historical, lookback)
            p2_all = calculate_rolling_stats(opp_historical, lookback)
            p1_blend = calculate_blended_stats(historical, surface, lookback, surface_weight)
            p2_blend = calculate_blended_stats(opp_historical, surface, lookback, surface_weight)
            
            if not all([p1_all, p2_all, p1_blend, p2_blend]):
                continue
            
            # Get ranks
            try:
                p1_rank = int(match.get('rank', 100))
            except:
                p1_rank = 100
            try:
                p2_rank = int(match.get('opp_rank', 100))
            except:
                p2_rank = 100
            
            # Round depth
            round_depth = round_to_numeric(match.get('round', 'R32'))
            
            # Safe getter
            def safe(stats, key, default=0):
                val = stats.get(key)
                return val if val is not None else default
            
            features = [
                # === ELO (4) ===
                pdata['elo_overall'] - opp_data['elo_overall'],
                pdata.get(surface_elo_key, 1500) - opp_data.get(surface_elo_key, 1500),
                pdata['elo_overall'],
                opp_data['elo_overall'],
                
                # === RANK (3) ===
                p2_rank - p1_rank,  # Positive = P1 ranked higher (lower number)
                np.log1p(p2_rank) - np.log1p(p1_rank),  # Log rank diff
                p1_rank,
                
                # === FORM - All surfaces (4) ===
                safe(p1_all, 'win_pct', 0.5) - safe(p2_all, 'win_pct', 0.5),
                safe(p1_all, 'avg_dr', 1) - safe(p2_all, 'avg_dr', 1),
                safe(p1_all, 'streak', 0),  # P1 current streak
                safe(p2_all, 'streak', 0),  # P2 current streak
                
                # === SERVE - All surfaces (5) ===
                safe(p1_all, 'first_in_pct', 60) - safe(p2_all, 'first_in_pct', 60),
                safe(p1_all, 'first_won_pct', 65) - safe(p2_all, 'first_won_pct', 65),
                safe(p1_all, 'second_won_pct', 50) - safe(p2_all, 'second_won_pct', 50),
                safe(p1_all, 'ace_pct', 5) - safe(p2_all, 'ace_pct', 5),
                safe(p1_all, 'df_pct', 3) - safe(p2_all, 'df_pct', 3),
                
                # === RETURN - All surfaces (4) ===
                safe(p1_all, 'rpw_pct', 35) - safe(p2_all, 'rpw_pct', 35),
                safe(p1_all, 'bp_conv_pct', 40) - safe(p2_all, 'bp_conv_pct', 40),
                safe(p1_all, 'bp_saved_pct', 60) - safe(p2_all, 'bp_saved_pct', 60),
                safe(p1_all, 'v_first_won_pct', 35) - safe(p2_all, 'v_first_won_pct', 35),  # Return vs 1st serve
                
                # === BLENDED SERVE - Surface weighted (4) ===
                safe(p1_blend, 'first_won_pct', 65) - safe(p2_blend, 'first_won_pct', 65),
                safe(p1_blend, 'second_won_pct', 50) - safe(p2_blend, 'second_won_pct', 50),
                safe(p1_blend, 'ace_pct', 5) - safe(p2_blend, 'ace_pct', 5),
                safe(p1_blend, 'bp_saved_pct', 60) - safe(p2_blend, 'bp_saved_pct', 60),
                
                # === BLENDED RETURN - Surface weighted (3) ===
                safe(p1_blend, 'rpw_pct', 35) - safe(p2_blend, 'rpw_pct', 35),
                safe(p1_blend, 'bp_conv_pct', 40) - safe(p2_blend, 'bp_conv_pct', 40),
                safe(p1_blend, 'v_first_won_pct', 35) - safe(p2_blend, 'v_first_won_pct', 35),
                
                # === BLENDED FORM (3) ===
                safe(p1_blend, 'win_pct', 0.5) - safe(p2_blend, 'win_pct', 0.5),
                safe(p1_blend, 'avg_dr', 1) - safe(p2_blend, 'avg_dr', 1),
                safe(p1_blend, 'streak', 0) - safe(p2_blend, 'streak', 0),
                
                # === SURFACE EXPERIENCE (3) ===
                safe(p1_blend, 'surface_match_count', 0) - safe(p2_blend, 'surface_match_count', 0),
                safe(p1_blend, 'surface_match_count', 0),
                safe(p2_blend, 'surface_match_count', 0),
                
                # === MATCH CONTEXT (2) ===
                round_depth,  # Later rounds might favor experience
                safe(p1_blend, 'avg_match_time', 90) - safe(p2_blend, 'avg_match_time', 90),  # Endurance indicator
            ]
            
            X.append(features)
            y.append(1 if match.get('result') == 'W' else 0)
            info.append({'player': player_name, 'opponent': opp_name, 'surface': surface})
    
    return np.array(X), np.array(y), info


# Feature names
FEATURE_NAMES = [
    # ELO (4)
    'elo_diff', 'surface_elo_diff', 'p1_elo', 'p2_elo',
    # Rank (3)
    'rank_diff', 'log_rank_diff', 'p1_rank',
    # Form (4)
    'win_pct_diff', 'dr_diff', 'p1_streak', 'p2_streak',
    # Serve (5)
    'first_in_diff', 'first_won_diff', 'second_won_diff', 'ace_diff', 'df_diff',
    # Return (4)
    'rpw_diff', 'bp_conv_diff', 'bp_saved_diff', 'v_first_won_diff',
    # Blended serve (4)
    'blend_first_won_diff', 'blend_second_won_diff', 'blend_ace_diff', 'blend_bp_saved_diff',
    # Blended return (3)
    'blend_rpw_diff', 'blend_bp_conv_diff', 'blend_v_first_won_diff',
    # Blended form (3)
    'blend_win_pct_diff', 'blend_dr_diff', 'blend_streak_diff',
    # Surface exp (3)
    'surface_exp_diff', 'p1_surface_exp', 'p2_surface_exp',
    # Context (2)
    'round_depth', 'match_time_diff'
]


def test_parameters():
    """Test different parameter combinations"""
    data = load_data()
    print(f"Loaded {data['player_count']} players")
    
    results = []
    
    # Test different lookback windows and surface weights
    for lookback in [5, 10, 15]:
        for surf_weight in [0.5, 0.6, 0.7]:
            print(f"\nTesting lookback={lookback}, surface_weight={surf_weight}...")
            
            X, y, _ = build_enhanced_training_data(data, lookback, surf_weight)
            
            if len(X) < 100:
                continue
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            model = LogisticRegression(max_iter=1000, C=0.1)
            model.fit(X_train_s, y_train)
            
            acc = model.score(X_test_s, y_test)
            auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])
            cv = cross_val_score(model, X_train_s, y_train, cv=5).mean()
            
            results.append({
                'lookback': lookback,
                'surf_weight': surf_weight,
                'acc': acc,
                'auc': auc,
                'cv': cv,
                'samples': len(X)
            })
            
            print(f"  Acc: {acc:.1%}, AUC: {auc:.3f}, CV: {cv:.1%}, N={len(X)}")
    
    # Find best
    best = max(results, key=lambda x: x['auc'])
    print(f"\n{'='*60}")
    print(f"BEST PARAMETERS: lookback={best['lookback']}, surface_weight={best['surf_weight']}")
    print(f"  Accuracy: {best['acc']:.1%}")
    print(f"  AUC-ROC:  {best['auc']:.3f}")
    print(f"  CV:       {best['cv']:.1%}")
    
    return best


def analyze_features(lookback=10, surf_weight=0.6):
    """Analyze feature importance and correlations"""
    data = load_data()
    X, y, _ = build_enhanced_training_data(data, lookback, surf_weight)
    
    print(f"\n{'='*60}")
    print("FEATURE ANALYSIS")
    print(f"{'='*60}")
    print(f"Samples: {len(X)}")
    
    # Correlations
    print("\nTop Correlations with Winning:")
    correlations = []
    for i, name in enumerate(FEATURE_NAMES):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        correlations.append((name, corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, corr in correlations[:15]:
        print(f"  {name}: {'+' if corr > 0 else ''}{corr:.3f}")
    
    # Feature selection
    print("\n" + "="*60)
    print("FEATURE SELECTION (SelectKBest)")
    print("="*60)
    
    selector = SelectKBest(f_classif, k=15)
    selector.fit(X, y)
    
    scores = list(zip(FEATURE_NAMES, selector.scores_))
    scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 15 Features by F-score:")
    for name, score in scores[:15]:
        print(f"  {name}: {score:.1f}")
    
    # Train final model with best features
    print("\n" + "="*60)
    print("FINAL MODEL (Top 15 Features)")
    print("="*60)
    
    X_selected = selector.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, C=0.1)
    model.fit(X_train_s, y_train)
    
    print(f"\nWith feature selection:")
    print(f"  Test Accuracy: {model.score(X_test_s, y_test):.1%}")
    print(f"  AUC-ROC: {roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1]):.3f}")
    
    # Compare to full model
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler_f = StandardScaler()
    X_train_fs = scaler_f.fit_transform(X_train_f)
    X_test_fs = scaler_f.transform(X_test_f)
    
    model_full = LogisticRegression(max_iter=1000, C=0.1)
    model_full.fit(X_train_fs, y_train_f)
    
    print(f"\nWith all {len(FEATURE_NAMES)} features:")
    print(f"  Test Accuracy: {model_full.score(X_test_fs, y_test_f):.1%}")
    print(f"  AUC-ROC: {roc_auc_score(y_test_f, model_full.predict_proba(X_test_fs)[:, 1]):.3f}")
    
    # Feature importances from full model
    print("\n" + "="*60)
    print("FEATURE WEIGHTS (Logistic Regression)")
    print("="*60)
    
    weights = list(zip(FEATURE_NAMES, model_full.coef_[0]))
    weights.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\nTop features by model weight:")
    for name, weight in weights[:15]:
        print(f"  {name}: {'+' if weight > 0 else ''}{weight:.3f}")


if __name__ == '__main__':
    # Test parameters
    best = test_parameters()
    
    # Analyze with best parameters
    analyze_features(best['lookback'], best['surf_weight'])
