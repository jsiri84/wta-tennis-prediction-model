"""
Additional experiments to improve game spread prediction
"""

import json
import numpy as np
import re
from collections import defaultdict
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')


def parse_score(score_str):
    if not score_str:
        return None
    if any(x in score_str.upper() for x in ['RET', 'W/O', 'WO', 'DEF', 'ABD']):
        return None
    
    player_games, opp_games = 0, 0
    tiebreaks = 0
    sets = score_str.strip().split()
    
    for s in sets:
        tiebreaks += s.count('(')
        s_clean = re.sub(r'\(\d+\)', '', s)
        match = re.match(r'(\d+)-(\d+)', s_clean)
        if match:
            player_games += int(match.group(1))
            opp_games += int(match.group(2))
    
    if player_games == 0 and opp_games == 0:
        return None
    
    return {
        'p_games': player_games,
        'o_games': opp_games,
        'spread': player_games - opp_games,
        'total': player_games + opp_games,
        'sets': len(sets),
        'tiebreaks': tiebreaks
    }


def load_data():
    with open('all_players_matches.json', 'r') as f:
        return json.load(f)


def get_round_level(round_str):
    """Convert round to numeric level (higher = later round)"""
    round_map = {'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4, 'QF': 5, 'SF': 6, 'F': 7, 'RR': 2}
    return round_map.get(round_str, 3)


def get_tournament_level(tourn):
    """Get tournament importance: 1=250, 2=500, 3=1000, 4=GS"""
    tourn_lower = tourn.lower()
    if any(gs in tourn_lower for gs in ['australian open', 'french open', 'roland garros', 'wimbledon', 'us open']):
        return 4
    if '1000' in tourn_lower or any(t in tourn_lower for t in ['indian wells', 'miami', 'madrid', 'rome', 'beijing', 'canada']):
        return 3
    if '500' in tourn_lower:
        return 2
    return 1


def calculate_stats(matches, lookback=15):
    if not matches or len(matches) < 3:
        return None
    
    recent = matches[:lookback]
    wins = sum(1 for m in recent if m['result'] == 'W')
    
    spreads, totals = [], []
    straight_sets, three_sets, tiebreaks = 0, 0, 0
    
    for m in recent:
        parsed = parse_score(m.get('score', ''))
        if parsed:
            spreads.append(parsed['spread'])
            totals.append(parsed['total'])
            tiebreaks += parsed['tiebreaks']
            if parsed['sets'] == 2:
                straight_sets += 1
            elif parsed['sets'] == 3:
                three_sets += 1
    
    # Serve/return stats
    serve_won, return_won, aces, dfs = [], [], [], []
    bp_saved, bp_conv = [], []
    
    for m in recent:
        serve = m.get('serve', {})
        ret = m.get('return', {})
        if serve.get('first_won_pct'):
            serve_won.append(serve['first_won_pct'])
        if ret.get('rpw_pct'):
            return_won.append(ret['rpw_pct'])
        if serve.get('ace_pct'):
            aces.append(serve['ace_pct'])
        if serve.get('df_pct'):
            dfs.append(serve['df_pct'])
        if serve.get('bp_saved_pct'):
            bp_saved.append(serve['bp_saved_pct'])
        if ret.get('bp_conv_pct'):
            bp_conv.append(ret['bp_conv_pct'])
    
    n = len(recent)
    n_parsed = len(spreads) if spreads else 1
    
    return {
        'win_pct': wins / n,
        'avg_spread': np.mean(spreads) if spreads else 0,
        'spread_std': np.std(spreads) if len(spreads) > 1 else 3,
        'avg_total': np.mean(totals) if totals else 20,
        'straight_set_rate': straight_sets / n_parsed,
        'three_set_rate': three_sets / n_parsed,
        'tiebreak_rate': tiebreaks / n,
        'serve_won': np.mean(serve_won) if serve_won else 65,
        'return_won': np.mean(return_won) if return_won else 35,
        'ace_pct': np.mean(aces) if aces else 5,
        'df_pct': np.mean(dfs) if dfs else 3,
        'bp_saved': np.mean(bp_saved) if bp_saved else 60,
        'bp_conv': np.mean(bp_conv) if bp_conv else 40,
    }


def calculate_h2h_spread(matches, opponent_name):
    """Calculate historical spread against specific opponent"""
    opp_lower = opponent_name.lower()
    h2h_spreads = []
    
    for m in matches:
        match_opp = m.get('opponent', '').lower()
        if match_opp == opp_lower or match_opp.split()[-1] == opp_lower.split()[-1]:
            parsed = parse_score(m.get('score', ''))
            if parsed:
                h2h_spreads.append(parsed['spread'])
    
    if not h2h_spreads:
        return {'avg': 0, 'count': 0}
    
    return {'avg': np.mean(h2h_spreads), 'count': len(h2h_spreads)}


def build_features_with_context(data):
    """Build features including match context (round, tournament level)"""
    
    player_data = {}
    for name, info in data['players'].items():
        player_data[name] = {
            'elo': info.get('elo_overall', 1500),
            'elo_hard': info.get('elo_hard', 1500),
            'elo_clay': info.get('elo_clay', 1500),
            'elo_grass': info.get('elo_grass', 1500),
            'matches': sorted(info.get('matches', []), key=lambda x: x.get('date', ''), reverse=True)
        }
    
    name_lookup = {}
    for name in player_data:
        name_lookup[name.lower()] = name
        name_lookup[name.lower().replace(' ', '')] = name
        parts = name.split()
        if len(parts) > 1:
            name_lookup[parts[-1].lower()] = name
    
    X, y, y_class = [], [], []
    
    for player_name, pdata in player_data.items():
        for i, match in enumerate(pdata['matches']):
            if i < 5:
                continue
            
            parsed = parse_score(match.get('score', ''))
            if not parsed:
                continue
            
            spread = parsed['spread']
            
            opp_name = match.get('opponent', '')
            opp_key = opp_name.lower().replace(' ', '')
            if opp_key not in name_lookup:
                last = opp_name.split()[-1].lower() if opp_name else ''
                if last in name_lookup:
                    opp_key = last
                else:
                    continue
            
            opp_full = name_lookup.get(opp_key)
            if not opp_full or opp_full not in player_data:
                continue
            
            opp_data = player_data[opp_full]
            
            match_date = match.get('date', '')
            hist = [m for m in pdata['matches'][i+1:] if m.get('date', '') < match_date]
            opp_hist = [m for m in opp_data['matches'] if m.get('date', '') < match_date]
            
            if len(hist) < 5 or len(opp_hist) < 5:
                continue
            
            surface = match.get('surface', 'Hard').lower()
            surface_key = f'elo_{surface}' if surface in ['hard', 'clay', 'grass'] else 'elo_hard'
            
            p1 = calculate_stats(hist)
            p2 = calculate_stats(opp_hist)
            
            if not p1 or not p2:
                continue
            
            # Head-to-head spread history
            h2h = calculate_h2h_spread(hist, opp_name)
            
            # Match context
            round_level = get_round_level(match.get('round', 'R32'))
            tourn_level = get_tournament_level(match.get('tournament', ''))
            
            features = [
                # ELO (3)
                pdata['elo'] - opp_data['elo'],
                pdata.get(surface_key, 1500) - opp_data.get(surface_key, 1500),
                (pdata['elo'] + opp_data['elo']) / 2,
                
                # Spread history (5)
                p1['avg_spread'] - p2['avg_spread'],
                p1['avg_spread'],
                p2['avg_spread'],
                p1['spread_std'] + p2['spread_std'],
                abs(p1['avg_spread']) + abs(p2['avg_spread']),
                
                # Match tendencies (4)
                p1['straight_set_rate'] - p2['straight_set_rate'],
                p1['three_set_rate'] + p2['three_set_rate'],
                p1['tiebreak_rate'] + p2['tiebreak_rate'],
                p1['avg_total'] + p2['avg_total'],
                
                # Dominance (2)
                p1['win_pct'] - p2['win_pct'],
                p1['serve_won'] - p2['serve_won'],
                
                # Serve/Return (4)
                p1['ace_pct'] - p2['ace_pct'],
                p1['df_pct'] - p2['df_pct'],
                p1['bp_saved'] - p2['bp_saved'],
                p1['bp_conv'] - p2['bp_conv'],
                
                # NEW: Head-to-head spread
                h2h['avg'],
                h2h['count'],
                
                # NEW: Match context
                round_level,
                tourn_level,
                round_level * tourn_level,  # Interaction: late round at big tournament
            ]
            
            X.append(features)
            y.append(spread)
            
            # Classification target: close (<4 games) vs decisive (4+ games)
            y_class.append(1 if abs(spread) >= 4 else 0)
    
    return np.array(X), np.array(y), np.array(y_class)


def test_improvements():
    """Test various improvements"""
    data = load_data()
    print(f"Loaded {data['player_count']} players")
    
    X, y, y_class = build_features_with_context(data)
    print(f"Built {len(X)} samples")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    _, _, yc_train, yc_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    print("\n" + "="*70)
    print("IMPROVEMENT 1: Adding H2H spread + Match Context")
    print("="*70)
    
    model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    model.fit(X_train_s, y_train)
    
    pred = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, pred)
    within_2 = np.mean(np.abs(pred - y_test) <= 2)
    within_3 = np.mean(np.abs(pred - y_test) <= 3)
    
    print(f"MAE: {mae:.3f} games")
    print(f"Within 2 games: {within_2:.1%}")
    print(f"Within 3 games: {within_3:.1%}")
    
    # Feature importance
    feature_names = [
        'elo_diff', 'surface_elo_diff', 'match_quality',
        'spread_diff', 'p1_spread', 'p2_spread', 'volatility', 'dominance',
        'straight_set_diff', '3set_rate', 'tb_rate', 'match_length',
        'win_pct_diff', 'serve_won_diff',
        'ace_diff', 'df_diff', 'bp_saved_diff', 'bp_conv_diff',
        'h2h_spread', 'h2h_count',
        'round_level', 'tourn_level', 'round_x_tourn'
    ]
    
    importances = model.feature_importances_
    print("\nTop 10 Features:")
    for idx in np.argsort(importances)[::-1][:10]:
        print(f"  {feature_names[idx]}: {importances[idx]:.3f}")
    
    print("\n" + "="*70)
    print("IMPROVEMENT 2: Two-Stage Prediction")
    print("(First predict close/decisive, then spread)")
    print("="*70)
    
    # Stage 1: Classify close vs decisive
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    clf.fit(X_train_s, yc_train)
    
    class_pred = clf.predict(X_test_s)
    class_acc = accuracy_score(yc_test, class_pred)
    print(f"Close/Decisive Classification Accuracy: {class_acc:.1%}")
    
    # Stage 2: Separate models for close and decisive matches
    close_mask_train = yc_train == 0
    decisive_mask_train = yc_train == 1
    
    model_close = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model_decisive = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    
    if close_mask_train.sum() > 10:
        model_close.fit(X_train_s[close_mask_train], y_train[close_mask_train])
    if decisive_mask_train.sum() > 10:
        model_decisive.fit(X_train_s[decisive_mask_train], y_train[decisive_mask_train])
    
    # Predict using two-stage
    pred_2stage = np.zeros(len(y_test))
    for i in range(len(y_test)):
        if class_pred[i] == 0:  # Close match
            pred_2stage[i] = model_close.predict(X_test_s[i:i+1])[0]
        else:  # Decisive match
            pred_2stage[i] = model_decisive.predict(X_test_s[i:i+1])[0]
    
    mae_2stage = mean_absolute_error(y_test, pred_2stage)
    within_2_2stage = np.mean(np.abs(pred_2stage - y_test) <= 2)
    within_3_2stage = np.mean(np.abs(pred_2stage - y_test) <= 3)
    
    print(f"Two-Stage MAE: {mae_2stage:.3f} games")
    print(f"Within 2 games: {within_2_2stage:.1%}")
    print(f"Within 3 games: {within_3_2stage:.1%}")
    
    print("\n" + "="*70)
    print("IMPROVEMENT 3: Predict Total Games Instead")
    print("="*70)
    
    # Total games might be easier to predict
    y_total = np.array([parse_score(m.get('score', ''))['total'] 
                        for p in data['players'].values() 
                        for m in p['matches'] 
                        if parse_score(m.get('score', ''))])[:len(y)]
    
    if len(y_total) == len(y):
        y_total_train, y_total_test = y_total[:len(y_train)], y_total[len(y_train):len(y_train)+len(y_test)]
        
        model_total = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
        model_total.fit(X_train_s, y_total_train)
        
        pred_total = model_total.predict(X_test_s)
        mae_total = mean_absolute_error(y_total_test, pred_total)
        
        print(f"Total Games MAE: {mae_total:.2f} games")
        print(f"Average total games: {np.mean(y_total_test):.1f}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Baseline (current model):        MAE = ~1.96 games")
    print(f"+ H2H spread + Context:          MAE = {mae:.3f} games")
    print(f"+ Two-Stage Prediction:          MAE = {mae_2stage:.3f} games")


if __name__ == '__main__':
    test_improvements()
