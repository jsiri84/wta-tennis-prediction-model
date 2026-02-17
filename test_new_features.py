"""
Test new features: Extreme Momentum + Tournament Performance
"""

import json
import re
import numpy as np
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error


def load_data():
    with open('all_players_matches.json', 'r') as f:
        return json.load(f)


def calculate_extreme_momentum(matches, player_data, name_lookup, current_opp_elo, base_elo):
    """Extreme momentum feature"""
    if not matches:
        return {'extreme_signal': 0, 'raw_cascade': 0}
    
    last_match = matches[0]
    opp_name = last_match.get('opponent', '')
    opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
    
    if not opp_key or opp_key not in player_data:
        return {'extreme_signal': 0, 'raw_cascade': 0}
    
    last_opp_elo = player_data[opp_key].get('elo', 1500)
    won_last = last_match.get('result') == 'W'
    
    if not won_last:
        return {'extreme_signal': 0, 'raw_cascade': 0}
    
    score = last_match.get('score', '')
    sets = re.findall(r'(\d+)-(\d+)', score)
    if not sets:
        return {'extreme_signal': 0, 'raw_cascade': 0}
    
    p1 = sum(int(s[0]) for s in sets)
    p2 = sum(int(s[1]) for s in sets)
    last_spread = p1 - p2
    
    scalp = last_opp_elo - base_elo
    step_down = last_opp_elo - current_opp_elo
    
    if scalp > 0 and step_down > 0:
        raw_cascade = (scalp / 100) * (step_down / 100) * (last_spread / 5)
    else:
        raw_cascade = 0
    
    extreme_signal = 0
    if scalp >= 100 and step_down >= 150:
        extreme_signal = 1
        if last_spread >= 4:
            extreme_signal = 2
            if scalp >= 150 and step_down >= 200:
                extreme_signal = 3
    
    return {'extreme_signal': extreme_signal, 'raw_cascade': raw_cascade}


def calculate_tournament_performance(matches, player_data, name_lookup, max_rounds=4):
    """Tournament performance feature"""
    if not matches:
        return None
    
    recent = matches[:max_rounds]
    opponents_beaten = []
    spreads = []
    clean_sets_count = 0
    total_games_won = 0
    total_games_lost = 0
    wins = 0
    serve_pcts = []
    return_pcts = []
    bp_conv_pcts = []
    
    for match in recent:
        opp_name = match.get('opponent', '')
        opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
        won = match.get('result') == 'W'
        
        if opp_key and opp_key in player_data:
            opp_elo = player_data[opp_key].get('elo', 1500)
        else:
            opp_elo = 1600
        
        score = match.get('score', '')
        sets = re.findall(r'(\d+)-(\d+)', score)
        
        if sets:
            p1_games = sum(int(s[0]) for s in sets)
            p2_games = sum(int(s[1]) for s in sets)
            spread = p1_games - p2_games
            
            total_games_won += p1_games
            total_games_lost += p2_games
            spreads.append(spread if won else -spread)
            
            if won:
                for s in sets:
                    if int(s[0]) >= 6 and int(s[1]) <= 2 and int(s[0]) > int(s[1]):
                        clean_sets_count += 1
            
            if won:
                wins += 1
                opponents_beaten.append(opp_elo)
        
        serve = match.get('serve', {})
        if serve:
            fw = serve.get('first_won_pct') or 0
            sw = serve.get('second_won_pct') or 0
            if fw or sw:
                serve_pcts.append(fw * 0.6 + sw * 0.4)
        
        ret = match.get('return', {})
        if ret:
            if ret.get('rpw_pct'):
                return_pcts.append(ret['rpw_pct'])
            if ret.get('bp_conv_pct'):
                bp_conv_pcts.append(ret['bp_conv_pct'])
    
    if not recent:
        return None
    
    if total_games_lost > 0:
        game_dominance = total_games_won / total_games_lost
    else:
        game_dominance = 2.0
    
    return {
        'wins_in_tournament': wins,
        'avg_opp_elo_beaten': np.mean(opponents_beaten) if opponents_beaten else 0,
        'max_opp_elo_beaten': max(opponents_beaten) if opponents_beaten else 0,
        'avg_spread_in_tourn': np.mean(spreads) if spreads else 0,
        'clean_sets': clean_sets_count,
        'game_dominance': game_dominance,
        'serve_pct_in_tourn': np.mean(serve_pcts) if serve_pcts else 55,
        'return_pct_in_tourn': np.mean(return_pcts) if return_pcts else 35,
        'bp_conv_in_tourn': np.mean(bp_conv_pcts) if bp_conv_pcts else 40,
    }


def build_data(data, include_extreme=False, include_tournament=False):
    """Build training data with optional new features"""
    
    player_data = {}
    for name, info in data['players'].items():
        player_data[name] = {
            'elo': info.get('elo_overall', 1500),
            'elo_hard': info.get('elo_hard', 1500),
            'matches': sorted(info.get('matches', []), key=lambda x: x.get('date', ''), reverse=True)
        }
    
    name_lookup = {}
    for name in player_data:
        name_lookup[name.lower()] = name
        name_lookup[name.lower().replace(' ', '')] = name
        parts = name.split()
        if len(parts) > 1:
            name_lookup[parts[-1].lower()] = name
    
    X = []
    y = []
    y_spread = []
    y_total = []
    
    for player_name, pdata in player_data.items():
        matches = pdata['matches']
        
        for i, match in enumerate(matches):
            historical = matches[i+1:] if i+1 < len(matches) else []
            
            if len(historical) < 5:
                continue
            
            opp_name = match.get('opponent', '')
            opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
            
            if not opp_key or opp_key not in player_data:
                continue
            
            opp_data = player_data[opp_key]
            match_date = match.get('date', '')
            opp_hist = [m for m in opp_data['matches'] if m.get('date', '') < match_date]
            
            if len(opp_hist) < 5:
                continue
            
            surface = match.get('surface', 'Hard').lower()
            surface_key = f'elo_{surface}' if surface in ['hard', 'clay', 'grass'] else 'elo_hard'
            
            # Base features
            features = [
                pdata['elo'] - opp_data['elo'],
                pdata.get(surface_key, 1500) - opp_data.get(surface_key, 1500),
            ]
            
            # Extreme momentum
            if include_extreme:
                p1_ext = calculate_extreme_momentum(historical, player_data, name_lookup, opp_data['elo'], pdata['elo'])
                p2_ext = calculate_extreme_momentum(opp_hist, player_data, name_lookup, pdata['elo'], opp_data['elo'])
                
                features.extend([
                    p1_ext['extreme_signal'] - p2_ext['extreme_signal'],
                    p1_ext['raw_cascade'] - p2_ext['raw_cascade'],
                ])
            
            # Tournament performance
            if include_tournament:
                p1_tourn = calculate_tournament_performance(historical, player_data, name_lookup)
                p2_tourn = calculate_tournament_performance(opp_hist, player_data, name_lookup)
                
                features.extend([
                    (p1_tourn['avg_opp_elo_beaten'] if p1_tourn else 0) - (p2_tourn['avg_opp_elo_beaten'] if p2_tourn else 0),
                    (p1_tourn['max_opp_elo_beaten'] if p1_tourn else 0) - (p2_tourn['max_opp_elo_beaten'] if p2_tourn else 0),
                    (p1_tourn['avg_spread_in_tourn'] if p1_tourn else 0) - (p2_tourn['avg_spread_in_tourn'] if p2_tourn else 0),
                    (p1_tourn['game_dominance'] if p1_tourn else 1) - (p2_tourn['game_dominance'] if p2_tourn else 1),
                    (p1_tourn['clean_sets'] if p1_tourn else 0) - (p2_tourn['clean_sets'] if p2_tourn else 0),
                    (p1_tourn['serve_pct_in_tourn'] if p1_tourn else 55) - (p2_tourn['serve_pct_in_tourn'] if p2_tourn else 55),
                    (p1_tourn['return_pct_in_tourn'] if p1_tourn else 35) - (p2_tourn['return_pct_in_tourn'] if p2_tourn else 35),
                    (p1_tourn['bp_conv_in_tourn'] if p1_tourn else 40) - (p2_tourn['bp_conv_in_tourn'] if p2_tourn else 40),
                ])
            
            # Labels
            score = match.get('score', '')
            sets = re.findall(r'(\d+)-(\d+)', score)
            if not sets:
                continue
            p1_games = sum(int(s[0]) for s in sets)
            p2_games = sum(int(s[1]) for s in sets)
            total = p1_games + p2_games
            spread = p1_games - p2_games
            
            if total < 12 or total > 50:
                continue
            
            X.append(features)
            y.append(1 if match.get('result') == 'W' else 0)
            y_spread.append(spread)
            y_total.append(total)
    
    return np.array(X), np.array(y), np.array(y_spread), np.array(y_total)


def evaluate(X, y, y_spread, y_total):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Winner
    winner_model = LogisticRegression(max_iter=1000, C=0.1)
    proba = cross_val_predict(winner_model, X_scaled, y, cv=kf, method='predict_proba')[:, 1]
    winner_acc = accuracy_score(y, (proba > 0.5).astype(int))
    winner_auc = roc_auc_score(y, proba)
    
    # Spread
    spread_model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    spread_pred = cross_val_predict(spread_model, X_scaled, y_spread, cv=kf)
    spread_mae = mean_absolute_error(y_spread, spread_pred)
    
    # Total
    total_model = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    total_pred = cross_val_predict(total_model, X_scaled, y_total, cv=kf)
    total_mae = mean_absolute_error(y_total, total_pred)
    
    # O/U accuracy at 21.5
    ou_accuracy = np.mean(((total_pred > 21.5) == (y_total > 21.5)))
    
    return winner_acc, winner_auc, spread_mae, total_mae, ou_accuracy


def main():
    print("="*70)
    print("TESTING NEW FEATURES: EXTREME MOMENTUM + TOURNAMENT PERFORMANCE")
    print("="*70)
    print()
    
    data = load_data()
    print(f"Loaded {data['player_count']} players")
    print()
    
    # Baseline
    print("Building BASELINE (ELO only)...")
    X_base, y, y_spread, y_total = build_data(data, include_extreme=False, include_tournament=False)
    print(f"  {X_base.shape[0]} samples, {X_base.shape[1]} features")
    base_acc, base_auc, base_spread, base_total, base_ou = evaluate(X_base, y, y_spread, y_total)
    
    # + Extreme only
    print("\nBuilding + EXTREME MOMENTUM...")
    X_ext, y, y_spread, y_total = build_data(data, include_extreme=True, include_tournament=False)
    print(f"  {X_ext.shape[0]} samples, {X_ext.shape[1]} features")
    ext_acc, ext_auc, ext_spread, ext_total, ext_ou = evaluate(X_ext, y, y_spread, y_total)
    
    # + Tournament only
    print("\nBuilding + TOURNAMENT PERFORMANCE...")
    X_tourn, y, y_spread, y_total = build_data(data, include_extreme=False, include_tournament=True)
    print(f"  {X_tourn.shape[0]} samples, {X_tourn.shape[1]} features")
    tourn_acc, tourn_auc, tourn_spread, tourn_total, tourn_ou = evaluate(X_tourn, y, y_spread, y_total)
    
    # + Both
    print("\nBuilding + BOTH FEATURES...")
    X_both, y, y_spread, y_total = build_data(data, include_extreme=True, include_tournament=True)
    print(f"  {X_both.shape[0]} samples, {X_both.shape[1]} features")
    both_acc, both_auc, both_spread, both_total, both_ou = evaluate(X_both, y, y_spread, y_total)
    
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()
    print(f"{'Config':<25} {'Features':>8} {'Win Acc':>10} {'Win AUC':>10} {'Spread':>10} {'Total':>10} {'O/U 21.5':>10}")
    print("-"*90)
    print(f"{'Baseline (ELO)':<25} {X_base.shape[1]:>8} {base_acc:>9.1%} {base_auc:>10.3f} {base_spread:>10.3f} {base_total:>10.3f} {base_ou:>9.1%}")
    print(f"{'+ Extreme Momentum':<25} {X_ext.shape[1]:>8} {ext_acc:>9.1%} {ext_auc:>10.3f} {ext_spread:>10.3f} {ext_total:>10.3f} {ext_ou:>9.1%}")
    print(f"{'+ Tournament Perf':<25} {X_tourn.shape[1]:>8} {tourn_acc:>9.1%} {tourn_auc:>10.3f} {tourn_spread:>10.3f} {tourn_total:>10.3f} {tourn_ou:>9.1%}")
    print(f"{'+ BOTH':<25} {X_both.shape[1]:>8} {both_acc:>9.1%} {both_auc:>10.3f} {both_spread:>10.3f} {both_total:>10.3f} {both_ou:>9.1%}")
    
    print()
    print("="*70)
    print("CHANGES FROM BASELINE")
    print("="*70)
    print()
    print(f"{'Config':<25} {'Win Acc':>12} {'Win AUC':>12} {'Spread MAE':>12} {'Total MAE':>12} {'O/U Acc':>12}")
    print("-"*90)
    print(f"{'+ Extreme Momentum':<25} {(ext_acc-base_acc)*100:>+11.2f}% {ext_auc-base_auc:>+12.4f} {ext_spread-base_spread:>+12.4f} {ext_total-base_total:>+12.4f} {(ext_ou-base_ou)*100:>+11.2f}%")
    print(f"{'+ Tournament Perf':<25} {(tourn_acc-base_acc)*100:>+11.2f}% {tourn_auc-base_auc:>+12.4f} {tourn_spread-base_spread:>+12.4f} {tourn_total-base_total:>+12.4f} {(tourn_ou-base_ou)*100:>+11.2f}%")
    print(f"{'+ BOTH':<25} {(both_acc-base_acc)*100:>+11.2f}% {both_auc-base_auc:>+12.4f} {both_spread-base_spread:>+12.4f} {both_total-base_total:>+12.4f} {(both_ou-base_ou)*100:>+11.2f}%")


if __name__ == '__main__':
    main()
