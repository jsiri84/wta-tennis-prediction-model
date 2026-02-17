"""
Experiment v3: Test each ELO-level feature individually
"""

import json
import re
import numpy as np
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

# Import from v2 experiment
from elo_level_experiment_v2 import (
    load_data, parse_score, calculate_rolling_stats, calculate_surface_form,
    calculate_blended_stats, calculate_head_to_head,
    calculate_vs_strong_opponents, calculate_elo_adjusted_form, calculate_level_jump
)


def build_data_with_selected_features(data, use_vs_strong=False, use_adjusted=False, use_level_jump=False):
    """Build training data with selectable new features"""
    
    player_data = {}
    for name, info in data['players'].items():
        player_data[name] = {
            'elo_overall': info.get('elo_overall', 1500),
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
    
    X = []
    y = []
    y_spread = []
    y_total = []
    
    for player_name, pdata in player_data.items():
        matches = pdata['matches']
        
        for i, match in enumerate(matches):
            historical_matches = matches[i+1:] if i+1 < len(matches) else []
            
            if len(historical_matches) < 5:
                continue
            
            opp_name = match.get('opponent', '')
            opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
            
            if not opp_key or opp_key not in player_data:
                continue
            
            opp_data = player_data[opp_key]
            match_date = match.get('date', '')
            opp_historical = [m for m in opp_data['matches'] if m.get('date', '') < match_date]
            
            if len(opp_historical) < 5:
                continue
            
            surface = match.get('surface', 'Hard').lower()
            surface_elo_key = f'elo_{surface}' if surface in ['hard', 'clay', 'grass'] else 'elo_hard'
            
            p1_form = calculate_rolling_stats(historical_matches, 15)
            p2_form = calculate_rolling_stats(opp_historical, 15)
            
            if not p1_form or not p2_form:
                continue
            
            p1_surf_form = calculate_surface_form(historical_matches, surface, 15)
            p2_surf_form = calculate_surface_form(opp_historical, surface, 15)
            p1_blended = calculate_blended_stats(historical_matches, surface, 15, 0.6)
            p2_blended = calculate_blended_stats(opp_historical, surface, 15, 0.6)
            
            if not p1_blended or not p2_blended:
                continue
            
            p1_h2h = calculate_head_to_head(historical_matches, opp_name)
            
            # Base features
            features = [
                pdata['elo_overall'] - opp_data['elo_overall'],
                pdata.get(surface_elo_key, 1500) - opp_data.get(surface_elo_key, 1500),
                pdata['elo_overall'],
                opp_data['elo_overall'],
                p1_form['win_pct'] - p2_form['win_pct'],
                p1_form['avg_dr'] - p2_form['avg_dr'],
                p1_form['win_pct'],
                p2_form['win_pct'],
                p1_form['first_in_pct'] - p2_form['first_in_pct'],
                p1_form['first_won_pct'] - p2_form['first_won_pct'],
                p1_form['ace_pct'] - p2_form['ace_pct'],
                p1_form['df_pct'] - p2_form['df_pct'],
                p1_form['bp_saved_pct'] - p2_form['bp_saved_pct'],
                p1_form['rpw_pct'] - p2_form['rpw_pct'],
                p1_form['bp_conv_pct'] - p2_form['bp_conv_pct'],
                (p1_surf_form['win_pct'] if p1_surf_form else 0.5) - (p2_surf_form['win_pct'] if p2_surf_form else 0.5),
                p1_blended['first_won_pct'] - p2_blended['first_won_pct'],
                p1_blended['ace_pct'] - p2_blended['ace_pct'],
                p1_blended['bp_saved_pct'] - p2_blended['bp_saved_pct'],
                p1_blended['rpw_pct'] - p2_blended['rpw_pct'],
                p1_blended['bp_conv_pct'] - p2_blended['bp_conv_pct'],
                p1_blended['win_pct'] - p2_blended['win_pct'],
                p1_blended['avg_dr'] - p2_blended['avg_dr'],
                p1_blended.get('surface_match_count', 0) - p2_blended.get('surface_match_count', 0),
                p1_blended.get('surface_match_count', 0),
                p2_blended.get('surface_match_count', 0),
                p1_h2h['diff'],
                p1_h2h['win_pct'],
                p1_h2h['total'],
            ]
            
            # Optional new features
            if use_vs_strong:
                p1_vs = calculate_vs_strong_opponents(historical_matches, player_data, name_lookup)
                p2_vs = calculate_vs_strong_opponents(opp_historical, player_data, name_lookup)
                features.extend([
                    p1_vs['win_pct_vs_strong'] - p2_vs['win_pct_vs_strong'],
                    p1_vs['matches_vs_strong'] - p2_vs['matches_vs_strong'],
                ])
            
            if use_adjusted:
                p1_adj = calculate_elo_adjusted_form(historical_matches, player_data, name_lookup)
                p2_adj = calculate_elo_adjusted_form(opp_historical, player_data, name_lookup)
                features.extend([
                    p1_adj['adjusted_win_pct'] - p2_adj['adjusted_win_pct'],
                    p1_adj['avg_opponent_elo'] - p2_adj['avg_opponent_elo'],
                ])
            
            if use_level_jump:
                p1_jump = calculate_level_jump(historical_matches, player_data, name_lookup, opp_data['elo_overall'])
                p2_jump = calculate_level_jump(opp_historical, player_data, name_lookup, pdata['elo_overall'])
                features.extend([
                    p1_jump['level_jump'] - p2_jump['level_jump'],
                    p1_jump['level_jump_pct'] - p2_jump['level_jump_pct'],
                ])
            
            # Labels
            p1_games, p2_games, total = parse_score(match.get('score', ''))
            if p1_games is None or total < 12 or total > 50:
                continue
            
            X.append(features)
            y.append(1 if match.get('result') == 'W' else 0)
            y_spread.append(p1_games - p2_games)
            y_total.append(total)
    
    return np.array(X), np.array(y), np.array(y_spread), np.array(y_total)


def evaluate(X, y, y_spread, y_total):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Winner
    winner_model = LogisticRegression(max_iter=1000, C=0.1)
    winner_proba = cross_val_predict(winner_model, X_scaled, y, cv=kf, method='predict_proba')[:, 1]
    winner_acc = accuracy_score(y, (winner_proba > 0.5).astype(int))
    winner_auc = roc_auc_score(y, winner_proba)
    
    # Spread
    spread_model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    spread_pred = cross_val_predict(spread_model, X_scaled, y_spread, cv=kf)
    spread_mae = mean_absolute_error(y_spread, spread_pred)
    
    # Total
    total_model = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    total_pred = cross_val_predict(total_model, X_scaled, y_total, cv=kf)
    total_mae = mean_absolute_error(y_total, total_pred)
    ou_acc = ((y_total > 21.5) == (total_pred > 21.5)).mean()
    
    return winner_acc, winner_auc, spread_mae, total_mae, ou_acc


def main():
    print("="*70)
    print("ELO LEVEL FEATURES - INDIVIDUAL FEATURE TEST")
    print("="*70)
    print()
    
    data = load_data()
    print(f"Loaded {data['player_count']} players")
    print()
    
    configs = [
        ('Baseline', False, False, False),
        ('+ Vs Strong Only', True, False, False),
        ('+ Adjusted Form Only', False, True, False),
        ('+ Level Jump Only', False, False, True),
        ('+ All Three', True, True, True),
    ]
    
    results = []
    
    for name, vs_strong, adjusted, level_jump in configs:
        print(f"Testing: {name}...")
        X, y, y_spread, y_total = build_data_with_selected_features(
            data, use_vs_strong=vs_strong, use_adjusted=adjusted, use_level_jump=level_jump
        )
        print(f"  {X.shape[0]} samples, {X.shape[1]} features")
        
        acc, auc, spread, total, ou = evaluate(X, y, y_spread, y_total)
        results.append((name, X.shape[1], acc, auc, spread, total, ou))
    
    # Print results
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()
    print(f"{'Configuration':<25} {'Feat':>5} {'Win Acc':>8} {'AUC':>7} {'Spread':>7} {'Total':>7} {'O/U':>7}")
    print("-"*70)
    
    baseline = results[0]
    for name, feat, acc, auc, spread, total, ou in results:
        print(f"{name:<25} {feat:>5} {acc:>7.1%} {auc:>7.3f} {spread:>7.2f} {total:>7.2f} {ou:>6.1%}")
    
    # Deltas
    print()
    print("DELTA FROM BASELINE:")
    print("-"*70)
    
    for name, feat, acc, auc, spread, total, ou in results[1:]:
        acc_d = acc - baseline[2]
        auc_d = auc - baseline[3]
        spread_d = spread - baseline[4]
        total_d = total - baseline[5]
        ou_d = ou - baseline[6]
        
        print(f"{name:<25}       {acc_d:>+6.2%} {auc_d:>+7.4f} {spread_d:>+7.3f} {total_d:>+7.3f} {ou_d:>+6.2%}")
    
    # Best for each metric
    print()
    print("="*70)
    print("BEST CONFIGURATION FOR EACH METRIC:")
    print("="*70)
    
    best_acc = max(results, key=lambda x: x[2])
    best_auc = max(results, key=lambda x: x[3])
    best_spread = min(results, key=lambda x: x[4])
    best_total = min(results, key=lambda x: x[5])
    best_ou = max(results, key=lambda x: x[6])
    
    print(f"Winner Accuracy: {best_acc[0]} ({best_acc[2]:.1%})")
    print(f"Winner AUC:      {best_auc[0]} ({best_auc[3]:.3f})")
    print(f"Spread MAE:      {best_spread[0]} ({best_spread[4]:.2f})")
    print(f"Total MAE:       {best_total[0]} ({best_total[5]:.2f})")
    print(f"O/U 21.5:        {best_ou[0]} ({best_ou[6]:.1%})")


if __name__ == '__main__':
    main()
