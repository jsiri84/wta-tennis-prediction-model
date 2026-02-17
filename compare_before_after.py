"""
Compare model performance BEFORE and AFTER new features
Focus on SPREAD model since new features were added there
"""

import json
import re
import numpy as np
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from unified_model import (
    load_data, calculate_player_stats, calculate_surface_stats, calculate_h2h,
    get_tournament_level, get_round_level,
    calculate_vs_strong_opponents, calculate_elo_adjusted_form, calculate_level_jump,
    calculate_performance_at_level, calculate_form_vs_expected,
    calculate_extreme_momentum, calculate_tournament_performance,
    parse_score
)


def build_data(data, include_new_features=False):
    """Build training data with/without new features"""
    
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
    
    X_spread = []
    y_spread = []
    
    for player_name, pdata in player_data.items():
        for i, match in enumerate(pdata['matches']):
            if i < 5:
                continue
            
            parsed = parse_score(match.get('score', ''))
            if not parsed:
                continue
            
            spread = parsed['spread']
            total = parsed['total']
            
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
            
            p1 = calculate_player_stats(hist)
            p2 = calculate_player_stats(opp_hist)
            p1_surf = calculate_surface_stats(hist, surface)
            p2_surf = calculate_surface_stats(opp_hist, surface)
            h2h = calculate_h2h(hist, opp_full)
            
            if not p1 or not p2:
                continue
            
            # Original features (BEFORE new features)
            p1_vs_strong = calculate_vs_strong_opponents(hist, player_data, name_lookup)
            p2_vs_strong = calculate_vs_strong_opponents(opp_hist, player_data, name_lookup)
            p1_adj_form = calculate_elo_adjusted_form(hist, player_data, name_lookup)
            p2_adj_form = calculate_elo_adjusted_form(opp_hist, player_data, name_lookup)
            p1_level_jump = calculate_level_jump(hist, player_data, name_lookup, opp_data['elo'])
            p2_level_jump = calculate_level_jump(opp_hist, player_data, name_lookup, pdata['elo'])
            p1_at_level = calculate_performance_at_level(hist, player_data, name_lookup, opp_data['elo'])
            p2_at_level = calculate_performance_at_level(opp_hist, player_data, name_lookup, pdata['elo'])
            p1_form_vs_exp = calculate_form_vs_expected(hist, pdata['elo'], pdata.get(surface_key, 1500), player_data, name_lookup, surface, 3)
            p2_form_vs_exp = calculate_form_vs_expected(opp_hist, opp_data['elo'], opp_data.get(surface_key, 1500), player_data, name_lookup, surface, 3)
            
            # ORIGINAL SPREAD FEATURES (35 features - same as before)
            spread_features = [
                pdata['elo'] - opp_data['elo'],
                pdata.get(surface_key, 1500) - opp_data.get(surface_key, 1500),
                p1['avg_spread'] - p2['avg_spread'],
                p1['avg_spread'],
                p2['avg_spread'],
                p1['spread_std'] + p2['spread_std'],
                abs(p1['avg_spread']) + abs(p2['avg_spread']),
                p1['win_pct'] - p2['win_pct'],
                p1['avg_dr'] - p2['avg_dr'],
                (p1_surf['win_pct'] if p1_surf else 0.5) - (p2_surf['win_pct'] if p2_surf else 0.5),
                p1['first_won_pct'] - p2['first_won_pct'],
                p1['ace_pct'] - p2['ace_pct'],
                p1['second_won_pct'] - p2['second_won_pct'],
                p1['bp_saved_pct'] - p2['bp_saved_pct'],
                p1['rpw_pct'] - p2['rpw_pct'],
                p1['bp_conv_pct'] - p2['bp_conv_pct'],
                p1['straight_set_rate'] - p2['straight_set_rate'],
                abs(p1['win_pct'] - p2['win_pct']),
                h2h['diff'],
                h2h['win_pct'],
                h2h['total'],
                p1_vs_strong['win_pct_vs_strong'] - p2_vs_strong['win_pct_vs_strong'],
                p1_vs_strong['matches_vs_strong'] - p2_vs_strong['matches_vs_strong'],
                p1_adj_form['adjusted_win_pct'] - p2_adj_form['adjusted_win_pct'],
                p1_adj_form['avg_opponent_elo'] - p2_adj_form['avg_opponent_elo'],
                p1_level_jump['level_jump'] - p2_level_jump['level_jump'],
                p1_level_jump['level_jump_pct'] - p2_level_jump['level_jump_pct'],
                p1_at_level['win_pct_at_level'] - p2_at_level['win_pct_at_level'],
                p1_at_level['avg_spread_at_level'] - p2_at_level['avg_spread_at_level'],
                p1_at_level['serve_pct_at_level'] - p2_at_level['serve_pct_at_level'],
                p1_at_level['return_pct_at_level'] - p2_at_level['return_pct_at_level'],
                (p1_form_vs_exp['form_diff'] if p1_form_vs_exp else 0) - (p2_form_vs_exp['form_diff'] if p2_form_vs_exp else 0),
                (p1_form_vs_exp['surface_form_diff'] if p1_form_vs_exp else 0) - (p2_form_vs_exp['surface_form_diff'] if p2_form_vs_exp else 0),
                (p1_form_vs_exp['spread_form_diff'] if p1_form_vs_exp else 0) - (p2_form_vs_exp['spread_form_diff'] if p2_form_vs_exp else 0),
                (p1_form_vs_exp['streak'] if p1_form_vs_exp else 0) - (p2_form_vs_exp['streak'] if p2_form_vs_exp else 0),
            ]
            
            # NEW FEATURES (10 additional)
            if include_new_features:
                p1_extreme = calculate_extreme_momentum(hist, player_data, name_lookup, opp_data['elo'], pdata['elo'])
                p2_extreme = calculate_extreme_momentum(opp_hist, player_data, name_lookup, pdata['elo'], opp_data['elo'])
                p1_tourn = calculate_tournament_performance(hist, player_data, name_lookup, 4)
                p2_tourn = calculate_tournament_performance(opp_hist, player_data, name_lookup, 4)
                
                spread_features.extend([
                    p1_extreme['extreme_signal'] - p2_extreme['extreme_signal'],
                    p1_extreme['raw_cascade'] - p2_extreme['raw_cascade'],
                    (p1_tourn['avg_opp_elo_beaten'] if p1_tourn else 0) - (p2_tourn['avg_opp_elo_beaten'] if p2_tourn else 0),
                    (p1_tourn['max_opp_elo_beaten'] if p1_tourn else 0) - (p2_tourn['max_opp_elo_beaten'] if p2_tourn else 0),
                    (p1_tourn['avg_spread_in_tourn'] if p1_tourn else 0) - (p2_tourn['avg_spread_in_tourn'] if p2_tourn else 0),
                    (p1_tourn['game_dominance'] if p1_tourn else 1) - (p2_tourn['game_dominance'] if p2_tourn else 1),
                    (p1_tourn['clean_sets'] if p1_tourn else 0) - (p2_tourn['clean_sets'] if p2_tourn else 0),
                    (p1_tourn['serve_pct_in_tourn'] if p1_tourn else 55) - (p2_tourn['serve_pct_in_tourn'] if p2_tourn else 55),
                    (p1_tourn['return_pct_in_tourn'] if p1_tourn else 35) - (p2_tourn['return_pct_in_tourn'] if p2_tourn else 35),
                    (p1_tourn['bp_conv_in_tourn'] if p1_tourn else 40) - (p2_tourn['bp_conv_in_tourn'] if p2_tourn else 40),
                ])
            
            if total < 12 or total > 50:
                continue
            
            X_spread.append(spread_features)
            y_spread.append(spread)
    
    return np.array(X_spread), np.array(y_spread)


def evaluate_spread(X, y_spread):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    pred = cross_val_predict(model, X_scaled, y_spread, cv=kf)
    mae = mean_absolute_error(y_spread, pred)
    
    return mae


def main():
    print("="*70)
    print("SPREAD MODEL COMPARISON: BEFORE vs AFTER NEW FEATURES")
    print("="*70)
    print()
    print("Note: Winner prediction uses a SEPARATE model (TennisPredictor)")
    print("      that was not changed. These new features only affect SPREAD.")
    print()
    
    data = load_data()
    print(f"Loaded {data['player_count']} players")
    print()
    
    print("Building BEFORE (without new features)...")
    X_before, y_spread = build_data(data, include_new_features=False)
    print(f"  {X_before.shape[0]} samples, {X_before.shape[1]} features")
    before_mae = evaluate_spread(X_before, y_spread)
    
    print("\nBuilding AFTER (with new features)...")
    X_after, y_spread = build_data(data, include_new_features=True)
    print(f"  {X_after.shape[0]} samples, {X_after.shape[1]} features")
    after_mae = evaluate_spread(X_after, y_spread)
    
    print()
    print("="*70)
    print("SPREAD MODEL RESULTS")
    print("="*70)
    print()
    print(f"{'Config':<40} {'Features':>10} {'Spread MAE':>12}")
    print("-"*65)
    print(f"{'BEFORE (original features)':<40} {X_before.shape[1]:>10} {before_mae:>12.4f}")
    print(f"{'AFTER (+ extreme + tournament)':<40} {X_after.shape[1]:>10} {after_mae:>12.4f}")
    print(f"{'CHANGE':<40} {'+10':>10} {after_mae - before_mae:>+12.5f}")
    
    print()
    if after_mae < before_mae:
        print("=> IMPROVEMENT in spread prediction!")
    else:
        print("=> No improvement in spread prediction")


if __name__ == '__main__':
    main()
