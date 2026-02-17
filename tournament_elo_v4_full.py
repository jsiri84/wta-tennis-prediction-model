"""
Test Extreme Momentum on FULL MODEL
"""

import json
import re
import numpy as np
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from elo_level_experiment_v2 import (
    load_data, calculate_rolling_stats, calculate_surface_form,
    calculate_head_to_head,
    calculate_vs_strong_opponents, calculate_elo_adjusted_form, calculate_level_jump
)
from dominance_experiment_v4 import calculate_performance_at_level
from form_experiment_v3 import calculate_form_vs_expected


def calculate_extreme_momentum(matches, player_data, name_lookup, current_opp_elo, base_elo):
    """Calculate momentum ONLY in extreme cases."""
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
    
    # Raw cascade
    if scalp > 0 and step_down > 0:
        raw_cascade = (scalp / 100) * (step_down / 100) * (last_spread / 5)
    else:
        raw_cascade = 0
    
    # Extreme signal levels
    extreme_signal = 0
    if scalp >= 100 and step_down >= 150:
        extreme_signal = 1
        if last_spread >= 4:
            extreme_signal = 2
            if scalp >= 150 and step_down >= 200:
                extreme_signal = 3
    
    return {
        'extreme_signal': extreme_signal,
        'raw_cascade': raw_cascade,
    }


def build_full_data(data, include_extreme=False):
    """Build full training data"""
    
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
    
    X = []
    y_spread = []
    
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
            
            p1_form = calculate_rolling_stats(historical, 15)
            p2_form = calculate_rolling_stats(opp_hist, 15)
            
            if not p1_form or not p2_form:
                continue
            
            p1_surf = calculate_surface_form(historical, surface, 15)
            p2_surf = calculate_surface_form(opp_hist, surface, 15)
            p1_h2h = calculate_head_to_head(historical, opp_name)
            
            p1_vs_strong = calculate_vs_strong_opponents(historical, player_data, name_lookup)
            p2_vs_strong = calculate_vs_strong_opponents(opp_hist, player_data, name_lookup)
            p1_adj = calculate_elo_adjusted_form(historical, player_data, name_lookup)
            p2_adj = calculate_elo_adjusted_form(opp_hist, player_data, name_lookup)
            p1_jump = calculate_level_jump(historical, player_data, name_lookup, opp_data['elo'])
            p2_jump = calculate_level_jump(opp_hist, player_data, name_lookup, pdata['elo'])
            p1_at = calculate_performance_at_level(historical, player_data, name_lookup, opp_data['elo'])
            p2_at = calculate_performance_at_level(opp_hist, player_data, name_lookup, pdata['elo'])
            p1_fve = calculate_form_vs_expected(historical, pdata['elo'], pdata.get(surface_key, 1500), player_data, name_lookup, surface, 3)
            p2_fve = calculate_form_vs_expected(opp_hist, opp_data['elo'], opp_data.get(surface_key, 1500), player_data, name_lookup, surface, 3)
            
            # Full spread features (15 existing)
            features = [
                pdata['elo'] - opp_data['elo'],
                pdata.get(surface_key, 1500) - opp_data.get(surface_key, 1500),
                p1_form['win_pct'] - p2_form['win_pct'],
                p1_form['avg_dr'] - p2_form['avg_dr'],
                (p1_surf['win_pct'] if p1_surf else 0.5) - (p2_surf['win_pct'] if p2_surf else 0.5),
                p1_form['first_won_pct'] - p2_form['first_won_pct'],
                p1_form['rpw_pct'] - p2_form['rpw_pct'],
                p1_h2h['diff'],
                p1_vs_strong['win_pct_vs_strong'] - p2_vs_strong['win_pct_vs_strong'],
                p1_adj['adjusted_win_pct'] - p2_adj['adjusted_win_pct'],
                p1_jump['level_jump'] - p2_jump['level_jump'],
                p1_at['win_pct_at_level'] - p2_at['win_pct_at_level'],
                p1_at['avg_spread_at_level'] - p2_at['avg_spread_at_level'],
                (p1_fve['form_diff'] if p1_fve else 0) - (p2_fve['form_diff'] if p2_fve else 0),
                (p1_fve['spread_form_diff'] if p1_fve else 0) - (p2_fve['spread_form_diff'] if p2_fve else 0),
            ]
            
            # Extreme momentum (2 features)
            if include_extreme:
                p1_ext = calculate_extreme_momentum(historical, player_data, name_lookup, opp_data['elo'], pdata['elo'])
                p2_ext = calculate_extreme_momentum(opp_hist, player_data, name_lookup, pdata['elo'], opp_data['elo'])
                
                features.extend([
                    p1_ext['extreme_signal'] - p2_ext['extreme_signal'],
                    p1_ext['raw_cascade'] - p2_ext['raw_cascade'],
                ])
            
            # Labels
            sets = re.findall(r'(\d+)-(\d+)', match.get('score', ''))
            if not sets:
                continue
            p1_games = sum(int(s[0]) for s in sets)
            p2_games = sum(int(s[1]) for s in sets)
            total = p1_games + p2_games
            spread = p1_games - p2_games
            
            if total < 12 or total > 50:
                continue
            
            X.append(features)
            y_spread.append(spread)
    
    return np.array(X), np.array(y_spread)


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
    print("EXTREME MOMENTUM ON FULL MODEL")
    print("="*70)
    print()
    
    data = load_data()
    print(f"Loaded {data['player_count']} players")
    print()
    
    print("Building BASELINE (full features)...")
    X_base, y_spread = build_full_data(data, include_extreme=False)
    print(f"  {X_base.shape[0]} samples, {X_base.shape[1]} features")
    base_mae = evaluate_spread(X_base, y_spread)
    
    print("\nBuilding WITH EXTREME MOMENTUM...")
    X_ext, y_spread = build_full_data(data, include_extreme=True)
    print(f"  {X_ext.shape[0]} samples, {X_ext.shape[1]} features")
    ext_mae = evaluate_spread(X_ext, y_spread)
    
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()
    print(f"{'Config':<35} {'Features':>10} {'Spread MAE':>12}")
    print("-"*60)
    print(f"{'Full Model (baseline)':<35} {X_base.shape[1]:>10} {base_mae:>12.4f}")
    print(f"{'+ Extreme Momentum':<35} {X_ext.shape[1]:>10} {ext_mae:>12.4f}")
    print(f"{'Change':<35} {'+2':>10} {ext_mae - base_mae:>+12.5f}")
    
    print()
    if ext_mae < base_mae:
        print("=> IMPROVEMENT! Recommend integrating.")
    else:
        print("=> No improvement on full model.")


if __name__ == '__main__':
    main()
