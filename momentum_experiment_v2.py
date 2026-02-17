"""
Test momentum features on FULL model
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
    calculate_blended_stats, calculate_head_to_head,
    calculate_vs_strong_opponents, calculate_elo_adjusted_form, calculate_level_jump
)
from dominance_experiment_v4 import calculate_performance_at_level
from form_experiment_v3 import calculate_form_vs_expected


def calculate_momentum_boost(matches, player_data, name_lookup, current_opponent_elo):
    """Calculate momentum from beating tougher opponents"""
    if not matches:
        return None
    
    last_match = matches[0]
    opp_name = last_match.get('opponent', '')
    opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
    
    if not opp_key or opp_key not in player_data:
        return None
    
    last_opp_elo = player_data[opp_key].get('elo', 1500)
    won_last = last_match.get('result') == 'W'
    
    score = last_match.get('score', '')
    sets = re.findall(r'(\d+)-(\d+)', score)
    last_spread = 0
    if sets:
        p1 = sum(int(s[0]) for s in sets)
        p2 = sum(int(s[1]) for s in sets)
        last_spread = p1 - p2
    
    # Momentum = how much tougher was last opp vs current?
    momentum = last_opp_elo - current_opponent_elo
    
    return {
        'momentum_boost': momentum if won_last else -momentum,
        'last_spread': last_spread if won_last else -last_spread,
        'facing_easier_after_win': 1 if (won_last and momentum > 100) else 0,
        'dominant_scalp': 1 if (won_last and last_spread >= 5 and momentum > 0) else 0,
    }


def build_full_data(data, include_momentum=False):
    """Build full training data with optional momentum"""
    
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
            
            # Full spread features
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
            
            # Momentum features
            if include_momentum:
                p1_mom = calculate_momentum_boost(historical, player_data, name_lookup, opp_data['elo'])
                p2_mom = calculate_momentum_boost(opp_hist, player_data, name_lookup, pdata['elo'])
                
                if p1_mom and p2_mom:
                    features.extend([
                        p1_mom['momentum_boost'] - p2_mom['momentum_boost'],
                        p1_mom['last_spread'] - p2_mom['last_spread'],
                        p1_mom['facing_easier_after_win'] - p2_mom['facing_easier_after_win'],
                        p1_mom['dominant_scalp'] - p2_mom['dominant_scalp'],
                    ])
                else:
                    features.extend([0, 0, 0, 0])
            
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
    print("MOMENTUM FEATURES ON FULL MODEL")
    print("="*70)
    print()
    
    data = load_data()
    print(f"Loaded {data['player_count']} players")
    print()
    
    print("Building BASELINE (full features)...")
    X_base, y_spread = build_full_data(data, include_momentum=False)
    print(f"  {X_base.shape[0]} samples, {X_base.shape[1]} features")
    base_mae = evaluate_spread(X_base, y_spread)
    
    print("\nBuilding WITH MOMENTUM...")
    X_mom, y_spread = build_full_data(data, include_momentum=True)
    print(f"  {X_mom.shape[0]} samples, {X_mom.shape[1]} features")
    mom_mae = evaluate_spread(X_mom, y_spread)
    
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()
    print(f"{'Config':<30} {'Features':>10} {'Spread MAE':>12}")
    print("-"*55)
    print(f"{'Full Model (baseline)':<30} {X_base.shape[1]:>10} {base_mae:>12.3f}")
    print(f"{'+ Momentum Features':<30} {X_mom.shape[1]:>10} {mom_mae:>12.3f}")
    print(f"{'Change':<30} {'+4':>10} {mom_mae - base_mae:>+12.4f}")
    
    print()
    print("NEW FEATURES:")
    print("  1. momentum_boost: Last opp ELO - current opp ELO (if won)")
    print("  2. last_spread: How dominant was the last win?")
    print("  3. facing_easier_after_win: Beat tough, now facing easier")
    print("  4. dominant_scalp: Beat tough opponent by 5+ games")


if __name__ == '__main__':
    main()
