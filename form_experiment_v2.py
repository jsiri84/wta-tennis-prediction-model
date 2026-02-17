"""
Experiment v2: Form vs Expected on FULL model
"""

import json
import re
import numpy as np
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

from elo_level_experiment_v2 import (
    load_data, parse_score, calculate_rolling_stats, calculate_surface_form,
    calculate_blended_stats, calculate_head_to_head,
    calculate_vs_strong_opponents, calculate_elo_adjusted_form, calculate_level_jump
)

from dominance_experiment_v4 import calculate_performance_at_level


def elo_expected_score(player_elo, opponent_elo):
    """Calculate expected score (0-1) based on ELO difference"""
    return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))


def calculate_form_vs_expected(matches, player_elo, player_surface_elo, player_data, name_lookup, surface, window=3):
    """
    Calculate how player is performing vs their expected level.
    """
    recent = matches[:window]
    if len(recent) < 2:
        return None
    
    actual_wins = 0
    expected_wins = 0
    surface_actual = 0
    surface_expected = 0
    surface_matches = 0
    actual_spreads = []
    expected_spreads = []
    
    for match in recent:
        opp_name = match.get('opponent', '')
        opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
        
        if not opp_key or opp_key not in player_data:
            continue
        
        opp_elo = player_data[opp_key].get('elo', 1500)
        match_surface = match.get('surface', '').lower()
        
        exp_win = elo_expected_score(player_elo, opp_elo)
        expected_wins += exp_win
        
        won = match.get('result') == 'W'
        if won:
            actual_wins += 1
        
        if match_surface == surface.lower():
            surface_matches += 1
            opp_surface_elo = player_data[opp_key].get(f'elo_{surface.lower()}', opp_elo)
            surface_exp = elo_expected_score(player_surface_elo, opp_surface_elo)
            surface_expected += surface_exp
            if won:
                surface_actual += 1
        
        score = match.get('score', '')
        sets = re.findall(r'(\d+)-(\d+)', score)
        if sets:
            p1_games = sum(int(s[0]) for s in sets)
            p2_games = sum(int(s[1]) for s in sets)
            actual_spreads.append(p1_games - p2_games)
            elo_diff = player_elo - opp_elo
            expected_spreads.append(elo_diff / 50)
    
    n = len(recent)
    
    # Streak
    streak = 0
    for m in recent:
        if m.get('result') == 'W':
            if streak >= 0:
                streak += 1
            else:
                break
        else:
            if streak <= 0:
                streak -= 1
            else:
                break
    
    form_diff = actual_wins - expected_wins
    form_pct_diff = (actual_wins / n) - (expected_wins / n) if n > 0 else 0
    
    if surface_matches >= 2:
        surface_form_diff = surface_actual - surface_expected
    else:
        surface_form_diff = form_diff * 0.5
    
    if actual_spreads and expected_spreads:
        spread_form_diff = np.mean(actual_spreads) - np.mean(expected_spreads)
    else:
        spread_form_diff = 0
    
    return {
        'form_diff': form_diff,
        'form_pct_diff': form_pct_diff,
        'surface_form_diff': surface_form_diff,
        'spread_form_diff': spread_form_diff,
        'streak': streak,
    }


def build_full_data(data, include_form=False, window=3):
    """Build full training data with optional form features"""
    
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
    X_total = []
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
            
            p1_form = calculate_rolling_stats(historical, 15)
            p2_form = calculate_rolling_stats(opp_hist, 15)
            
            if not p1_form or not p2_form:
                continue
            
            p1_surf = calculate_surface_form(historical, surface, 15)
            p2_surf = calculate_surface_form(opp_hist, surface, 15)
            p1_blend = calculate_blended_stats(historical, surface, 15, 0.6)
            p2_blend = calculate_blended_stats(opp_hist, surface, 15, 0.6)
            
            if not p1_blend or not p2_blend:
                continue
            
            p1_h2h = calculate_head_to_head(historical, opp_name)
            
            # ELO-level features
            p1_vs_strong = calculate_vs_strong_opponents(historical, player_data, name_lookup)
            p2_vs_strong = calculate_vs_strong_opponents(opp_hist, player_data, name_lookup)
            p1_adj = calculate_elo_adjusted_form(historical, player_data, name_lookup)
            p2_adj = calculate_elo_adjusted_form(opp_hist, player_data, name_lookup)
            p1_jump = calculate_level_jump(historical, player_data, name_lookup, opp_data['elo'])
            p2_jump = calculate_level_jump(opp_hist, player_data, name_lookup, pdata['elo'])
            
            # At opponent's level
            p1_at = calculate_performance_at_level(historical, player_data, name_lookup, opp_data['elo'])
            p2_at = calculate_performance_at_level(opp_hist, player_data, name_lookup, pdata['elo'])
            
            # SPREAD features (35 + form)
            spread_features = [
                pdata['elo'] - opp_data['elo'],
                pdata.get(surface_key, 1500) - opp_data.get(surface_key, 1500),
                pdata['elo'],
                opp_data['elo'],
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
                (p1_surf['win_pct'] if p1_surf else 0.5) - (p2_surf['win_pct'] if p2_surf else 0.5),
                p1_blend['first_won_pct'] - p2_blend['first_won_pct'],
                p1_blend['ace_pct'] - p2_blend['ace_pct'],
                p1_blend['bp_saved_pct'] - p2_blend['bp_saved_pct'],
                p1_blend['rpw_pct'] - p2_blend['rpw_pct'],
                p1_blend['bp_conv_pct'] - p2_blend['bp_conv_pct'],
                p1_blend['win_pct'] - p2_blend['win_pct'],
                p1_blend['avg_dr'] - p2_blend['avg_dr'],
                p1_blend.get('surface_match_count', 0) - p2_blend.get('surface_match_count', 0),
                p1_blend.get('surface_match_count', 0),
                p2_blend.get('surface_match_count', 0),
                p1_h2h['diff'],
                p1_h2h['win_pct'],
                p1_h2h['total'],
                p1_vs_strong['win_pct_vs_strong'] - p2_vs_strong['win_pct_vs_strong'],
                p1_vs_strong['matches_vs_strong'] - p2_vs_strong['matches_vs_strong'],
                p1_adj['adjusted_win_pct'] - p2_adj['adjusted_win_pct'],
                p1_adj['avg_opponent_elo'] - p2_adj['avg_opponent_elo'],
                p1_jump['level_jump'] - p2_jump['level_jump'],
                p1_jump['level_jump_pct'] - p2_jump['level_jump_pct'],
                p1_at['win_pct_at_level'] - p2_at['win_pct_at_level'],
                p1_at['avg_spread_at_level'] - p2_at['avg_spread_at_level'],
                p1_at['serve_pct_at_level'] - p2_at['serve_pct_at_level'],
                p1_at['return_pct_at_level'] - p2_at['return_pct_at_level'],
            ]
            
            # TOTAL features (25 + form)
            total_features = [
                abs(pdata['elo'] - opp_data['elo']),
                abs(pdata.get(surface_key, 1500) - opp_data.get(surface_key, 1500)),
                (pdata['elo'] + opp_data['elo']) / 2,
                p1_form['win_pct'] + p2_form['win_pct'],
                abs(p1_form['win_pct'] - p2_form['win_pct']),
                p1_form['first_won_pct'] + p2_form['first_won_pct'],
                p1_form['bp_saved_pct'] + p2_form['bp_saved_pct'],
                p1_form['rpw_pct'] + p2_form['rpw_pct'],
                p1_form['bp_conv_pct'] + p2_form['bp_conv_pct'],
                p1_form['ace_pct'] + p2_form['ace_pct'],
                1 if surface == 'hard' else 0,
                1 if surface == 'clay' else 0,
                1 if surface == 'grass' else 0,
            ]
            
            # Form vs Expected features
            if include_form:
                p1_fve = calculate_form_vs_expected(
                    historical, pdata['elo'], pdata.get(surface_key, 1500),
                    player_data, name_lookup, surface, window
                )
                p2_fve = calculate_form_vs_expected(
                    opp_hist, opp_data['elo'], opp_data.get(surface_key, 1500),
                    player_data, name_lookup, surface, window
                )
                
                if p1_fve and p2_fve:
                    # For spread: differences
                    spread_features.extend([
                        p1_fve['form_diff'] - p2_fve['form_diff'],
                        p1_fve['surface_form_diff'] - p2_fve['surface_form_diff'],
                        p1_fve['spread_form_diff'] - p2_fve['spread_form_diff'],
                        p1_fve['streak'] - p2_fve['streak'],
                    ])
                    
                    # For total: combined (hot players may have shorter matches, cold players longer)
                    total_features.extend([
                        p1_fve['form_diff'] + p2_fve['form_diff'],  # Both hot = more decisive?
                        abs(p1_fve['form_diff'] - p2_fve['form_diff']),  # Form gap
                        p1_fve['streak'] + p2_fve['streak'],  # Streak momentum
                    ])
                else:
                    spread_features.extend([0, 0, 0, 0])
                    total_features.extend([0, 0, 0])
            
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
            
            X_spread.append(spread_features)
            X_total.append(total_features)
            y.append(1 if match.get('result') == 'W' else 0)
            y_spread.append(spread)
            y_total.append(total)
    
    return np.array(X_spread), np.array(X_total), np.array(y), np.array(y_spread), np.array(y_total)


def evaluate(X_spread, X_total, y, y_spread, y_total):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Winner (from spread features)
    scaler_s = StandardScaler()
    X_s = scaler_s.fit_transform(X_spread)
    winner_model = LogisticRegression(max_iter=1000, C=0.1)
    proba = cross_val_predict(winner_model, X_s, y, cv=kf, method='predict_proba')[:, 1]
    winner_acc = accuracy_score(y, (proba > 0.5).astype(int))
    winner_auc = roc_auc_score(y, proba)
    
    # Spread
    spread_model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    spread_pred = cross_val_predict(spread_model, X_s, y_spread, cv=kf)
    spread_mae = mean_absolute_error(y_spread, spread_pred)
    
    # Total
    scaler_t = StandardScaler()
    X_t = scaler_t.fit_transform(X_total)
    total_model = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    total_pred = cross_val_predict(total_model, X_t, y_total, cv=kf)
    total_mae = mean_absolute_error(y_total, total_pred)
    ou_acc = ((y_total > 21.5) == (total_pred > 21.5)).mean()
    
    return winner_acc, winner_auc, spread_mae, total_mae, ou_acc


def main():
    print("="*70)
    print("FORM VS EXPECTED - FULL MODEL TEST")
    print("="*70)
    print()
    
    data = load_data()
    print(f"Loaded {data['player_count']} players")
    print()
    
    configs = [
        ('Current Full Model', False, 3),
        ('+ Form (3-match window)', True, 3),
        ('+ Form (5-match window)', True, 5),
    ]
    
    results = []
    
    for name, include_form, window in configs:
        print(f"Testing: {name}...")
        X_s, X_t, y, y_spread, y_total = build_full_data(data, include_form, window)
        print(f"  Spread features: {X_s.shape[1]}, Total features: {X_t.shape[1]}")
        
        acc, auc, spread, total, ou = evaluate(X_s, X_t, y, y_spread, y_total)
        results.append((name, X_s.shape[1], X_t.shape[1], acc, auc, spread, total, ou))
    
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()
    print(f"{'Config':<25} {'SpFt':>5} {'TtFt':>5} {'Win%':>7} {'AUC':>7} {'Spread':>7} {'Total':>7} {'O/U':>7}")
    print("-"*80)
    
    baseline = results[0]
    for name, sp_feat, tt_feat, acc, auc, spread, total, ou in results:
        print(f"{name:<25} {sp_feat:>5} {tt_feat:>5} {acc:>6.1%} {auc:>7.3f} {spread:>7.2f} {total:>7.2f} {ou:>6.1%}")
    
    print()
    print("IMPROVEMENT FROM BASELINE:")
    print("-"*80)
    for name, sp_feat, tt_feat, acc, auc, spread, total, ou in results[1:]:
        acc_d = (acc - baseline[3]) * 100
        auc_d = auc - baseline[4]
        spread_d = spread - baseline[5]
        total_d = total - baseline[6]
        ou_d = (ou - baseline[7]) * 100
        
        print(f"{name:<25}             {acc_d:>+6.2f}% {auc_d:>+7.4f} {spread_d:>+7.3f} {total_d:>+7.3f} {ou_d:>+6.2f}%")


if __name__ == '__main__':
    main()
