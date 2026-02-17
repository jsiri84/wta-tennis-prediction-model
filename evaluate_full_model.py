"""
Full Model Evaluation with Cross-Validation
"""

import json
import re
import numpy as np
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

# Import all the feature functions from unified_model
from unified_model import (
    load_data, calculate_player_stats, calculate_surface_stats, calculate_h2h,
    get_tournament_level, get_round_level,
    calculate_vs_strong_opponents, calculate_elo_adjusted_form, calculate_level_jump,
    calculate_performance_at_level, calculate_form_vs_expected,
    calculate_extreme_momentum, calculate_tournament_performance,
    parse_score
)


def build_full_data(data):
    """Build FULL training data with ALL features"""
    
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
    y_spread = []
    y_total = []
    y_winner = []
    
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
            
            tourn_level = get_tournament_level(match.get('tournament', ''))
            round_level = get_round_level(match.get('round', 'R32'))
            
            # All features
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
            p1_extreme = calculate_extreme_momentum(hist, player_data, name_lookup, opp_data['elo'], pdata['elo'])
            p2_extreme = calculate_extreme_momentum(opp_hist, player_data, name_lookup, pdata['elo'], opp_data['elo'])
            p1_tourn = calculate_tournament_performance(hist, player_data, name_lookup, 4)
            p2_tourn = calculate_tournament_performance(opp_hist, player_data, name_lookup, 4)
            
            # SPREAD FEATURES
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
                # NEW: Extreme momentum
                p1_extreme['extreme_signal'] - p2_extreme['extreme_signal'],
                p1_extreme['raw_cascade'] - p2_extreme['raw_cascade'],
                # NEW: Tournament performance
                (p1_tourn['avg_opp_elo_beaten'] if p1_tourn else 0) - (p2_tourn['avg_opp_elo_beaten'] if p2_tourn else 0),
                (p1_tourn['max_opp_elo_beaten'] if p1_tourn else 0) - (p2_tourn['max_opp_elo_beaten'] if p2_tourn else 0),
                (p1_tourn['avg_spread_in_tourn'] if p1_tourn else 0) - (p2_tourn['avg_spread_in_tourn'] if p2_tourn else 0),
                (p1_tourn['game_dominance'] if p1_tourn else 1) - (p2_tourn['game_dominance'] if p2_tourn else 1),
                (p1_tourn['clean_sets'] if p1_tourn else 0) - (p2_tourn['clean_sets'] if p2_tourn else 0),
                (p1_tourn['serve_pct_in_tourn'] if p1_tourn else 55) - (p2_tourn['serve_pct_in_tourn'] if p2_tourn else 55),
                (p1_tourn['return_pct_in_tourn'] if p1_tourn else 35) - (p2_tourn['return_pct_in_tourn'] if p2_tourn else 35),
                (p1_tourn['bp_conv_in_tourn'] if p1_tourn else 40) - (p2_tourn['bp_conv_in_tourn'] if p2_tourn else 40),
            ]
            
            # TOTAL FEATURES
            total_features = [
                abs(pdata['elo'] - opp_data['elo']),
                abs(pdata.get(surface_key, 1500) - opp_data.get(surface_key, 1500)),
                (pdata['elo'] + opp_data['elo']) / 2,
                (p1['avg_total'] + p2['avg_total']) / 2,
                p1['avg_total'],
                p2['avg_total'],
                p1['total_std'] + p2['total_std'],
                (p1['max_total'] + p2['max_total']) / 2,
                p1['three_set_rate'] + p2['three_set_rate'],
                p1['straight_set_rate'] + p2['straight_set_rate'],
                p1['tiebreak_rate'] + p2['tiebreak_rate'],
                abs(p1['win_pct'] - p2['win_pct']),
                (p1['hold_rate'] + p2['hold_rate']) / 2,
                (p1['first_won_pct'] + p2['first_won_pct']) / 2,
                (p1['ace_pct'] + p2['ace_pct']) / 2,
                (p1['bp_saved_pct'] + p2['bp_saved_pct']) / 2,
                (p1['break_rate'] + p2['break_rate']) / 2,
                (p1['rpw_pct'] + p2['rpw_pct']) / 2,
                (p1['bp_conv_pct'] + p2['bp_conv_pct']) / 2,
                tourn_level,
                round_level,
                tourn_level * round_level,
                1 if surface == 'hard' else 0,
                1 if surface == 'clay' else 0,
                1 if surface == 'grass' else 0,
            ]
            
            X_spread.append(spread_features)
            X_total.append(total_features)
            y_spread.append(spread)
            y_total.append(total)
            y_winner.append(1 if match.get('result') == 'W' else 0)
    
    return np.array(X_spread), np.array(X_total), np.array(y_spread), np.array(y_total), np.array(y_winner)


def main():
    print("="*70)
    print("FULL MODEL CROSS-VALIDATION EVALUATION")
    print("="*70)
    print()
    
    data = load_data()
    print(f"Loaded {data['player_count']} players")
    print()
    
    print("Building full feature set...")
    X_spread, X_total, y_spread, y_total, y_winner = build_full_data(data)
    print(f"  {len(X_spread)} samples")
    print(f"  {X_spread.shape[1]} spread features")
    print(f"  {X_total.shape[1]} total features")
    print()
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Scale features
    spread_scaler = StandardScaler()
    total_scaler = StandardScaler()
    X_spread_scaled = spread_scaler.fit_transform(X_spread)
    X_total_scaled = total_scaler.fit_transform(X_total)
    
    # Winner model (using spread features)
    print("Evaluating WINNER model...")
    winner_model = LogisticRegression(max_iter=1000, C=0.1)
    proba = cross_val_predict(winner_model, X_spread_scaled, y_winner, cv=kf, method='predict_proba')[:, 1]
    winner_acc = accuracy_score(y_winner, (proba > 0.5).astype(int))
    winner_auc = roc_auc_score(y_winner, proba)
    
    # Spread model
    print("Evaluating SPREAD model...")
    spread_model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    spread_pred = cross_val_predict(spread_model, X_spread_scaled, y_spread, cv=kf)
    spread_mae = mean_absolute_error(y_spread, spread_pred)
    
    # Total model
    print("Evaluating TOTAL model...")
    total_model = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    total_pred = cross_val_predict(total_model, X_total_scaled, y_total, cv=kf)
    total_mae = mean_absolute_error(y_total, total_pred)
    
    # O/U accuracy at various lines
    ou_205 = np.mean(((total_pred > 20.5) == (y_total > 20.5)))
    ou_215 = np.mean(((total_pred > 21.5) == (y_total > 21.5)))
    ou_225 = np.mean(((total_pred > 22.5) == (y_total > 22.5)))
    
    print()
    print("="*70)
    print("RESULTS (5-fold Cross-Validation)")
    print("="*70)
    print()
    print("WINNER PREDICTION:")
    print(f"  Accuracy:  {winner_acc:.1%}")
    print(f"  AUC-ROC:   {winner_auc:.3f}")
    print()
    print("SPREAD PREDICTION:")
    print(f"  MAE:       {spread_mae:.2f} games")
    print()
    print("TOTAL GAMES PREDICTION:")
    print(f"  MAE:       {total_mae:.2f} games")
    print()
    print("OVER/UNDER ACCURACY:")
    print(f"  O/U 20.5:  {ou_205:.1%}")
    print(f"  O/U 21.5:  {ou_215:.1%}")
    print(f"  O/U 22.5:  {ou_225:.1%}")


if __name__ == '__main__':
    main()
