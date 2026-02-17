"""
Experiment v2: Dominance features on TOP of full model
"""

import json
import re
import numpy as np
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

# Import existing feature builders
from elo_level_experiment_v2 import (
    load_data, parse_score, calculate_rolling_stats, calculate_surface_form,
    calculate_blended_stats, calculate_head_to_head,
    calculate_vs_strong_opponents, calculate_elo_adjusted_form, calculate_level_jump
)


def calculate_dominance_vs_strong(matches, player_data, name_lookup, strong_threshold=1900, lookback=20):
    """Calculate dominance metrics vs strong opponents"""
    recent = matches[:lookback]
    
    spreads = []
    serve_pcts = []
    return_pcts = []
    bp_converted = []
    bp_saved = []
    
    for match in recent:
        opp_name = match.get('opponent', '')
        opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
        
        if not opp_key or opp_key not in player_data:
            continue
            
        opp_elo = player_data[opp_key].get('elo_overall', player_data[opp_key].get('elo', 1500))
        
        if opp_elo >= strong_threshold:
            # Parse spread
            sets = re.findall(r'(\d+)-(\d+)', match.get('score', ''))
            if sets:
                p1_games = sum(int(s[0]) for s in sets)
                p2_games = sum(int(s[1]) for s in sets)
                spreads.append(p1_games - p2_games)
            
            # Serve stats
            serve = match.get('serve', {})
            if serve:
                first_won = serve.get('first_won_pct') or 0
                second_won = serve.get('second_won_pct') or 0
                if first_won or second_won:
                    serve_pcts.append(first_won * 0.6 + second_won * 0.4)
                if serve.get('bp_saved_pct'):
                    bp_saved.append(serve['bp_saved_pct'])
            
            # Return stats
            ret = match.get('return', {})
            if ret:
                if ret.get('rpw_pct'):
                    return_pcts.append(ret['rpw_pct'])
                if ret.get('bp_conv_pct'):
                    bp_converted.append(ret['bp_conv_pct'])
    
    return {
        'avg_spread_vs_strong': np.mean(spreads) if spreads else 0,
        'serve_pct_vs_strong': np.mean(serve_pcts) if serve_pcts else 55,
        'return_pct_vs_strong': np.mean(return_pcts) if return_pcts else 35,
        'bp_conv_vs_strong': np.mean(bp_converted) if bp_converted else 40,
        'bp_saved_vs_strong': np.mean(bp_saved) if bp_saved else 60,
        'matches_vs_strong': len(spreads)
    }


def calculate_dominance_vs_similar(matches, player_elo, player_data, name_lookup, elo_range=150, lookback=20):
    """Calculate dominance vs similar-level opponents"""
    recent = matches[:lookback]
    
    spreads = []
    serve_pcts = []
    
    for match in recent:
        opp_name = match.get('opponent', '')
        opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
        
        if not opp_key or opp_key not in player_data:
            continue
            
        opp_elo = player_data[opp_key].get('elo_overall', player_data[opp_key].get('elo', 1500))
        
        if abs(opp_elo - player_elo) <= elo_range:
            sets = re.findall(r'(\d+)-(\d+)', match.get('score', ''))
            if sets:
                p1_games = sum(int(s[0]) for s in sets)
                p2_games = sum(int(s[1]) for s in sets)
                spreads.append(p1_games - p2_games)
            
            serve = match.get('serve', {})
            if serve:
                first_won = serve.get('first_won_pct') or 0
                second_won = serve.get('second_won_pct') or 0
                if first_won or second_won:
                    serve_pcts.append(first_won * 0.6 + second_won * 0.4)
    
    return {
        'avg_spread_vs_similar': np.mean(spreads) if spreads else 0,
        'serve_pct_vs_similar': np.mean(serve_pcts) if serve_pcts else 55,
        'matches_vs_similar': len(spreads)
    }


def calculate_close_match_tendency(matches, lookback=15):
    """Calculate tendency to play close matches"""
    recent = matches[:lookback]
    
    three_setters = 0
    close_matches = 0
    
    for match in recent:
        score = match.get('score', '')
        sets = re.findall(r'(\d+)-(\d+)', score)
        
        if sets:
            p1_games = sum(int(s[0]) for s in sets)
            p2_games = sum(int(s[1]) for s in sets)
            spread = abs(p1_games - p2_games)
            
            if len(sets) == 3:
                three_setters += 1
            if spread <= 3:
                close_matches += 1
    
    n = len(recent) if recent else 1
    return {
        'three_set_rate': three_setters / n,
        'close_match_rate': close_matches / n
    }


def build_full_data(data, include_dominance=False):
    """Build training data with FULL features + optional dominance"""
    
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
            
            # Calculate all stats
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
            
            # ELO-level features (already integrated)
            p1_vs_strong = calculate_vs_strong_opponents(historical, player_data, name_lookup)
            p2_vs_strong = calculate_vs_strong_opponents(opp_hist, player_data, name_lookup)
            p1_adj = calculate_elo_adjusted_form(historical, player_data, name_lookup)
            p2_adj = calculate_elo_adjusted_form(opp_hist, player_data, name_lookup)
            p1_jump = calculate_level_jump(historical, player_data, name_lookup, opp_data['elo_overall'])
            p2_jump = calculate_level_jump(opp_hist, player_data, name_lookup, pdata['elo_overall'])
            
            # FULL base features (matching prediction_model.py with ELO-level features)
            features = [
                # ELO (4)
                pdata['elo_overall'] - opp_data['elo_overall'],
                pdata.get(surface_key, 1500) - opp_data.get(surface_key, 1500),
                pdata['elo_overall'],
                opp_data['elo_overall'],
                
                # Form (4)
                p1_form['win_pct'] - p2_form['win_pct'],
                p1_form['avg_dr'] - p2_form['avg_dr'],
                p1_form['win_pct'],
                p2_form['win_pct'],
                
                # Serve (5)
                p1_form['first_in_pct'] - p2_form['first_in_pct'],
                p1_form['first_won_pct'] - p2_form['first_won_pct'],
                p1_form['ace_pct'] - p2_form['ace_pct'],
                p1_form['df_pct'] - p2_form['df_pct'],
                p1_form['bp_saved_pct'] - p2_form['bp_saved_pct'],
                
                # Return (2)
                p1_form['rpw_pct'] - p2_form['rpw_pct'],
                p1_form['bp_conv_pct'] - p2_form['bp_conv_pct'],
                
                # Surface (1)
                (p1_surf['win_pct'] if p1_surf else 0.5) - (p2_surf['win_pct'] if p2_surf else 0.5),
                
                # Blended (7)
                p1_blend['first_won_pct'] - p2_blend['first_won_pct'],
                p1_blend['ace_pct'] - p2_blend['ace_pct'],
                p1_blend['bp_saved_pct'] - p2_blend['bp_saved_pct'],
                p1_blend['rpw_pct'] - p2_blend['rpw_pct'],
                p1_blend['bp_conv_pct'] - p2_blend['bp_conv_pct'],
                p1_blend['win_pct'] - p2_blend['win_pct'],
                p1_blend['avg_dr'] - p2_blend['avg_dr'],
                
                # Surface exp (3)
                p1_blend.get('surface_match_count', 0) - p2_blend.get('surface_match_count', 0),
                p1_blend.get('surface_match_count', 0),
                p2_blend.get('surface_match_count', 0),
                
                # H2H (3)
                p1_h2h['diff'],
                p1_h2h['win_pct'],
                p1_h2h['total'],
                
                # ELO-level features (6)
                p1_vs_strong['win_pct_vs_strong'] - p2_vs_strong['win_pct_vs_strong'],
                p1_vs_strong['matches_vs_strong'] - p2_vs_strong['matches_vs_strong'],
                p1_adj['adjusted_win_pct'] - p2_adj['adjusted_win_pct'],
                p1_adj['avg_opponent_elo'] - p2_adj['avg_opponent_elo'],
                p1_jump['level_jump'] - p2_jump['level_jump'],
                p1_jump['level_jump_pct'] - p2_jump['level_jump_pct'],
            ]
            
            # NEW: Dominance features
            if include_dominance:
                p1_dom_strong = calculate_dominance_vs_strong(historical, player_data, name_lookup)
                p2_dom_strong = calculate_dominance_vs_strong(opp_hist, player_data, name_lookup)
                p1_dom_similar = calculate_dominance_vs_similar(historical, pdata['elo_overall'], player_data, name_lookup)
                p2_dom_similar = calculate_dominance_vs_similar(opp_hist, opp_data['elo_overall'], player_data, name_lookup)
                p1_close = calculate_close_match_tendency(historical)
                p2_close = calculate_close_match_tendency(opp_hist)
                
                features.extend([
                    # Dominance vs strong (5)
                    p1_dom_strong['avg_spread_vs_strong'] - p2_dom_strong['avg_spread_vs_strong'],
                    p1_dom_strong['serve_pct_vs_strong'] - p2_dom_strong['serve_pct_vs_strong'],
                    p1_dom_strong['return_pct_vs_strong'] - p2_dom_strong['return_pct_vs_strong'],
                    p1_dom_strong['bp_conv_vs_strong'] - p2_dom_strong['bp_conv_vs_strong'],
                    p1_dom_strong['bp_saved_vs_strong'] - p2_dom_strong['bp_saved_vs_strong'],
                    
                    # Dominance vs similar (2)
                    p1_dom_similar['avg_spread_vs_similar'] - p2_dom_similar['avg_spread_vs_similar'],
                    p1_dom_similar['serve_pct_vs_similar'] - p2_dom_similar['serve_pct_vs_similar'],
                    
                    # Close match tendency (2) - reversed so negative = relies on fine margins
                    p2_close['three_set_rate'] - p1_close['three_set_rate'],
                    p2_close['close_match_rate'] - p1_close['close_match_rate'],
                ])
            
            # Parse score for labels
            sets = re.findall(r'(\d+)-(\d+)', match.get('score', ''))
            if not sets:
                continue
            p1_games = sum(int(s[0]) for s in sets)
            p2_games = sum(int(s[1]) for s in sets)
            total = p1_games + p2_games
            
            if total < 12 or total > 50:
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
    print("DOMINANCE FEATURES - FULL MODEL TEST")
    print("="*70)
    print()
    
    data = load_data()
    print(f"Loaded {data['player_count']} players")
    print()
    
    # Current model (full features with ELO-level)
    print("Building CURRENT MODEL (full features + ELO-level)...")
    X_curr, y, y_spread, y_total = build_full_data(data, include_dominance=False)
    print(f"  {X_curr.shape[0]} samples, {X_curr.shape[1]} features")
    
    print("Evaluating current model...")
    curr_acc, curr_auc, curr_spread, curr_total, curr_ou = evaluate(X_curr, y, y_spread, y_total)
    
    # With dominance features
    print()
    print("Building WITH DOMINANCE FEATURES...")
    X_dom, y, y_spread, y_total = build_full_data(data, include_dominance=True)
    print(f"  {X_dom.shape[0]} samples, {X_dom.shape[1]} features")
    
    print("Evaluating with dominance...")
    dom_acc, dom_auc, dom_spread, dom_total, dom_ou = evaluate(X_dom, y, y_spread, y_total)
    
    # Results
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()
    print(f"{'Metric':<20} {'Current':>12} {'+ Dominance':>12} {'Change':>12}")
    print("-"*60)
    print(f"{'Winner Accuracy':<20} {curr_acc:>11.1%} {dom_acc:>11.1%} {(dom_acc-curr_acc)*100:>+11.2f}%")
    print(f"{'Winner AUC':<20} {curr_auc:>11.3f} {dom_auc:>11.3f} {dom_auc-curr_auc:>+11.4f}")
    print(f"{'Spread MAE':<20} {curr_spread:>11.2f} {dom_spread:>11.2f} {dom_spread-curr_spread:>+11.3f}")
    print(f"{'Total MAE':<20} {curr_total:>11.2f} {dom_total:>11.2f} {dom_total-curr_total:>+11.3f}")
    print(f"{'O/U 21.5 Acc':<20} {curr_ou:>11.1%} {dom_ou:>11.1%} {(dom_ou-curr_ou)*100:>+11.2f}%")
    
    print()
    print("="*70)
    print("NEW DOMINANCE FEATURES (9 total)")
    print("="*70)
    print()
    print("Dominance vs Strong Opponents (ELO > 1900):")
    print("  - avg_spread_vs_strong: Win margin vs top players")
    print("  - serve_pct_vs_strong: Serve dominance vs top players")
    print("  - return_pct_vs_strong: Return dominance vs top players")
    print("  - bp_conv_vs_strong: Break points converted vs top players")
    print("  - bp_saved_vs_strong: Break points saved vs top players")
    print()
    print("Dominance vs Similar Opponents (within 150 ELO):")
    print("  - avg_spread_vs_similar: Win margin vs peers")
    print("  - serve_pct_vs_similar: Serve dominance vs peers")
    print()
    print("Close Match Tendency:")
    print("  - three_set_rate_diff: Who plays more 3-setters (fine margins)")
    print("  - close_match_rate_diff: Who has more close matches (spread<=3)")


if __name__ == '__main__':
    main()
