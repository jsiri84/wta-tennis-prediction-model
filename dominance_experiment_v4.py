"""
Experiment v4: Test best new features on FULL model
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


def calculate_performance_at_level(matches, player_data, name_lookup, target_elo, elo_range=100, lookback=25):
    """Performance against opponents near a target ELO level"""
    recent = matches[:lookback]
    
    spreads = []
    wins = 0
    matches_at_level = 0
    serve_pcts = []
    return_pcts = []
    
    for match in recent:
        opp_name = match.get('opponent', '')
        opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
        
        if not opp_key or opp_key not in player_data:
            continue
        
        opp_elo = player_data[opp_key].get('elo_overall', player_data[opp_key].get('elo', 1500))
        
        if abs(opp_elo - target_elo) <= elo_range:
            matches_at_level += 1
            
            if match.get('result') == 'W':
                wins += 1
            
            sets = re.findall(r'(\d+)-(\d+)', match.get('score', ''))
            if sets:
                p1 = sum(int(s[0]) for s in sets)
                p2 = sum(int(s[1]) for s in sets)
                spreads.append(p1 - p2)
            
            serve = match.get('serve', {})
            if serve:
                fw = serve.get('first_won_pct') or 0
                sw = serve.get('second_won_pct') or 0
                if fw or sw:
                    serve_pcts.append(fw * 0.6 + sw * 0.4)
            
            ret = match.get('return', {})
            if ret and ret.get('rpw_pct'):
                return_pcts.append(ret['rpw_pct'])
    
    return {
        'win_pct_at_level': wins / matches_at_level if matches_at_level > 0 else 0.5,
        'avg_spread_at_level': np.mean(spreads) if spreads else 0,
        'serve_pct_at_level': np.mean(serve_pcts) if serve_pcts else 55,
        'return_pct_at_level': np.mean(return_pcts) if return_pcts else 35,
        'matches_at_level': matches_at_level
    }


def calculate_momentum_form(matches, lookback=5):
    """Recent momentum - last 5 matches"""
    recent = matches[:lookback]
    if len(recent) < 3:
        return {'recent_win_pct': 0.5, 'recent_avg_spread': 0}
    
    wins = sum(1 for m in recent if m.get('result') == 'W')
    spreads = []
    for m in recent:
        sets = re.findall(r'(\d+)-(\d+)', m.get('score', ''))
        if sets:
            p1 = sum(int(s[0]) for s in sets)
            p2 = sum(int(s[1]) for s in sets)
            spreads.append(p1 - p2)
    
    return {
        'recent_win_pct': wins / len(recent),
        'recent_avg_spread': np.mean(spreads) if spreads else 0,
    }


def build_full_data(data, include_at_level=False, include_momentum=False):
    """Build full training data with optional new features"""
    
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
            p1_jump = calculate_level_jump(historical, player_data, name_lookup, opp_data['elo_overall'])
            p2_jump = calculate_level_jump(opp_hist, player_data, name_lookup, pdata['elo_overall'])
            
            # FULL features (35 total)
            features = [
                pdata['elo_overall'] - opp_data['elo_overall'],
                pdata.get(surface_key, 1500) - opp_data.get(surface_key, 1500),
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
            ]
            
            # NEW: At opponent's level
            if include_at_level:
                p1_at = calculate_performance_at_level(historical, player_data, name_lookup, opp_data['elo_overall'])
                p2_at = calculate_performance_at_level(opp_hist, player_data, name_lookup, pdata['elo_overall'])
                
                features.extend([
                    p1_at['win_pct_at_level'] - p2_at['win_pct_at_level'],
                    p1_at['avg_spread_at_level'] - p2_at['avg_spread_at_level'],
                    p1_at['serve_pct_at_level'] - p2_at['serve_pct_at_level'],
                    p1_at['return_pct_at_level'] - p2_at['return_pct_at_level'],
                ])
            
            # NEW: Momentum
            if include_momentum:
                p1_mom = calculate_momentum_form(historical, 5)
                p2_mom = calculate_momentum_form(opp_hist, 5)
                
                features.extend([
                    p1_mom['recent_win_pct'] - p2_mom['recent_win_pct'],
                    p1_mom['recent_avg_spread'] - p2_mom['recent_avg_spread'],
                ])
            
            # Labels
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
    
    winner_model = LogisticRegression(max_iter=1000, C=0.1)
    winner_proba = cross_val_predict(winner_model, X_scaled, y, cv=kf, method='predict_proba')[:, 1]
    winner_acc = accuracy_score(y, (winner_proba > 0.5).astype(int))
    winner_auc = roc_auc_score(y, winner_proba)
    
    spread_model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    spread_pred = cross_val_predict(spread_model, X_scaled, y_spread, cv=kf)
    spread_mae = mean_absolute_error(y_spread, spread_pred)
    
    total_model = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    total_pred = cross_val_predict(total_model, X_scaled, y_total, cv=kf)
    total_mae = mean_absolute_error(y_total, total_pred)
    ou_acc = ((y_total > 21.5) == (total_pred > 21.5)).mean()
    
    return winner_acc, winner_auc, spread_mae, total_mae, ou_acc


def main():
    print("="*70)
    print("FULL MODEL + NEW FEATURES TEST")
    print("="*70)
    print()
    
    data = load_data()
    print(f"Loaded {data['player_count']} players")
    print()
    
    configs = [
        ('Current Full Model', False, False),
        ('+ At Opponent Level', True, False),
        ('+ Momentum', False, True),
        ('+ Both New Features', True, True),
    ]
    
    results = []
    
    for name, at_level, momentum in configs:
        print(f"Testing: {name}...")
        X, y, y_spread, y_total = build_full_data(data, include_at_level=at_level, include_momentum=momentum)
        print(f"  {X.shape[0]} samples, {X.shape[1]} features")
        
        acc, auc, spread, total, ou = evaluate(X, y, y_spread, y_total)
        results.append((name, X.shape[1], acc, auc, spread, total, ou))
    
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()
    print(f"{'Config':<25} {'Feat':>5} {'Win%':>7} {'AUC':>7} {'Spread':>7} {'Total':>7} {'O/U':>7}")
    print("-"*70)
    
    baseline = results[0]
    for name, feat, acc, auc, spread, total, ou in results:
        print(f"{name:<25} {feat:>5} {acc:>6.1%} {auc:>7.3f} {spread:>7.2f} {total:>7.2f} {ou:>6.1%}")
    
    print()
    print("IMPROVEMENT FROM BASELINE:")
    print("-"*70)
    for name, feat, acc, auc, spread, total, ou in results[1:]:
        acc_d = (acc - baseline[2]) * 100
        auc_d = auc - baseline[3]
        spread_d = spread - baseline[4]
        total_d = total - baseline[5]
        ou_d = (ou - baseline[6]) * 100
        
        print(f"{name:<25}       {acc_d:>+6.2f}% {auc_d:>+7.4f} {spread_d:>+7.3f} {total_d:>+7.3f} {ou_d:>+6.2f}%")
    
    print()
    print("="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    best_acc = max(results, key=lambda x: x[2])
    best_spread = min(results, key=lambda x: x[4])
    
    print(f"\nBest for Winner Accuracy: {best_acc[0]} ({best_acc[2]:.1%})")
    print(f"Best for Spread MAE: {best_spread[0]} ({best_spread[4]:.2f})")


if __name__ == '__main__':
    main()
