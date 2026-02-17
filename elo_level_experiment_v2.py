"""
Experiment v2: ELO Level-Adjusted Features with FULL feature set

Tests adding the 3 new features on top of all existing features.
"""

import json
import re
import numpy as np
from collections import defaultdict
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error


def load_data():
    with open('all_players_matches.json', 'r') as f:
        return json.load(f)


def parse_score(score_str):
    """Parse score string like '6-4, 7-5' into games"""
    if not score_str:
        return None, None, None
    try:
        sets = re.findall(r'(\d+)-(\d+)', score_str)
        if not sets:
            return None, None, None
        p1_games = sum(int(s[0]) for s in sets)
        p2_games = sum(int(s[1]) for s in sets)
        return p1_games, p2_games, p1_games + p2_games
    except:
        return None, None, None


def calculate_rolling_stats(matches, lookback=15):
    """Calculate rolling statistics from recent matches"""
    recent = matches[:lookback]
    if len(recent) < 3:
        return None
    
    wins = sum(1 for m in recent if m.get('result') == 'W')
    
    first_in = [m.get('first_serve_pct', 0) for m in recent if m.get('first_serve_pct')]
    first_won = [m.get('first_serve_won_pct', 0) for m in recent if m.get('first_serve_won_pct')]
    aces = [m.get('aces', 0) / max(m.get('total_points_served', 100), 1) * 100 for m in recent]
    dfs = [m.get('double_faults', 0) / max(m.get('total_points_served', 100), 1) * 100 for m in recent]
    bp_saved = [m.get('bp_saved_pct', 0) for m in recent if m.get('bp_saved_pct')]
    rpw = [m.get('return_points_won_pct', 0) for m in recent if m.get('return_points_won_pct')]
    bp_conv = [m.get('bp_converted_pct', 0) for m in recent if m.get('bp_converted_pct')]
    
    dr_list = []
    for m in recent:
        p1, p2, _ = parse_score(m.get('score', ''))
        if p1 and p2:
            dr_list.append(p1 / max(p2, 1))
    
    return {
        'win_pct': wins / len(recent) if recent else 0.5,
        'first_in_pct': np.mean(first_in) if first_in else 60,
        'first_won_pct': np.mean(first_won) if first_won else 70,
        'ace_pct': np.mean(aces) if aces else 5,
        'df_pct': np.mean(dfs) if dfs else 3,
        'bp_saved_pct': np.mean(bp_saved) if bp_saved else 60,
        'rpw_pct': np.mean(rpw) if rpw else 35,
        'bp_conv_pct': np.mean(bp_conv) if bp_conv else 40,
        'avg_dr': np.mean(dr_list) if dr_list else 1.0,
    }


def calculate_surface_form(matches, surface, lookback=15):
    """Calculate form on specific surface"""
    surface_matches = [m for m in matches if m.get('surface', '').lower() == surface.lower()][:lookback]
    if len(surface_matches) < 3:
        return None
    return calculate_rolling_stats(surface_matches, lookback)


def calculate_blended_stats(matches, surface, lookback=15, surface_weight=0.6):
    """Calculate surface-weighted blended stats"""
    all_form = calculate_rolling_stats(matches, lookback)
    surface_form = calculate_surface_form(matches, surface, lookback)
    
    if not all_form:
        return None
    
    if not surface_form:
        result = all_form.copy()
        result['surface_match_count'] = 0
        return result
    
    result = {}
    for key in all_form:
        result[key] = surface_weight * surface_form[key] + (1 - surface_weight) * all_form[key]
    
    surface_matches = [m for m in matches[:lookback] if m.get('surface', '').lower() == surface.lower()]
    result['surface_match_count'] = len(surface_matches)
    
    return result


def calculate_head_to_head(matches, opponent_name):
    """Calculate head-to-head record vs specific opponent"""
    h2h_matches = [m for m in matches if opponent_name.lower() in m.get('opponent', '').lower()]
    
    if not h2h_matches:
        return {'diff': 0, 'win_pct': 0.5, 'total': 0}
    
    wins = sum(1 for m in h2h_matches if m.get('result') == 'W')
    losses = len(h2h_matches) - wins
    
    return {
        'diff': wins - losses,
        'win_pct': wins / len(h2h_matches) if h2h_matches else 0.5,
        'total': len(h2h_matches)
    }


# ============================================================
# NEW ELO-LEVEL FEATURES
# ============================================================

def calculate_vs_strong_opponents(matches, player_data, name_lookup, strong_threshold=1900, lookback=20):
    """Win rate against strong opponents (ELO > threshold)"""
    recent = matches[:lookback]
    wins_vs_strong = 0
    matches_vs_strong = 0
    
    for match in recent:
        opp_name = match.get('opponent', '')
        opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
        
        if opp_key and opp_key in player_data:
            opp_elo = player_data[opp_key].get('elo_overall', 1500)
            if opp_elo >= strong_threshold:
                matches_vs_strong += 1
                if match.get('result') == 'W':
                    wins_vs_strong += 1
    
    return {
        'win_pct_vs_strong': wins_vs_strong / matches_vs_strong if matches_vs_strong > 0 else 0.5,
        'matches_vs_strong': matches_vs_strong
    }


def calculate_elo_adjusted_form(matches, player_data, name_lookup, lookback=15):
    """Opponent-adjusted form (weight wins by opponent ELO)"""
    recent = matches[:lookback]
    weighted_wins = 0
    total_weight = 0
    opponent_elos = []
    
    for match in recent:
        opp_name = match.get('opponent', '')
        opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
        
        if opp_key and opp_key in player_data:
            opp_elo = player_data[opp_key].get('elo_overall', 1500)
            opponent_elos.append(opp_elo)
            weight = opp_elo / 1800
            total_weight += weight
            if match.get('result') == 'W':
                weighted_wins += weight
    
    return {
        'adjusted_win_pct': weighted_wins / total_weight if total_weight > 0 else 0.5,
        'avg_opponent_elo': np.mean(opponent_elos) if opponent_elos else 1700
    }


def calculate_level_jump(matches, player_data, name_lookup, current_opponent_elo, lookback=10):
    """Level jump: how much harder is current opponent vs recent opponents"""
    recent = matches[:lookback]
    opponent_elos = []
    
    for match in recent:
        opp_name = match.get('opponent', '')
        opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
        
        if opp_key and opp_key in player_data:
            opp_elo = player_data[opp_key].get('elo_overall', 1500)
            opponent_elos.append(opp_elo)
    
    avg_recent_opp_elo = np.mean(opponent_elos) if opponent_elos else 1700
    level_jump = current_opponent_elo - avg_recent_opp_elo
    
    return {
        'level_jump': level_jump,
        'level_jump_pct': level_jump / avg_recent_opp_elo if avg_recent_opp_elo > 0 else 0,
        'avg_recent_opp_elo': avg_recent_opp_elo
    }


# ============================================================
# BUILD FULL TRAINING DATA
# ============================================================

def build_full_training_data(data, include_new_features=False):
    """Build training data with FULL feature set (matching prediction_model.py)"""
    
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
            
            # Calculate all stats
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
            
            # FULL feature vector (matching prediction_model.py)
            features = [
                # ELO (4)
                pdata['elo_overall'] - opp_data['elo_overall'],
                pdata.get(surface_elo_key, 1500) - opp_data.get(surface_elo_key, 1500),
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
                
                # Surface form (1)
                (p1_surf_form['win_pct'] if p1_surf_form else 0.5) - (p2_surf_form['win_pct'] if p2_surf_form else 0.5),
                
                # Blended serve (3)
                p1_blended['first_won_pct'] - p2_blended['first_won_pct'],
                p1_blended['ace_pct'] - p2_blended['ace_pct'],
                p1_blended['bp_saved_pct'] - p2_blended['bp_saved_pct'],
                
                # Blended return (2)
                p1_blended['rpw_pct'] - p2_blended['rpw_pct'],
                p1_blended['bp_conv_pct'] - p2_blended['bp_conv_pct'],
                
                # Blended form (2)
                p1_blended['win_pct'] - p2_blended['win_pct'],
                p1_blended['avg_dr'] - p2_blended['avg_dr'],
                
                # Surface experience (3)
                p1_blended.get('surface_match_count', 0) - p2_blended.get('surface_match_count', 0),
                p1_blended.get('surface_match_count', 0),
                p2_blended.get('surface_match_count', 0),
                
                # H2H (3)
                p1_h2h['diff'],
                p1_h2h['win_pct'],
                p1_h2h['total'],
            ]
            
            # NEW ELO-LEVEL FEATURES
            if include_new_features:
                # Option 1: Vs strong opponents
                p1_vs_strong = calculate_vs_strong_opponents(historical_matches, player_data, name_lookup)
                p2_vs_strong = calculate_vs_strong_opponents(opp_historical, player_data, name_lookup)
                
                features.extend([
                    p1_vs_strong['win_pct_vs_strong'] - p2_vs_strong['win_pct_vs_strong'],
                    p1_vs_strong['matches_vs_strong'] - p2_vs_strong['matches_vs_strong'],
                ])
                
                # Option 2: ELO-adjusted form
                p1_adj = calculate_elo_adjusted_form(historical_matches, player_data, name_lookup)
                p2_adj = calculate_elo_adjusted_form(opp_historical, player_data, name_lookup)
                
                features.extend([
                    p1_adj['adjusted_win_pct'] - p2_adj['adjusted_win_pct'],
                    p1_adj['avg_opponent_elo'] - p2_adj['avg_opponent_elo'],
                ])
                
                # Option 3: Level jump
                p1_jump = calculate_level_jump(historical_matches, player_data, name_lookup, opp_data['elo_overall'])
                p2_jump = calculate_level_jump(opp_historical, player_data, name_lookup, pdata['elo_overall'])
                
                features.extend([
                    p1_jump['level_jump'] - p2_jump['level_jump'],
                    p1_jump['level_jump_pct'] - p2_jump['level_jump_pct'],
                ])
            
            # Parse score for labels
            p1_games, p2_games, total = parse_score(match.get('score', ''))
            if p1_games is None or total < 12 or total > 50:
                continue
            
            won = match.get('result') == 'W'
            spread = p1_games - p2_games
            
            X.append(features)
            y.append(1 if won else 0)
            y_spread.append(spread)
            y_total.append(total)
    
    return np.array(X), np.array(y), np.array(y_spread), np.array(y_total)


def evaluate(X, y, y_spread, y_total, name):
    """Evaluate configuration"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Winner
    winner_model = LogisticRegression(max_iter=1000, C=0.1)
    winner_proba = cross_val_predict(winner_model, X_scaled, y, cv=kf, method='predict_proba')[:, 1]
    winner_pred = (winner_proba > 0.5).astype(int)
    winner_acc = accuracy_score(y, winner_pred)
    winner_auc = roc_auc_score(y, winner_proba)
    
    # Spread
    spread_model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    spread_pred = cross_val_predict(spread_model, X_scaled, y_spread, cv=kf)
    spread_mae = mean_absolute_error(y_spread, spread_pred)
    
    # Total
    total_model = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    total_pred = cross_val_predict(total_model, X_scaled, y_total, cv=kf)
    total_mae = mean_absolute_error(y_total, total_pred)
    
    # O/U
    ou_acc = ((y_total > 21.5) == (total_pred > 21.5)).mean()
    
    return {
        'name': name,
        'features': X.shape[1],
        'samples': X.shape[0],
        'winner_acc': winner_acc,
        'winner_auc': winner_auc,
        'spread_mae': spread_mae,
        'total_mae': total_mae,
        'ou_acc': ou_acc
    }


def run_experiment():
    print("="*70)
    print("ELO LEVEL FEATURES - FULL MODEL TEST")
    print("="*70)
    print()
    
    data = load_data()
    print(f"Loaded {data['player_count']} players")
    print()
    
    # Baseline (full features, no new ELO-level features)
    print("Building BASELINE (full features, no ELO-level)...")
    X_base, y, y_spread, y_total = build_full_training_data(data, include_new_features=False)
    print(f"  {X_base.shape[0]} samples, {X_base.shape[1]} features")
    
    print("Evaluating baseline...")
    baseline = evaluate(X_base, y, y_spread, y_total, "Baseline")
    
    # With new features
    print()
    print("Building WITH NEW FEATURES (+ 6 ELO-level features)...")
    X_new, y, y_spread, y_total = build_full_training_data(data, include_new_features=True)
    print(f"  {X_new.shape[0]} samples, {X_new.shape[1]} features")
    
    print("Evaluating with new features...")
    with_new = evaluate(X_new, y, y_spread, y_total, "+ ELO Level Features")
    
    # Print results
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()
    print(f"{'Config':<25} {'Features':>8} {'Winner':>10} {'AUC':>8} {'Spread':>8} {'Total':>8} {'O/U':>8}")
    print(f"{'':25} {'':>8} {'Acc':>10} {'':>8} {'MAE':>8} {'MAE':>8} {'21.5':>8}")
    print("-"*70)
    
    for r in [baseline, with_new]:
        print(f"{r['name']:<25} {r['features']:>8} {r['winner_acc']:>9.1%} {r['winner_auc']:>8.3f} {r['spread_mae']:>8.2f} {r['total_mae']:>8.2f} {r['ou_acc']:>7.1%}")
    
    # Deltas
    print()
    print("IMPROVEMENT FROM NEW FEATURES:")
    print("-"*70)
    
    acc_delta = with_new['winner_acc'] - baseline['winner_acc']
    auc_delta = with_new['winner_auc'] - baseline['winner_auc']
    spread_delta = with_new['spread_mae'] - baseline['spread_mae']
    total_delta = with_new['total_mae'] - baseline['total_mae']
    ou_delta = with_new['ou_acc'] - baseline['ou_acc']
    
    print(f"Winner Accuracy:  {acc_delta:+.2%}  {'(BETTER)' if acc_delta > 0.001 else '(worse)' if acc_delta < -0.001 else '(same)'}")
    print(f"Winner AUC:       {auc_delta:+.4f}  {'(BETTER)' if auc_delta > 0.001 else '(worse)' if auc_delta < -0.001 else '(same)'}")
    print(f"Spread MAE:       {spread_delta:+.3f}  {'(BETTER)' if spread_delta < -0.01 else '(worse)' if spread_delta > 0.01 else '(same)'}")
    print(f"Total MAE:        {total_delta:+.3f}  {'(BETTER)' if total_delta < -0.01 else '(worse)' if total_delta > 0.01 else '(same)'}")
    print(f"O/U 21.5:         {ou_delta:+.2%}  {'(BETTER)' if ou_delta > 0.001 else '(worse)' if ou_delta < -0.001 else '(same)'}")
    
    return baseline, with_new


if __name__ == '__main__':
    run_experiment()
