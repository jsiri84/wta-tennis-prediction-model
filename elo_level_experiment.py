"""
Experiment: ELO Level-Adjusted Features

Tests 3 approaches to account for opponent strength in recent matches:
1. Win rate vs strong opponents (ELO > threshold)
2. Opponent-adjusted form (weight wins by opponent ELO)
3. Level jump feature (avg recent opponent ELO vs current opponent)
"""

import json
import re
import numpy as np
from collections import defaultdict
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error


# Load data
def load_data():
    with open('all_players_matches.json', 'r') as f:
        return json.load(f)


def parse_score(score_str):
    """Parse score string like '6-4, 7-5' into games won/lost"""
    if not score_str:
        return None, None, None
    
    try:
        sets = re.findall(r'(\d+)-(\d+)', score_str)
        if not sets:
            return None, None, None
        
        p1_games = sum(int(s[0]) for s in sets)
        p2_games = sum(int(s[1]) for s in sets)
        total = p1_games + p2_games
        spread = p1_games - p2_games
        
        return p1_games, p2_games, total
    except:
        return None, None, None


def calculate_rolling_stats(matches, lookback=15):
    """Calculate rolling statistics from recent matches"""
    recent = matches[:lookback]
    if len(recent) < 3:
        return None
    
    wins = sum(1 for m in recent if m.get('result') == 'W')
    
    # Serve stats
    first_in = [m.get('first_serve_pct', 0) for m in recent if m.get('first_serve_pct')]
    first_won = [m.get('first_serve_won_pct', 0) for m in recent if m.get('first_serve_won_pct')]
    aces = [m.get('aces', 0) / max(m.get('total_points_served', 100), 1) * 100 for m in recent]
    dfs = [m.get('double_faults', 0) / max(m.get('total_points_served', 100), 1) * 100 for m in recent]
    bp_saved = [m.get('bp_saved_pct', 0) for m in recent if m.get('bp_saved_pct')]
    
    # Return stats
    rpw = [m.get('return_points_won_pct', 0) for m in recent if m.get('return_points_won_pct')]
    bp_conv = [m.get('bp_converted_pct', 0) for m in recent if m.get('bp_converted_pct')]
    
    # Dominance ratio
    dr_list = []
    for m in recent:
        if m.get('games_won') and m.get('games_lost'):
            dr_list.append(m['games_won'] / max(m['games_lost'], 1))
    
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
        'matches': recent
    }


# ============================================================
# NEW FEATURE FUNCTIONS
# ============================================================

def calculate_vs_strong_opponents(matches, player_data, name_lookup, strong_threshold=1900, lookback=20):
    """
    Option 1: Win rate against strong opponents (ELO > threshold)
    
    Returns:
        - win_pct_vs_strong: Win percentage against strong opponents
        - matches_vs_strong: Number of matches against strong opponents
    """
    recent = matches[:lookback]
    
    wins_vs_strong = 0
    matches_vs_strong = 0
    
    for match in recent:
        opp_name = match.get('opponent', '')
        
        # Find opponent ELO
        opp_key = None
        for key in [opp_name.lower(), opp_name.lower().replace(' ', '')]:
            if key in name_lookup:
                opp_key = name_lookup[key]
                break
        
        if opp_key and opp_key in player_data:
            opp_elo = player_data[opp_key].get('elo_overall', 1500)
            
            if opp_elo >= strong_threshold:
                matches_vs_strong += 1
                if match.get('result') == 'W':
                    wins_vs_strong += 1
    
    win_pct_vs_strong = wins_vs_strong / matches_vs_strong if matches_vs_strong > 0 else 0.5
    
    return {
        'win_pct_vs_strong': win_pct_vs_strong,
        'matches_vs_strong': matches_vs_strong
    }


def calculate_elo_adjusted_form(matches, player_data, name_lookup, lookback=15):
    """
    Option 2: Opponent-adjusted form (weight wins by opponent ELO)
    
    A win against a 2100 ELO opponent is worth more than a win against 1400 ELO.
    Uses ELO difference to weight wins/losses.
    
    Returns:
        - adjusted_win_pct: Wins weighted by opponent strength
        - avg_opponent_elo: Average ELO of recent opponents
    """
    recent = matches[:lookback]
    
    weighted_wins = 0
    total_weight = 0
    opponent_elos = []
    
    for match in recent:
        opp_name = match.get('opponent', '')
        
        # Find opponent ELO
        opp_key = None
        for key in [opp_name.lower(), opp_name.lower().replace(' ', '')]:
            if key in name_lookup:
                opp_key = name_lookup[key]
                break
        
        if opp_key and opp_key in player_data:
            opp_elo = player_data[opp_key].get('elo_overall', 1500)
            opponent_elos.append(opp_elo)
            
            # Weight based on opponent strength (higher ELO = more weight)
            # Normalize around 1800 (roughly average top 200)
            weight = opp_elo / 1800
            
            total_weight += weight
            if match.get('result') == 'W':
                weighted_wins += weight
    
    adjusted_win_pct = weighted_wins / total_weight if total_weight > 0 else 0.5
    avg_opponent_elo = np.mean(opponent_elos) if opponent_elos else 1700
    
    return {
        'adjusted_win_pct': adjusted_win_pct,
        'avg_opponent_elo': avg_opponent_elo
    }


def calculate_level_jump(matches, player_data, name_lookup, current_opponent_elo, lookback=10):
    """
    Option 3: Level jump feature
    
    Measures how much harder the current opponent is compared to recent opponents.
    Positive = stepping up in competition
    Negative = facing easier competition than usual
    
    Returns:
        - level_jump: Current opponent ELO - avg recent opponent ELO
        - level_jump_pct: Level jump as percentage of avg
    """
    recent = matches[:lookback]
    
    opponent_elos = []
    
    for match in recent:
        opp_name = match.get('opponent', '')
        
        # Find opponent ELO
        opp_key = None
        for key in [opp_name.lower(), opp_name.lower().replace(' ', '')]:
            if key in name_lookup:
                opp_key = name_lookup[key]
                break
        
        if opp_key and opp_key in player_data:
            opp_elo = player_data[opp_key].get('elo_overall', 1500)
            opponent_elos.append(opp_elo)
    
    avg_recent_opp_elo = np.mean(opponent_elos) if opponent_elos else 1700
    
    level_jump = current_opponent_elo - avg_recent_opp_elo
    level_jump_pct = level_jump / avg_recent_opp_elo if avg_recent_opp_elo > 0 else 0
    
    return {
        'level_jump': level_jump,
        'level_jump_pct': level_jump_pct,
        'avg_recent_opp_elo': avg_recent_opp_elo
    }


# ============================================================
# BUILD TRAINING DATA WITH NEW FEATURES
# ============================================================

def build_training_data_with_elo_features(data, use_vs_strong=False, use_adjusted=False, use_level_jump=False):
    """Build training data with optional ELO-level features"""
    
    # Create player lookup
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
            
            # Find opponent
            opp_name = match.get('opponent', '')
            opp_key = None
            for key in [opp_name.lower(), opp_name.lower().replace(' ', '')]:
                if key in name_lookup:
                    opp_key = name_lookup[key]
                    break
            
            if not opp_key or opp_key not in player_data:
                continue
            
            opp_data = player_data[opp_key]
            match_date = match.get('date', '')
            opp_historical = [m for m in opp_data['matches'] if m.get('date', '') < match_date]
            
            if len(opp_historical) < 5:
                continue
            
            # Calculate base form
            p1_form = calculate_rolling_stats(historical_matches, 15)
            p2_form = calculate_rolling_stats(opp_historical, 15)
            
            if not p1_form or not p2_form:
                continue
            
            # Surface
            surface = match.get('surface', 'Hard').lower()
            surface_elo_key = f'elo_{surface}' if surface in ['hard', 'clay', 'grass'] else 'elo_hard'
            
            # Base features (simplified for experiment)
            features = [
                pdata['elo_overall'] - opp_data['elo_overall'],
                pdata.get(surface_elo_key, 1500) - opp_data.get(surface_elo_key, 1500),
                p1_form['win_pct'] - p2_form['win_pct'],
                p1_form['first_won_pct'] - p2_form['first_won_pct'],
                p1_form['rpw_pct'] - p2_form['rpw_pct'],
                p1_form['bp_saved_pct'] - p2_form['bp_saved_pct'],
            ]
            
            # Option 1: Performance vs strong opponents
            if use_vs_strong:
                p1_vs_strong = calculate_vs_strong_opponents(historical_matches, player_data, name_lookup)
                p2_vs_strong = calculate_vs_strong_opponents(opp_historical, player_data, name_lookup)
                
                features.extend([
                    p1_vs_strong['win_pct_vs_strong'] - p2_vs_strong['win_pct_vs_strong'],
                    p1_vs_strong['matches_vs_strong'] - p2_vs_strong['matches_vs_strong'],
                ])
            
            # Option 2: ELO-adjusted form
            if use_adjusted:
                p1_adj = calculate_elo_adjusted_form(historical_matches, player_data, name_lookup)
                p2_adj = calculate_elo_adjusted_form(opp_historical, player_data, name_lookup)
                
                features.extend([
                    p1_adj['adjusted_win_pct'] - p2_adj['adjusted_win_pct'],
                    p1_adj['avg_opponent_elo'] - p2_adj['avg_opponent_elo'],
                ])
            
            # Option 3: Level jump
            if use_level_jump:
                p1_jump = calculate_level_jump(historical_matches, player_data, name_lookup, opp_data['elo_overall'])
                p2_jump = calculate_level_jump(opp_historical, player_data, name_lookup, pdata['elo_overall'])
                
                features.extend([
                    p1_jump['level_jump'] - p2_jump['level_jump'],
                    p1_jump['level_jump_pct'] - p2_jump['level_jump_pct'],
                ])
            
            # Labels - parse score
            won = match.get('result') == 'W'
            score_str = match.get('score', '')
            p1_games, p2_games, total = parse_score(score_str)
            
            if p1_games is None:
                continue
            
            spread = p1_games - p2_games
            
            if total < 12 or total > 50:  # Filter bad data
                continue
            
            X.append(features)
            y.append(1 if won else 0)
            y_spread.append(spread)
            y_total.append(total)
    
    return np.array(X), np.array(y), np.array(y_spread), np.array(y_total)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_configuration(X, y, y_spread, y_total, config_name):
    """Evaluate a feature configuration across all 3 prediction types"""
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Winner prediction (Logistic Regression)
    winner_model = LogisticRegression(max_iter=1000, C=0.1)
    winner_pred_proba = cross_val_predict(winner_model, X_scaled, y, cv=kf, method='predict_proba')[:, 1]
    winner_pred = (winner_pred_proba > 0.5).astype(int)
    
    winner_acc = accuracy_score(y, winner_pred)
    winner_auc = roc_auc_score(y, winner_pred_proba)
    
    # Spread prediction
    spread_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    spread_pred = cross_val_predict(spread_model, X_scaled, y_spread, cv=kf)
    spread_mae = mean_absolute_error(y_spread, spread_pred)
    
    # Total games prediction
    total_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    total_pred = cross_val_predict(total_model, X_scaled, y_total, cv=kf)
    total_mae = mean_absolute_error(y_total, total_pred)
    
    # O/U accuracy
    ou_actual = (y_total > 21.5).astype(int)
    ou_pred = (total_pred > 21.5).astype(int)
    ou_acc = (ou_actual == ou_pred).mean()
    
    return {
        'config': config_name,
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'winner_acc': winner_acc,
        'winner_auc': winner_auc,
        'spread_mae': spread_mae,
        'total_mae': total_mae,
        'ou_acc': ou_acc
    }


def run_experiment():
    """Run full experiment comparing all configurations"""
    
    print("="*70)
    print("ELO LEVEL-ADJUSTED FEATURES EXPERIMENT")
    print("="*70)
    print()
    
    data = load_data()
    print(f"Loaded {data['player_count']} players with {data.get('total_matches', 0)} matches")
    
    # Quick sanity check
    sample_player = list(data['players'].keys())[0]
    sample_matches = data['players'][sample_player].get('matches', [])
    print(f"Sample player: {sample_player} has {len(sample_matches)} matches")
    if sample_matches:
        print(f"  Sample match keys: {list(sample_matches[0].keys())}")
    print()
    
    configurations = [
        {'name': 'Baseline', 'vs_strong': False, 'adjusted': False, 'level_jump': False},
        {'name': '+ Vs Strong', 'vs_strong': True, 'adjusted': False, 'level_jump': False},
        {'name': '+ Adjusted Form', 'vs_strong': False, 'adjusted': True, 'level_jump': False},
        {'name': '+ Level Jump', 'vs_strong': False, 'adjusted': False, 'level_jump': True},
        {'name': '+ All Three', 'vs_strong': True, 'adjusted': True, 'level_jump': True},
    ]
    
    results = []
    
    for config in configurations:
        print(f"Testing: {config['name']}...")
        
        X, y, y_spread, y_total = build_training_data_with_elo_features(
            data,
            use_vs_strong=config['vs_strong'],
            use_adjusted=config['adjusted'],
            use_level_jump=config['level_jump']
        )
        
        print(f"  Built {len(X)} samples")
        
        if len(X) == 0:
            print("  ERROR: No training samples generated!")
            continue
        
        result = evaluate_configuration(X, y, y_spread, y_total, config['name'])
        results.append(result)
        
        print(f"  Samples: {result['n_samples']}, Features: {result['n_features']}")
    
    # Print results table
    print()
    print("="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    print()
    print(f"{'Configuration':<20} {'Winner':>12} {'AUC':>8} {'Spread':>10} {'Total':>10} {'O/U 21.5':>10}")
    print(f"{'':20} {'Accuracy':>12} {'':>8} {'MAE':>10} {'MAE':>10} {'Accuracy':>10}")
    print("-"*70)
    
    baseline = results[0]
    
    for r in results:
        # Calculate deltas from baseline
        acc_delta = r['winner_acc'] - baseline['winner_acc']
        auc_delta = r['winner_auc'] - baseline['winner_auc']
        spread_delta = r['spread_mae'] - baseline['spread_mae']  # Lower is better
        total_delta = r['total_mae'] - baseline['total_mae']  # Lower is better
        ou_delta = r['ou_acc'] - baseline['ou_acc']
        
        print(f"{r['config']:<20} {r['winner_acc']:>10.1%}   {r['winner_auc']:>7.3f} {r['spread_mae']:>9.2f}  {r['total_mae']:>9.2f}  {r['ou_acc']:>9.1%}")
        
        if r['config'] != 'Baseline':
            # Show deltas
            acc_str = f"+{acc_delta:.1%}" if acc_delta > 0 else f"{acc_delta:.1%}"
            auc_str = f"+{auc_delta:.3f}" if auc_delta > 0 else f"{auc_delta:.3f}"
            spread_str = f"{spread_delta:+.2f}" if spread_delta != 0 else "0.00"
            total_str = f"{total_delta:+.2f}" if total_delta != 0 else "0.00"
            ou_str = f"+{ou_delta:.1%}" if ou_delta > 0 else f"{ou_delta:.1%}"
            
            # Color coding (in text form)
            acc_marker = "(+)" if acc_delta > 0.001 else "(-)" if acc_delta < -0.001 else "(=)"
            spread_marker = "(+)" if spread_delta < -0.01 else "(-)" if spread_delta > 0.01 else "(=)"
            total_marker = "(+)" if total_delta < -0.01 else "(-)" if total_delta > 0.01 else "(=)"
            
            print(f"{'  vs Baseline':<20} {acc_str:>10} {acc_marker} {auc_str:>7}  {spread_str:>8} {spread_marker} {total_str:>8} {total_marker} {ou_str:>9}")
    
    print()
    print("="*70)
    print("INTERPRETATION")
    print("="*70)
    
    # Find best configuration for each metric
    best_winner = max(results, key=lambda x: x['winner_auc'])
    best_spread = min(results, key=lambda x: x['spread_mae'])
    best_total = min(results, key=lambda x: x['total_mae'])
    best_ou = max(results, key=lambda x: x['ou_acc'])
    
    print()
    print(f"Best for Winner (AUC):    {best_winner['config']} ({best_winner['winner_auc']:.3f})")
    print(f"Best for Spread (MAE):    {best_spread['config']} ({best_spread['spread_mae']:.2f})")
    print(f"Best for Total (MAE):     {best_total['config']} ({best_total['total_mae']:.2f})")
    print(f"Best for O/U 21.5:        {best_ou['config']} ({best_ou['ou_acc']:.1%})")
    
    return results


if __name__ == '__main__':
    results = run_experiment()
