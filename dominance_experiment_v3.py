"""
Experiment v3: Dominance at OPPONENT'S Level

Key insight: When Player A (ELO 2100) faces Player B (ELO 1850):
- How does Player A perform against ~1850 ELO opponents?
- How does Player B perform against ~2100 ELO opponents?

This directly measures: "Can the favorite dominate at this level?"
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
    """
    Calculate how a player performs against opponents near a TARGET ELO level.
    
    Args:
        target_elo: The ELO level we care about (opponent's current ELO)
        elo_range: How wide the range (+/- from target)
    
    Returns performance metrics when playing opponents in [target_elo - range, target_elo + range]
    """
    recent = matches[:lookback]
    
    spreads = []
    wins = 0
    matches_at_level = 0
    serve_pcts = []
    return_pcts = []
    bp_conv = []
    bp_saved = []
    
    for match in recent:
        opp_name = match.get('opponent', '')
        opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
        
        if not opp_key or opp_key not in player_data:
            continue
        
        opp_elo = player_data[opp_key].get('elo_overall', player_data[opp_key].get('elo', 1500))
        
        # Check if opponent is near target level
        if abs(opp_elo - target_elo) <= elo_range:
            matches_at_level += 1
            
            if match.get('result') == 'W':
                wins += 1
            
            # Spread
            sets = re.findall(r'(\d+)-(\d+)', match.get('score', ''))
            if sets:
                p1 = sum(int(s[0]) for s in sets)
                p2 = sum(int(s[1]) for s in sets)
                spreads.append(p1 - p2)
            
            # Serve stats
            serve = match.get('serve', {})
            if serve:
                fw = serve.get('first_won_pct') or 0
                sw = serve.get('second_won_pct') or 0
                if fw or sw:
                    serve_pcts.append(fw * 0.6 + sw * 0.4)
                if serve.get('bp_saved_pct'):
                    bp_saved.append(serve['bp_saved_pct'])
            
            # Return stats
            ret = match.get('return', {})
            if ret:
                if ret.get('rpw_pct'):
                    return_pcts.append(ret['rpw_pct'])
                if ret.get('bp_conv_pct'):
                    bp_conv.append(ret['bp_conv_pct'])
    
    return {
        'win_pct_at_level': wins / matches_at_level if matches_at_level > 0 else 0.5,
        'avg_spread_at_level': np.mean(spreads) if spreads else 0,
        'serve_pct_at_level': np.mean(serve_pcts) if serve_pcts else 55,
        'return_pct_at_level': np.mean(return_pcts) if return_pcts else 35,
        'bp_conv_at_level': np.mean(bp_conv) if bp_conv else 40,
        'bp_saved_at_level': np.mean(bp_saved) if bp_saved else 60,
        'matches_at_level': matches_at_level
    }


def calculate_momentum_form(matches, lookback=5):
    """
    Recent momentum - last 5 matches only.
    Captures hot/cold streaks.
    """
    recent = matches[:lookback]
    if len(recent) < 3:
        return None
    
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
        'win_streak': sum(1 for m in recent if m.get('result') == 'W'),  # Simplistic but useful
    }


def calculate_upset_tendency(matches, player_elo, player_data, name_lookup, lookback=20):
    """
    How often does this player upset higher-ranked opponents?
    And how often do they get upset by lower-ranked?
    """
    recent = matches[:lookback]
    
    upsets_caused = 0  # Beat higher ELO
    upsets_suffered = 0  # Lost to lower ELO
    matches_vs_higher = 0
    matches_vs_lower = 0
    
    for match in recent:
        opp_name = match.get('opponent', '')
        opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
        
        if not opp_key or opp_key not in player_data:
            continue
        
        opp_elo = player_data[opp_key].get('elo_overall', player_data[opp_key].get('elo', 1500))
        won = match.get('result') == 'W'
        
        if opp_elo > player_elo + 50:  # Faced higher-ranked
            matches_vs_higher += 1
            if won:
                upsets_caused += 1
        elif opp_elo < player_elo - 50:  # Faced lower-ranked
            matches_vs_lower += 1
            if not won:
                upsets_suffered += 1
    
    return {
        'upset_rate': upsets_caused / matches_vs_higher if matches_vs_higher > 0 else 0.3,
        'upset_suffered_rate': upsets_suffered / matches_vs_lower if matches_vs_lower > 0 else 0.1,
        'matches_vs_higher': matches_vs_higher,
        'matches_vs_lower': matches_vs_lower
    }


def build_data(data, feature_set='baseline'):
    """
    Build training data with different feature sets:
    - baseline: Current model features
    - at_level: + Performance at opponent's level
    - momentum: + Recent momentum
    - upset: + Upset tendencies
    - all: All features
    """
    
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
            
            # Calculate base stats
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
            
            # Base features (simplified)
            features = [
                pdata['elo_overall'] - opp_data['elo_overall'],
                pdata.get(surface_key, 1500) - opp_data.get(surface_key, 1500),
                p1_form['win_pct'] - p2_form['win_pct'],
                p1_form['first_won_pct'] - p2_form['first_won_pct'],
                p1_form['rpw_pct'] - p2_form['rpw_pct'],
                (p1_surf['win_pct'] if p1_surf else 0.5) - (p2_surf['win_pct'] if p2_surf else 0.5),
                p1_h2h['diff'],
            ]
            
            # Performance at opponent's level
            if feature_set in ['at_level', 'all']:
                # P1's performance against opponents at P2's ELO level
                p1_at_p2_level = calculate_performance_at_level(historical, player_data, name_lookup, opp_data['elo_overall'])
                # P2's performance against opponents at P1's ELO level  
                p2_at_p1_level = calculate_performance_at_level(opp_hist, player_data, name_lookup, pdata['elo_overall'])
                
                features.extend([
                    p1_at_p2_level['win_pct_at_level'] - p2_at_p1_level['win_pct_at_level'],
                    p1_at_p2_level['avg_spread_at_level'] - p2_at_p1_level['avg_spread_at_level'],
                    p1_at_p2_level['serve_pct_at_level'] - p2_at_p1_level['serve_pct_at_level'],
                    p1_at_p2_level['return_pct_at_level'] - p2_at_p1_level['return_pct_at_level'],
                    p1_at_p2_level['bp_conv_at_level'] - p2_at_p1_level['bp_conv_at_level'],
                    p1_at_p2_level['matches_at_level'] - p2_at_p1_level['matches_at_level'],
                ])
            
            # Momentum
            if feature_set in ['momentum', 'all']:
                p1_mom = calculate_momentum_form(historical, 5)
                p2_mom = calculate_momentum_form(opp_hist, 5)
                
                if p1_mom and p2_mom:
                    features.extend([
                        p1_mom['recent_win_pct'] - p2_mom['recent_win_pct'],
                        p1_mom['recent_avg_spread'] - p2_mom['recent_avg_spread'],
                    ])
                else:
                    features.extend([0, 0])
            
            # Upset tendency
            if feature_set in ['upset', 'all']:
                p1_upset = calculate_upset_tendency(historical, pdata['elo_overall'], player_data, name_lookup)
                p2_upset = calculate_upset_tendency(opp_hist, opp_data['elo_overall'], player_data, name_lookup)
                
                features.extend([
                    p1_upset['upset_rate'] - p2_upset['upset_rate'],
                    p2_upset['upset_suffered_rate'] - p1_upset['upset_suffered_rate'],  # Reversed
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
    
    return winner_acc, winner_auc, spread_mae


def main():
    print("="*70)
    print("ALTERNATIVE FEATURES EXPERIMENT")
    print("="*70)
    print()
    
    data = load_data()
    print(f"Loaded {data['player_count']} players")
    print()
    
    configs = [
        ('Baseline', 'baseline'),
        ('+ At Opponent Level', 'at_level'),
        ('+ Momentum', 'momentum'),
        ('+ Upset Tendency', 'upset'),
        ('+ ALL New Features', 'all'),
    ]
    
    results = []
    
    for name, feature_set in configs:
        print(f"Testing: {name}...")
        X, y, y_spread, y_total = build_data(data, feature_set)
        print(f"  {X.shape[0]} samples, {X.shape[1]} features")
        
        acc, auc, spread_mae = evaluate(X, y, y_spread, y_total)
        results.append((name, X.shape[1], acc, auc, spread_mae))
    
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()
    print(f"{'Config':<25} {'Feat':>5} {'Win Acc':>9} {'AUC':>8} {'Spread MAE':>11}")
    print("-"*60)
    
    baseline = results[0]
    for name, feat, acc, auc, spread in results:
        print(f"{name:<25} {feat:>5} {acc:>8.1%} {auc:>8.3f} {spread:>10.2f}")
    
    print()
    print("DELTA FROM BASELINE:")
    print("-"*60)
    for name, feat, acc, auc, spread in results[1:]:
        print(f"{name:<25}       {(acc-baseline[2])*100:>+7.2f}% {auc-baseline[3]:>+8.4f} {spread-baseline[4]:>+10.3f}")
    
    print()
    print("="*70)
    print("NEW FEATURES TESTED")
    print("="*70)
    print()
    print("1. AT OPPONENT'S LEVEL (6 features)")
    print("   How does P1 perform against opponents at P2's ELO?")
    print("   How does P2 perform against opponents at P1's ELO?")
    print("   - win_pct, spread, serve%, return%, bp_conv, experience")
    print()
    print("2. MOMENTUM (2 features)")
    print("   Last 5 matches only - captures hot/cold streaks")
    print("   - recent_win_pct, recent_avg_spread")
    print()
    print("3. UPSET TENDENCY (2 features)")
    print("   - upset_rate: How often do they beat higher-ranked players?")
    print("   - upset_suffered_rate: How often do they lose to lower-ranked?")


if __name__ == '__main__':
    main()
