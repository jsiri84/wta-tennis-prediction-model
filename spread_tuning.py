"""
Game Spread Model Tuning - Testing additional features to reduce error
"""

import json
import numpy as np
import re
from collections import defaultdict
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def parse_score(score_str):
    """Parse score and return detailed breakdown"""
    if not score_str:
        return None
    
    score_upper = score_str.upper()
    if any(x in score_upper for x in ['RET', 'W/O', 'WO', 'DEF', 'ABD']):
        return None
    
    player_games = 0
    opp_games = 0
    sets_won = 0
    sets_lost = 0
    tiebreaks = 0
    
    sets = score_str.strip().split()
    
    for s in sets:
        has_tb = '(' in s
        if has_tb:
            tiebreaks += 1
        
        s_clean = re.sub(r'\(\d+\)', '', s)
        match = re.match(r'(\d+)-(\d+)', s_clean)
        if match:
            p_set = int(match.group(1))
            o_set = int(match.group(2))
            player_games += p_set
            opp_games += o_set
            if p_set > o_set:
                sets_won += 1
            else:
                sets_lost += 1
    
    if player_games == 0 and opp_games == 0:
        return None
    
    return {
        'player_games': player_games,
        'opp_games': opp_games,
        'spread': player_games - opp_games,
        'total_games': player_games + opp_games,
        'sets_won': sets_won,
        'sets_lost': sets_lost,
        'total_sets': sets_won + sets_lost,
        'tiebreaks': tiebreaks,
        'straight_sets': sets_lost == 0
    }


def load_data():
    with open('all_players_matches.json', 'r') as f:
        return json.load(f)


def calculate_advanced_stats(matches, lookback=15):
    """Calculate advanced stats including spread-related metrics"""
    if not matches:
        return None
    
    recent = matches[:lookback]
    
    wins = sum(1 for m in recent if m['result'] == 'W')
    win_pct = wins / len(recent) if recent else 0.5
    
    # Spread stats
    spreads = []
    total_games_list = []
    straight_set_wins = 0
    straight_set_losses = 0
    tiebreak_count = 0
    three_setters = 0
    
    for m in recent:
        parsed = parse_score(m.get('score', ''))
        if parsed:
            spreads.append(parsed['spread'])
            total_games_list.append(parsed['total_games'])
            tiebreak_count += parsed['tiebreaks']
            if parsed['total_sets'] == 3:
                three_setters += 1
            if m['result'] == 'W' and parsed['straight_sets']:
                straight_set_wins += 1
            elif m['result'] == 'L' and parsed['sets_won'] == 0:
                straight_set_losses += 1
    
    # Serve/Return stats
    serve_stats = defaultdict(list)
    return_stats = defaultdict(list)
    
    for m in recent:
        serve = m.get('serve', {})
        ret = m.get('return', {})
        
        for key in ['first_in_pct', 'first_won_pct', 'second_won_pct', 'ace_pct', 'df_pct', 'bp_saved_pct']:
            if serve.get(key) is not None:
                serve_stats[key].append(serve[key])
        
        for key in ['rpw_pct', 'bp_conv_pct']:
            if ret.get(key) is not None:
                return_stats[key].append(ret[key])
    
    # Dominance ratio
    dr_values = []
    for m in recent:
        serve = m.get('serve', {})
        ret = m.get('return', {})
        spw = (serve.get('first_won_pct') or 0) * 0.6 + (serve.get('second_won_pct') or 0) * 0.4
        rpw = ret.get('rpw_pct') or 0
        if rpw > 0 and (100 - rpw) > 0:
            dr_values.append(spw / (100 - rpw))
    
    # Service games hold percentage (approximation from first serve won)
    hold_pct = np.mean(serve_stats['first_won_pct']) if serve_stats['first_won_pct'] else 65
    
    # Break percentage (approximation from return points won)
    break_pct = np.mean(return_stats['bp_conv_pct']) if return_stats['bp_conv_pct'] else 40
    
    return {
        'win_pct': win_pct,
        'avg_dr': np.mean(dr_values) if dr_values else 1.0,
        
        # Serve stats
        'first_in_pct': np.mean(serve_stats['first_in_pct']) if serve_stats['first_in_pct'] else 60,
        'first_won_pct': np.mean(serve_stats['first_won_pct']) if serve_stats['first_won_pct'] else 65,
        'second_won_pct': np.mean(serve_stats['second_won_pct']) if serve_stats['second_won_pct'] else 50,
        'ace_pct': np.mean(serve_stats['ace_pct']) if serve_stats['ace_pct'] else 5,
        'df_pct': np.mean(serve_stats['df_pct']) if serve_stats['df_pct'] else 3,
        'bp_saved_pct': np.mean(serve_stats['bp_saved_pct']) if serve_stats['bp_saved_pct'] else 60,
        
        # Return stats
        'rpw_pct': np.mean(return_stats['rpw_pct']) if return_stats['rpw_pct'] else 35,
        'bp_conv_pct': np.mean(return_stats['bp_conv_pct']) if return_stats['bp_conv_pct'] else 40,
        
        # NEW: Spread-specific stats
        'avg_spread': np.mean(spreads) if spreads else 0,
        'spread_std': np.std(spreads) if len(spreads) > 1 else 3,
        'avg_total_games': np.mean(total_games_list) if total_games_list else 20,
        'straight_set_rate': (straight_set_wins + straight_set_losses) / len(recent) if recent else 0.5,
        'tiebreak_rate': tiebreak_count / len(recent) if recent else 0.1,
        'three_set_rate': three_setters / len(recent) if recent else 0.3,
        
        # Service/break game approximations
        'hold_pct': hold_pct,
        'break_pct': break_pct,
    }


def build_features(data, feature_set='basic'):
    """Build training data with specified feature set"""
    
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
    
    for player_name, pdata in player_data.items():
        for i, match in enumerate(pdata['matches']):
            if i < 5:
                continue
            
            parsed = parse_score(match.get('score', ''))
            if not parsed:
                continue
            
            game_spread = parsed['spread']
            
            opp_name = match.get('opponent', '')
            opp_key = opp_name.lower().replace(' ', '')
            
            if opp_key not in name_lookup:
                last_name = opp_name.split()[-1].lower() if opp_name else ''
                if last_name in name_lookup:
                    opp_key = last_name
                else:
                    continue
            
            opp_full_name = name_lookup.get(opp_key)
            if not opp_full_name or opp_full_name not in player_data:
                continue
            
            opp_data = player_data[opp_full_name]
            
            match_date = match.get('date', '')
            historical = [m for m in pdata['matches'][i+1:] if m.get('date', '') < match_date]
            opp_historical = [m for m in opp_data['matches'] if m.get('date', '') < match_date]
            
            if len(historical) < 5 or len(opp_historical) < 5:
                continue
            
            surface = match.get('surface', 'Hard').lower()
            surface_elo_key = f'elo_{surface}' if surface in ['hard', 'clay', 'grass'] else 'elo_hard'
            
            p1 = calculate_advanced_stats(historical, 15)
            p2 = calculate_advanced_stats(opp_historical, 15)
            
            if not p1 or not p2:
                continue
            
            # Build features based on feature set
            if feature_set == 'basic':
                features = [
                    pdata['elo_overall'] - opp_data['elo_overall'],
                    pdata.get(surface_elo_key, 1500) - opp_data.get(surface_elo_key, 1500),
                    p1['win_pct'] - p2['win_pct'],
                    p1['avg_dr'] - p2['avg_dr'],
                    p1['first_won_pct'] - p2['first_won_pct'],
                    p1['ace_pct'] - p2['ace_pct'],
                    p1['bp_saved_pct'] - p2['bp_saved_pct'],
                    p1['rpw_pct'] - p2['rpw_pct'],
                    p1['bp_conv_pct'] - p2['bp_conv_pct'],
                    p1['avg_spread'] - p2['avg_spread'],
                ]
            
            elif feature_set == 'spread_focused':
                features = [
                    # ELO
                    pdata['elo_overall'] - opp_data['elo_overall'],
                    pdata.get(surface_elo_key, 1500) - opp_data.get(surface_elo_key, 1500),
                    
                    # Historical spread performance (key!)
                    p1['avg_spread'] - p2['avg_spread'],
                    p1['avg_spread'],
                    p2['avg_spread'],
                    p1['spread_std'],
                    p2['spread_std'],
                    
                    # Match length tendencies
                    p1['avg_total_games'] - p2['avg_total_games'],
                    p1['straight_set_rate'] - p2['straight_set_rate'],
                    p1['three_set_rate'] - p2['three_set_rate'],
                    p1['tiebreak_rate'] - p2['tiebreak_rate'],
                    
                    # Dominance
                    p1['avg_dr'] - p2['avg_dr'],
                    p1['win_pct'] - p2['win_pct'],
                ]
            
            elif feature_set == 'full':
                features = [
                    # ELO (3)
                    pdata['elo_overall'] - opp_data['elo_overall'],
                    pdata.get(surface_elo_key, 1500) - opp_data.get(surface_elo_key, 1500),
                    (pdata['elo_overall'] + opp_data['elo_overall']) / 2,  # Match quality
                    
                    # Historical spread (5)
                    p1['avg_spread'] - p2['avg_spread'],
                    p1['avg_spread'],
                    p2['avg_spread'],
                    p1['spread_std'] + p2['spread_std'],  # Combined volatility
                    abs(p1['avg_spread']) + abs(p2['avg_spread']),  # Combined dominance
                    
                    # Match tendencies (4)
                    p1['straight_set_rate'] - p2['straight_set_rate'],
                    p1['three_set_rate'] + p2['three_set_rate'],  # Combined 3-set tendency
                    p1['tiebreak_rate'] + p2['tiebreak_rate'],  # Combined tiebreak tendency
                    p1['avg_total_games'] + p2['avg_total_games'],  # Expected match length
                    
                    # Dominance metrics (4)
                    p1['avg_dr'] - p2['avg_dr'],
                    p1['win_pct'] - p2['win_pct'],
                    p1['hold_pct'] - p2['hold_pct'],
                    p1['break_pct'] - p2['break_pct'],
                    
                    # Serve power (3)
                    p1['first_won_pct'] - p2['first_won_pct'],
                    p1['ace_pct'] - p2['ace_pct'],
                    p1['second_won_pct'] - p2['second_won_pct'],
                    
                    # Consistency (2)
                    p1['df_pct'] - p2['df_pct'],
                    p1['bp_saved_pct'] - p2['bp_saved_pct'],
                    
                    # Return (2)
                    p1['rpw_pct'] - p2['rpw_pct'],
                    p1['bp_conv_pct'] - p2['bp_conv_pct'],
                ]
            
            X.append(features)
            y.append(game_spread)
    
    return np.array(X), np.array(y)


def test_feature_sets():
    """Compare different feature sets"""
    data = load_data()
    print(f"Loaded {data['player_count']} players")
    
    feature_sets = ['basic', 'spread_focused', 'full']
    
    print("\n" + "="*70)
    print("FEATURE SET COMPARISON FOR GAME SPREAD PREDICTION")
    print("="*70)
    
    results = []
    
    for fs in feature_sets:
        X, y = build_features(data, fs)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        
        # Direction accuracy
        direction_acc = np.mean((pred > 0) == (y_test > 0))
        
        # Margin accuracy (within 2 games)
        within_2 = np.mean(np.abs(pred - y_test) <= 2)
        within_3 = np.mean(np.abs(pred - y_test) <= 3)
        
        results.append({
            'feature_set': fs,
            'n_features': X.shape[1],
            'mae': mae,
            'rmse': rmse,
            'direction_acc': direction_acc,
            'within_2': within_2,
            'within_3': within_3
        })
        
        print(f"\n{fs.upper()} ({X.shape[1]} features):")
        print(f"  MAE:  {mae:.2f} games")
        print(f"  RMSE: {rmse:.2f} games")
        print(f"  Direction accuracy: {direction_acc:.1%}")
        print(f"  Within 2 games: {within_2:.1%}")
        print(f"  Within 3 games: {within_3:.1%}")
    
    # Find best
    best = min(results, key=lambda x: x['mae'])
    print(f"\n{'='*70}")
    print(f"BEST: {best['feature_set'].upper()} with MAE={best['mae']:.2f}")
    print(f"{'='*70}")
    
    return results


def tune_hyperparameters():
    """Tune model hyperparameters"""
    data = load_data()
    X, y = build_features(data, 'full')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING")
    print("="*70)
    
    # Test different configurations
    configs = [
        {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1},
        {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1},
        {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.05},
        {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05},
        {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.15},
    ]
    
    best_mae = float('inf')
    best_config = None
    
    for config in configs:
        model = GradientBoostingRegressor(random_state=42, **config)
        model.fit(X_train_scaled, y_train)
        
        pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, pred)
        
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        
        print(f"\n{config}:")
        print(f"  Test MAE: {mae:.3f}")
        print(f"  CV MAE:   {cv_mae:.3f}")
        
        if mae < best_mae:
            best_mae = mae
            best_config = config
    
    print(f"\n{'='*70}")
    print(f"BEST CONFIG: {best_config}")
    print(f"BEST MAE: {best_mae:.3f}")
    print(f"{'='*70}")


def test_ensemble():
    """Test ensemble of models"""
    data = load_data()
    X, y = build_features(data, 'full')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n" + "="*70)
    print("ENSEMBLE TESTING")
    print("="*70)
    
    # Train multiple models
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    
    gb.fit(X_train_scaled, y_train)
    rf.fit(X_train_scaled, y_train)
    
    gb_pred = gb.predict(X_test_scaled)
    rf_pred = rf.predict(X_test_scaled)
    
    # Test different ensemble weights
    weights = [(1.0, 0.0), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.0, 1.0)]
    
    for w_gb, w_rf in weights:
        ensemble_pred = w_gb * gb_pred + w_rf * rf_pred
        mae = mean_absolute_error(y_test, ensemble_pred)
        print(f"GB:{w_gb:.1f} RF:{w_rf:.1f} -> MAE: {mae:.3f}")


if __name__ == '__main__':
    # Compare feature sets
    results = test_feature_sets()
    
    # Tune hyperparameters
    tune_hyperparameters()
    
    # Test ensemble
    test_ensemble()
