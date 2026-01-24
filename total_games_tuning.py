"""
Total Games Model Tuning

Experiments with features and hyperparameters to improve accuracy.
"""

import json
import numpy as np
import re
from collections import defaultdict
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def parse_score(score_str):
    """Parse score and return game counts"""
    if not score_str:
        return None
    
    score_upper = score_str.upper()
    if any(x in score_upper for x in ['RET', 'W/O', 'WO', 'DEF', 'ABD']):
        return None
    
    p_games, o_games = 0, 0
    tiebreaks = 0
    sets = score_str.strip().split()
    
    for s in sets:
        tiebreaks += s.count('(')
        s_clean = re.sub(r'\(\d+\)', '', s)
        match = re.match(r'(\d+)-(\d+)', s_clean)
        if match:
            p_games += int(match.group(1))
            o_games += int(match.group(2))
    
    if p_games == 0 and o_games == 0:
        return None
    
    return {
        'p_games': p_games,
        'o_games': o_games,
        'total': p_games + o_games,
        'sets': len(sets),
        'tiebreaks': tiebreaks,
        'is_straight_sets': len(sets) == 2
    }


def load_data():
    with open('all_players_matches.json', 'r') as f:
        return json.load(f)


def get_tournament_level(tourn):
    tourn_lower = tourn.lower()
    if any(gs in tourn_lower for gs in ['australian open', 'french open', 'roland garros', 'wimbledon', 'us open']):
        return 4
    if '1000' in tourn_lower or any(t in tourn_lower for t in ['indian wells', 'miami', 'madrid', 'rome', 'beijing', 'canada']):
        return 3
    if '500' in tourn_lower:
        return 2
    return 1


def get_round_level(round_str):
    round_map = {'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4, 'QF': 5, 'SF': 6, 'F': 7, 'RR': 2}
    return round_map.get(round_str, 3)


def calculate_player_stats(matches, lookback=15):
    """Calculate player stats relevant to total games"""
    if not matches or len(matches) < 3:
        return None
    
    recent = matches[:lookback]
    wins = sum(1 for m in recent if m['result'] == 'W')
    
    # Game stats
    totals = []
    tiebreaks = 0
    straight_sets = 0
    three_sets = 0
    player_games = []
    opponent_games = []
    close_sets = 0  # 6-4, 7-5, or tiebreaks
    
    for m in recent:
        parsed = parse_score(m.get('score', ''))
        if parsed:
            totals.append(parsed['total'])
            player_games.append(parsed['p_games'])
            opponent_games.append(parsed['o_games'])
            tiebreaks += parsed['tiebreaks']
            if parsed['is_straight_sets']:
                straight_sets += 1
            elif parsed['sets'] == 3:
                three_sets += 1
            
            # Count close sets from score
            score = m.get('score', '')
            close_sets += score.count('6-4') + score.count('4-6')
            close_sets += score.count('7-5') + score.count('5-7')
            close_sets += score.count('7-6') + score.count('6-7')
    
    # Serve stats
    serve_stats = defaultdict(list)
    return_stats = defaultdict(list)
    
    for m in recent:
        serve = m.get('serve', {})
        ret = m.get('return', {})
        
        for key in ['first_in_pct', 'first_won_pct', 'second_won_pct', 'ace_pct', 'df_pct', 'bp_saved_pct', 'dr']:
            if serve.get(key) is not None:
                serve_stats[key].append(serve[key])
        
        for key in ['rpw_pct', 'bp_conv_pct']:
            if ret.get(key) is not None:
                return_stats[key].append(ret[key])
    
    n = len(recent)
    n_parsed = len(totals) if totals else 1
    
    hold_rate = np.mean(serve_stats['first_won_pct']) * 0.6 + np.mean(serve_stats['second_won_pct']) * 0.4 if serve_stats['first_won_pct'] else 55
    break_rate = np.mean(return_stats['bp_conv_pct']) if return_stats['bp_conv_pct'] else 40
    
    return {
        'win_pct': wins / n,
        'avg_total_games': np.mean(totals) if totals else 20,
        'total_games_std': np.std(totals) if len(totals) > 1 else 3,
        'min_total': min(totals) if totals else 12,
        'max_total': max(totals) if totals else 26,
        'straight_set_rate': straight_sets / n_parsed,
        'three_set_rate': three_sets / n_parsed,
        'tiebreak_rate': tiebreaks / n,
        'close_set_rate': close_sets / (n_parsed * 2),  # Per set average
        
        # Average games per set
        'avg_player_games': np.mean(player_games) if player_games else 10,
        'avg_opponent_games': np.mean(opponent_games) if opponent_games else 10,
        
        # Serve stats
        'first_in_pct': np.mean(serve_stats['first_in_pct']) if serve_stats['first_in_pct'] else 60,
        'first_won_pct': np.mean(serve_stats['first_won_pct']) if serve_stats['first_won_pct'] else 65,
        'second_won_pct': np.mean(serve_stats['second_won_pct']) if serve_stats['second_won_pct'] else 50,
        'ace_pct': np.mean(serve_stats['ace_pct']) if serve_stats['ace_pct'] else 5,
        'df_pct': np.mean(serve_stats['df_pct']) if serve_stats['df_pct'] else 3,
        'bp_saved_pct': np.mean(serve_stats['bp_saved_pct']) if serve_stats['bp_saved_pct'] else 60,
        'dr': np.mean(serve_stats['dr']) if serve_stats['dr'] else 1.0,
        
        # Return stats
        'rpw_pct': np.mean(return_stats['rpw_pct']) if return_stats['rpw_pct'] else 35,
        'bp_conv_pct': np.mean(return_stats['bp_conv_pct']) if return_stats['bp_conv_pct'] else 40,
        
        # Derived
        'hold_rate': hold_rate,
        'break_rate': break_rate,
    }


def build_features(data, feature_set='full'):
    """Build features for total games prediction"""
    
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
    
    X, y = [], []
    
    for player_name, pdata in player_data.items():
        for i, match in enumerate(pdata['matches']):
            if i < 5:
                continue
            
            parsed = parse_score(match.get('score', ''))
            if not parsed:
                continue
            
            total_games = parsed['total']
            
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
            
            if not p1 or not p2:
                continue
            
            tourn_level = get_tournament_level(match.get('tournament', ''))
            round_level = get_round_level(match.get('round', 'R32'))
            
            # Build features based on set
            if feature_set == 'elo_focused':
                features = [
                    abs(pdata['elo'] - opp_data['elo']),
                    abs(pdata.get(surface_key, 1500) - opp_data.get(surface_key, 1500)),
                    (pdata['elo'] + opp_data['elo']) / 2,
                    min(pdata['elo'], opp_data['elo']),
                    pdata['elo'] - opp_data['elo'],
                ]
                
            elif feature_set == 'historical':
                features = [
                    (p1['avg_total_games'] + p2['avg_total_games']) / 2,
                    p1['avg_total_games'],
                    p2['avg_total_games'],
                    p1['total_games_std'] + p2['total_games_std'],
                    (p1['max_total'] + p2['max_total']) / 2,
                    (p1['min_total'] + p2['min_total']) / 2,
                    abs(p1['avg_total_games'] - p2['avg_total_games']),
                ]
                
            elif feature_set == 'match_type':
                features = [
                    p1['three_set_rate'] + p2['three_set_rate'],
                    p1['straight_set_rate'] + p2['straight_set_rate'],
                    p1['tiebreak_rate'] + p2['tiebreak_rate'],
                    p1['close_set_rate'] + p2['close_set_rate'],
                    abs(p1['win_pct'] - p2['win_pct']),
                ]
                
            elif feature_set == 'serve_return':
                features = [
                    (p1['hold_rate'] + p2['hold_rate']) / 2,
                    (p1['first_won_pct'] + p2['first_won_pct']) / 2,
                    (p1['ace_pct'] + p2['ace_pct']) / 2,
                    (p1['bp_saved_pct'] + p2['bp_saved_pct']) / 2,
                    (p1['break_rate'] + p2['break_rate']) / 2,
                    (p1['rpw_pct'] + p2['rpw_pct']) / 2,
                    (p1['bp_conv_pct'] + p2['bp_conv_pct']) / 2,
                    (p1['dr'] + p2['dr']) / 2,
                    abs(p1['dr'] - p2['dr']),
                ]
            
            elif feature_set == 'full':
                features = [
                    # ELO features (5)
                    abs(pdata['elo'] - opp_data['elo']),
                    abs(pdata.get(surface_key, 1500) - opp_data.get(surface_key, 1500)),
                    (pdata['elo'] + opp_data['elo']) / 2,
                    min(pdata['elo'], opp_data['elo']),
                    pdata['elo'] - opp_data['elo'],
                    
                    # Historical total games (7)
                    (p1['avg_total_games'] + p2['avg_total_games']) / 2,
                    p1['avg_total_games'],
                    p2['avg_total_games'],
                    p1['total_games_std'] + p2['total_games_std'],
                    (p1['max_total'] + p2['max_total']) / 2,
                    (p1['min_total'] + p2['min_total']) / 2,
                    abs(p1['avg_total_games'] - p2['avg_total_games']),
                    
                    # Match type tendencies (5)
                    p1['three_set_rate'] + p2['three_set_rate'],
                    p1['straight_set_rate'] + p2['straight_set_rate'],
                    p1['tiebreak_rate'] + p2['tiebreak_rate'],
                    p1['close_set_rate'] + p2['close_set_rate'],
                    abs(p1['win_pct'] - p2['win_pct']),
                    
                    # Serve/Return (9)
                    (p1['hold_rate'] + p2['hold_rate']) / 2,
                    (p1['first_won_pct'] + p2['first_won_pct']) / 2,
                    (p1['ace_pct'] + p2['ace_pct']) / 2,
                    (p1['bp_saved_pct'] + p2['bp_saved_pct']) / 2,
                    (p1['break_rate'] + p2['break_rate']) / 2,
                    (p1['rpw_pct'] + p2['rpw_pct']) / 2,
                    (p1['bp_conv_pct'] + p2['bp_conv_pct']) / 2,
                    (p1['dr'] + p2['dr']) / 2,
                    abs(p1['dr'] - p2['dr']),
                    
                    # Context (3)
                    tourn_level,
                    round_level,
                    tourn_level * round_level,
                    
                    # Surface (3)
                    1 if surface == 'hard' else 0,
                    1 if surface == 'clay' else 0,
                    1 if surface == 'grass' else 0,
                ]
                
            elif feature_set == 'optimized':
                # Based on feature importance results - focus on top features
                features = [
                    # Top ELO features
                    abs(pdata.get(surface_key, 1500) - opp_data.get(surface_key, 1500)),
                    abs(pdata['elo'] - opp_data['elo']),
                    (pdata['elo'] + opp_data['elo']) / 2,
                    
                    # Top historical features
                    (p1['avg_total_games'] + p2['avg_total_games']) / 2,
                    p1['total_games_std'] + p2['total_games_std'],
                    
                    # Top serve/return features
                    (p1['bp_saved_pct'] + p2['bp_saved_pct']) / 2,
                    (p1['ace_pct'] + p2['ace_pct']) / 2,
                    (p1['first_won_pct'] + p2['first_won_pct']) / 2,
                    (p1['rpw_pct'] + p2['rpw_pct']) / 2,
                    (p1['hold_rate'] + p2['hold_rate']) / 2,
                    
                    # Top match type features
                    p1['tiebreak_rate'] + p2['tiebreak_rate'],
                    p1['three_set_rate'] + p2['three_set_rate'],
                ]
            
            X.append(features)
            y.append(total_games)
    
    return np.array(X), np.array(y)


def test_feature_sets():
    """Test different feature sets"""
    print("="*70)
    print("FEATURE SET COMPARISON")
    print("="*70)
    
    data = load_data()
    
    feature_sets = ['elo_focused', 'historical', 'match_type', 'serve_return', 'full', 'optimized']
    
    results = []
    for fs in feature_sets:
        X, y = build_features(data, fs)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
        model.fit(X_train_s, y_train)
        
        pred = model.predict(X_test_s)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        
        results.append((fs, X.shape[1], mae, rmse))
        print(f"\n{fs} ({X.shape[1]} features):")
        print(f"  Test MAE:  {mae:.3f}")
        print(f"  Test RMSE: {rmse:.3f}")
    
    print("\n" + "-"*70)
    print("Summary:")
    print(f"{'Feature Set':<20} {'Features':<10} {'MAE':<10} {'RMSE':<10}")
    print("-"*50)
    for fs, n, mae, rmse in sorted(results, key=lambda x: x[2]):
        print(f"{fs:<20} {n:<10} {mae:<10.3f} {rmse:<10.3f}")
    
    return results


def test_hyperparameters():
    """Test hyperparameter combinations"""
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING")
    print("="*70)
    
    data = load_data()
    X, y = build_features(data, 'full')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Test different configurations
    configs = [
        {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1},
        {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05},
        {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05},
        {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.03},
        {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05},
        {'n_estimators': 400, 'max_depth': 4, 'learning_rate': 0.03},
        {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.05},
        {'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.02},
    ]
    
    results = []
    for cfg in configs:
        model = GradientBoostingRegressor(**cfg, random_state=42)
        model.fit(X_train_s, y_train)
        
        pred = model.predict(X_test_s)
        mae = mean_absolute_error(y_test, pred)
        
        cv = cross_val_score(model, X_train_s, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv.mean()
        
        results.append((cfg, mae, cv_mae))
        print(f"\n{cfg}:")
        print(f"  Test MAE: {mae:.3f}, CV MAE: {cv_mae:.3f}")
    
    print("\n" + "-"*70)
    print("Best by Test MAE:")
    best = min(results, key=lambda x: x[1])
    print(f"  {best[0]}: MAE={best[1]:.3f}")
    
    print("\nBest by CV MAE:")
    best_cv = min(results, key=lambda x: x[2])
    print(f"  {best_cv[0]}: CV MAE={best_cv[2]:.3f}")


def test_lookback_windows():
    """Test different lookback windows for player stats"""
    print("\n" + "="*70)
    print("LOOKBACK WINDOW TESTING")
    print("="*70)
    
    data = load_data()
    
    lookbacks = [5, 10, 15, 20, 25, 30]
    results = []
    
    for lb in lookbacks:
        # Rebuild with different lookback
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
        
        X, y = [], []
        
        for player_name, pdata in player_data.items():
            for i, match in enumerate(pdata['matches']):
                if i < 5:
                    continue
                
                parsed = parse_score(match.get('score', ''))
                if not parsed:
                    continue
                
                total_games = parsed['total']
                
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
                
                p1 = calculate_player_stats(hist, lookback=lb)
                p2 = calculate_player_stats(opp_hist, lookback=lb)
                
                if not p1 or not p2:
                    continue
                
                tourn_level = get_tournament_level(match.get('tournament', ''))
                round_level = get_round_level(match.get('round', 'R32'))
                
                features = [
                    abs(pdata['elo'] - opp_data['elo']),
                    abs(pdata.get(surface_key, 1500) - opp_data.get(surface_key, 1500)),
                    (pdata['elo'] + opp_data['elo']) / 2,
                    (p1['avg_total_games'] + p2['avg_total_games']) / 2,
                    p1['avg_total_games'],
                    p2['avg_total_games'],
                    p1['total_games_std'] + p2['total_games_std'],
                    p1['three_set_rate'] + p2['three_set_rate'],
                    p1['tiebreak_rate'] + p2['tiebreak_rate'],
                    abs(p1['win_pct'] - p2['win_pct']),
                    (p1['hold_rate'] + p2['hold_rate']) / 2,
                    (p1['first_won_pct'] + p2['first_won_pct']) / 2,
                    (p1['ace_pct'] + p2['ace_pct']) / 2,
                    (p1['bp_saved_pct'] + p2['bp_saved_pct']) / 2,
                    (p1['rpw_pct'] + p2['rpw_pct']) / 2,
                ]
                
                X.append(features)
                y.append(total_games)
        
        X, y = np.array(X), np.array(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
        model.fit(X_train_s, y_train)
        
        pred = model.predict(X_test_s)
        mae = mean_absolute_error(y_test, pred)
        
        results.append((lb, len(X), mae))
        print(f"\nLookback {lb}: MAE={mae:.3f} (n={len(X)})")
    
    print("\n" + "-"*70)
    print("Summary:")
    print(f"{'Lookback':<12} {'Samples':<10} {'MAE':<10}")
    print("-"*35)
    for lb, n, mae in sorted(results, key=lambda x: x[2]):
        print(f"{lb:<12} {n:<10} {mae:<10.3f}")


def test_model_types():
    """Compare different model types"""
    print("\n" + "="*70)
    print("MODEL TYPE COMPARISON")
    print("="*70)
    
    data = load_data()
    X, y = build_features(data, 'full')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Ridge (alpha=0.5)': Ridge(alpha=0.5),
        'Lasso': Lasso(alpha=0.1),
        'GB (200/5/0.05)': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
        'GB (300/4/0.03)': GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.03, random_state=42),
        'RF (100/6)': RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
        'RF (200/5)': RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42),
    }
    
    results = []
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        
        results.append((name, mae, rmse))
        print(f"\n{name}:")
        print(f"  MAE:  {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
    
    print("\n" + "-"*70)
    print("Ranking by MAE:")
    for i, (name, mae, rmse) in enumerate(sorted(results, key=lambda x: x[1]), 1):
        print(f"  {i}. {name}: MAE={mae:.3f}")


if __name__ == '__main__':
    print("TOTAL GAMES MODEL TUNING")
    print("="*70)
    
    # Test different components
    test_feature_sets()
    test_hyperparameters()
    test_lookback_windows()
    test_model_types()
    
    print("\n" + "="*70)
    print("TUNING COMPLETE")
    print("="*70)
