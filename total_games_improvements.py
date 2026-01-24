"""
Total Games Model Improvement Experiments

Testing additional features and tuning to improve accuracy.
"""

import json
import numpy as np
import re
from collections import defaultdict
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
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
    set_scores = []
    
    for s in sets:
        tiebreaks += s.count('(')
        s_clean = re.sub(r'\(\d+\)', '', s)
        match = re.match(r'(\d+)-(\d+)', s_clean)
        if match:
            pg, og = int(match.group(1)), int(match.group(2))
            p_games += pg
            o_games += og
            set_scores.append((pg, og))
    
    if p_games == 0 and o_games == 0:
        return None
    
    # Count decisive sets (6-0, 6-1, 6-2) vs close sets (6-4, 7-5, 7-6)
    decisive = sum(1 for pg, og in set_scores if abs(pg - og) >= 4)
    close = sum(1 for pg, og in set_scores if abs(pg - og) <= 2)
    
    return {
        'p_games': p_games,
        'o_games': o_games,
        'total': p_games + o_games,
        'sets': len(sets),
        'tiebreaks': tiebreaks,
        'is_straight_sets': len(sets) == 2,
        'decisive_sets': decisive,
        'close_sets': close,
        'avg_set_games': (p_games + o_games) / len(sets) if sets else 0
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


def calculate_extended_stats(matches, lookback=15):
    """Calculate extended player stats"""
    if not matches or len(matches) < 3:
        return None
    
    recent = matches[:lookback]
    wins = sum(1 for m in recent if m['result'] == 'W')
    
    # Game stats
    totals = []
    tiebreaks = 0
    straight_sets = 0
    three_sets = 0
    decisive_sets = 0
    close_sets = 0
    set_games = []
    
    # Match time tracking
    match_times = []
    
    for m in recent:
        parsed = parse_score(m.get('score', ''))
        if parsed:
            totals.append(parsed['total'])
            tiebreaks += parsed['tiebreaks']
            decisive_sets += parsed['decisive_sets']
            close_sets += parsed['close_sets']
            set_games.append(parsed['avg_set_games'])
            
            if parsed['is_straight_sets']:
                straight_sets += 1
            elif parsed['sets'] == 3:
                three_sets += 1
        
        # Try to get match time
        serve = m.get('serve', {})
        if serve.get('time'):
            try:
                time_parts = str(serve['time']).split(':')
                if len(time_parts) >= 2:
                    mins = int(time_parts[0]) * 60 + int(time_parts[1])
                    match_times.append(mins)
            except:
                pass
    
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
        'median_total': np.median(totals) if totals else 20,
        'straight_set_rate': straight_sets / n_parsed,
        'three_set_rate': three_sets / n_parsed,
        'tiebreak_rate': tiebreaks / n,
        'decisive_set_rate': decisive_sets / (n_parsed * 2),
        'close_set_rate': close_sets / (n_parsed * 2),
        'avg_set_games': np.mean(set_games) if set_games else 10,
        'avg_match_time': np.mean(match_times) if match_times else 80,
        
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
        
        # Service game efficiency (how often they hold to love/15)
        'serve_dominance': (np.mean(serve_stats['first_won_pct']) if serve_stats['first_won_pct'] else 65) + 
                          (np.mean(serve_stats['ace_pct']) if serve_stats['ace_pct'] else 5) -
                          (np.mean(serve_stats['df_pct']) if serve_stats['df_pct'] else 3),
    }


def get_h2h_total_games(p1_matches, p2_name, match_date):
    """Get average total games in head-to-head matches"""
    h2h_totals = []
    
    for m in p1_matches:
        if m.get('date', '') >= match_date:
            continue
        
        opp = m.get('opponent', '').lower()
        if p2_name.lower() in opp or opp in p2_name.lower():
            parsed = parse_score(m.get('score', ''))
            if parsed:
                h2h_totals.append(parsed['total'])
    
    if h2h_totals:
        return {
            'avg': np.mean(h2h_totals),
            'std': np.std(h2h_totals) if len(h2h_totals) > 1 else 3,
            'count': len(h2h_totals)
        }
    return None


def build_extended_features(data):
    """Build features with all possible additions"""
    
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
    feature_names = []
    
    print("Building extended features...")
    
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
            
            p1 = calculate_extended_stats(hist)
            p2 = calculate_extended_stats(opp_hist)
            
            if not p1 or not p2:
                continue
            
            tourn_level = get_tournament_level(match.get('tournament', ''))
            round_level = get_round_level(match.get('round', 'R32'))
            
            # Head-to-head
            h2h = get_h2h_total_games(hist, opp_full, match_date)
            h2h_avg = h2h['avg'] if h2h else (p1['avg_total_games'] + p2['avg_total_games']) / 2
            h2h_count = h2h['count'] if h2h else 0
            
            features = [
                # ELO features (5)
                abs(pdata['elo'] - opp_data['elo']),  # elo_gap
                abs(pdata.get(surface_key, 1500) - opp_data.get(surface_key, 1500)),  # surface_elo_gap
                (pdata['elo'] + opp_data['elo']) / 2,  # match_quality
                min(pdata['elo'], opp_data['elo']),  # weaker_player_elo
                pdata['elo'] - opp_data['elo'],  # elo_diff_signed
                
                # Historical total games (8)
                (p1['avg_total_games'] + p2['avg_total_games']) / 2,  # combined_avg_total
                p1['avg_total_games'],  # p1_avg_total
                p2['avg_total_games'],  # p2_avg_total
                p1['total_games_std'] + p2['total_games_std'],  # total_volatility
                (p1['max_total'] + p2['max_total']) / 2,  # upper_bound
                (p1['min_total'] + p2['min_total']) / 2,  # lower_bound
                abs(p1['avg_total_games'] - p2['avg_total_games']),  # avg_total_diff
                (p1['median_total'] + p2['median_total']) / 2,  # combined_median_total (NEW)
                
                # Match type tendencies (6)
                p1['three_set_rate'] + p2['three_set_rate'],  # combined_3set_rate
                p1['straight_set_rate'] + p2['straight_set_rate'],  # combined_straight_set
                p1['tiebreak_rate'] + p2['tiebreak_rate'],  # combined_tb_rate
                p1['close_set_rate'] + p2['close_set_rate'],  # combined_close_set (NEW)
                p1['decisive_set_rate'] + p2['decisive_set_rate'],  # combined_decisive_set (NEW)
                abs(p1['win_pct'] - p2['win_pct']),  # win_pct_gap
                
                # Serve/Return (11)
                (p1['hold_rate'] + p2['hold_rate']) / 2,  # combined_hold
                (p1['first_won_pct'] + p2['first_won_pct']) / 2,  # combined_first_won
                (p1['second_won_pct'] + p2['second_won_pct']) / 2,  # combined_second_won (NEW)
                (p1['ace_pct'] + p2['ace_pct']) / 2,  # combined_ace
                (p1['df_pct'] + p2['df_pct']) / 2,  # combined_df (NEW)
                (p1['bp_saved_pct'] + p2['bp_saved_pct']) / 2,  # combined_bp_saved
                (p1['break_rate'] + p2['break_rate']) / 2,  # combined_break
                (p1['rpw_pct'] + p2['rpw_pct']) / 2,  # combined_rpw
                (p1['bp_conv_pct'] + p2['bp_conv_pct']) / 2,  # combined_bp_conv
                (p1['dr'] + p2['dr']) / 2,  # combined_dr
                abs(p1['dr'] - p2['dr']),  # dr_gap
                
                # Serve dominance (NEW - 2)
                (p1['serve_dominance'] + p2['serve_dominance']) / 2,  # combined_serve_dom
                abs(p1['serve_dominance'] - p2['serve_dominance']),  # serve_dom_gap
                
                # Set-level stats (NEW - 2)
                (p1['avg_set_games'] + p2['avg_set_games']) / 2,  # combined_avg_set_games
                abs(p1['avg_set_games'] - p2['avg_set_games']),  # avg_set_games_gap
                
                # Match time (NEW - 2)
                (p1['avg_match_time'] + p2['avg_match_time']) / 2,  # combined_match_time
                abs(p1['avg_match_time'] - p2['avg_match_time']),  # match_time_gap
                
                # Head-to-head (NEW - 2)
                h2h_avg,  # h2h_avg_total
                h2h_count,  # h2h_count
                
                # Context (4)
                tourn_level,  # tourn_level
                round_level,  # round_level
                tourn_level * round_level,  # big_match
                1 if round_level >= 5 else 0,  # is_late_round (NEW)
                
                # Surface (3)
                1 if surface == 'hard' else 0,  # is_hard
                1 if surface == 'clay' else 0,  # is_clay
                1 if surface == 'grass' else 0,  # is_grass
            ]
            
            X.append(features)
            y.append(total_games)
    
    feature_names = [
        'elo_gap', 'surface_elo_gap', 'match_quality', 'weaker_player_elo', 'elo_diff_signed',
        'combined_avg_total', 'p1_avg_total', 'p2_avg_total', 'total_volatility', 'upper_bound',
        'lower_bound', 'avg_total_diff', 'combined_median_total',
        'combined_3set_rate', 'combined_straight_set', 'combined_tb_rate', 'combined_close_set',
        'combined_decisive_set', 'win_pct_gap',
        'combined_hold', 'combined_first_won', 'combined_second_won', 'combined_ace', 'combined_df',
        'combined_bp_saved', 'combined_break', 'combined_rpw', 'combined_bp_conv', 'combined_dr', 'dr_gap',
        'combined_serve_dom', 'serve_dom_gap',
        'combined_avg_set_games', 'avg_set_games_gap',
        'combined_match_time', 'match_time_gap',
        'h2h_avg_total', 'h2h_count',
        'tourn_level', 'round_level', 'big_match', 'is_late_round',
        'is_hard', 'is_clay', 'is_grass'
    ]
    
    return np.array(X), np.array(y), feature_names


def test_new_features():
    """Test impact of new features"""
    print("="*70)
    print("NEW FEATURE ANALYSIS")
    print("="*70)
    
    data = load_data()
    X, y, feature_names = build_extended_features(data)
    
    print(f"\nBuilt {len(X)} samples with {len(feature_names)} features")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Baseline with current best model
    print("\n--- Baseline (current model) ---")
    baseline_features = [0, 1, 2, 5, 6, 7, 8, 9, 13, 14, 15, 18, 19, 20, 22, 24, 25, 26, 27, 28, 38, 39, 40, 42, 43, 44]
    
    model = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(X_train_s[:, baseline_features], y_train)
    pred = model.predict(X_test_s[:, baseline_features])
    baseline_mae = mean_absolute_error(y_test, pred)
    print(f"Baseline MAE: {baseline_mae:.3f}")
    
    # Test with all features
    print("\n--- All Extended Features ---")
    model_full = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    model_full.fit(X_train_s, y_train)
    pred_full = model_full.predict(X_test_s)
    full_mae = mean_absolute_error(y_test, pred_full)
    print(f"Full Features MAE: {full_mae:.3f} ({full_mae - baseline_mae:+.3f})")
    
    # Feature importance for all features
    print("\n--- Feature Importance (All Features) ---")
    importances = model_full.feature_importances_
    for idx in np.argsort(importances)[::-1][:20]:
        print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Test new features individually
    print("\n--- Individual New Feature Impact ---")
    new_feature_indices = {
        'combined_median_total': 12,
        'combined_close_set': 16,
        'combined_decisive_set': 17,
        'combined_second_won': 21,
        'combined_df': 23,
        'combined_serve_dom': 30,
        'serve_dom_gap': 31,
        'combined_avg_set_games': 32,
        'avg_set_games_gap': 33,
        'combined_match_time': 34,
        'match_time_gap': 35,
        'h2h_avg_total': 36,
        'h2h_count': 37,
        'is_late_round': 41,
    }
    
    for name, idx in new_feature_indices.items():
        test_features = baseline_features + [idx]
        model_test = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
        model_test.fit(X_train_s[:, test_features], y_train)
        pred_test = model_test.predict(X_test_s[:, test_features])
        test_mae = mean_absolute_error(y_test, pred_test)
        diff = test_mae - baseline_mae
        indicator = "BETTER" if diff < -0.01 else ("WORSE" if diff > 0.01 else "SAME")
        print(f"  + {name}: MAE={test_mae:.3f} ({diff:+.3f}) [{indicator}]")
    
    return X, y, feature_names


def test_hyperparameter_grid():
    """Extensive hyperparameter search"""
    print("\n" + "="*70)
    print("HYPERPARAMETER GRID SEARCH")
    print("="*70)
    
    data = load_data()
    X, y, _ = build_extended_features(data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    param_grid = {
        'n_estimators': [150, 200, 250, 300],
        'max_depth': [5, 6, 7, 8],
        'learning_rate': [0.03, 0.05, 0.07],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    
    print("\nRunning grid search (this may take a while)...")
    
    gb = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train_s, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV MAE: {-grid_search.best_score_:.3f}")
    
    # Test on held-out set
    best_model = grid_search.best_estimator_
    pred = best_model.predict(X_test_s)
    test_mae = mean_absolute_error(y_test, pred)
    print(f"Test MAE with best params: {test_mae:.3f}")
    
    return grid_search.best_params_


def test_ensemble_methods():
    """Test ensemble approaches"""
    print("\n" + "="*70)
    print("ENSEMBLE METHODS")
    print("="*70)
    
    data = load_data()
    X, y, _ = build_extended_features(data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Individual models
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    ridge = Ridge(alpha=1.0)
    
    # Voting ensemble
    print("\n--- Voting Regressor ---")
    voting = VotingRegressor([('gb', gb), ('rf', rf), ('ridge', ridge)])
    voting.fit(X_train_s, y_train)
    pred_voting = voting.predict(X_test_s)
    voting_mae = mean_absolute_error(y_test, pred_voting)
    print(f"Voting MAE: {voting_mae:.3f}")
    
    # Stacking ensemble
    print("\n--- Stacking Regressor ---")
    stacking = StackingRegressor(
        estimators=[('gb', gb), ('rf', rf)],
        final_estimator=Ridge(alpha=1.0)
    )
    stacking.fit(X_train_s, y_train)
    pred_stacking = stacking.predict(X_test_s)
    stacking_mae = mean_absolute_error(y_test, pred_stacking)
    print(f"Stacking MAE: {stacking_mae:.3f}")
    
    # Simple average of top 2 models
    print("\n--- Simple Average (GB + RF) ---")
    gb.fit(X_train_s, y_train)
    rf.fit(X_train_s, y_train)
    pred_avg = (gb.predict(X_test_s) + rf.predict(X_test_s)) / 2
    avg_mae = mean_absolute_error(y_test, pred_avg)
    print(f"Average MAE: {avg_mae:.3f}")


def test_alternative_models():
    """Test alternative regression models"""
    print("\n" + "="*70)
    print("ALTERNATIVE MODELS")
    print("="*70)
    
    data = load_data()
    X, y, _ = build_extended_features(data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    models = {
        'Gradient Boosting (current)': GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42),
        'GB (deeper)': GradientBoostingRegressor(n_estimators=300, max_depth=8, learning_rate=0.03, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
        'RF (deeper)': RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Ridge': Ridge(alpha=1.0),
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
    
    print("\n--- Ranking ---")
    for i, (name, mae, rmse) in enumerate(sorted(results, key=lambda x: x[1]), 1):
        print(f"  {i}. {name}: MAE={mae:.3f}")


def correlation_analysis():
    """Deep correlation analysis"""
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)
    
    data = load_data()
    X, y, feature_names = build_extended_features(data)
    
    print("\nFeature correlations with total games:")
    correlations = []
    for i, name in enumerate(feature_names):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        correlations.append((name, corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for name, corr in correlations[:25]:
        print(f"  {name}: {corr:+.3f}")
    
    print("\n--- Key Insights ---")
    print("Positive correlation = more games expected")
    print("Negative correlation = fewer games expected")


if __name__ == '__main__':
    print("TOTAL GAMES MODEL - IMPROVEMENT EXPERIMENTS")
    print("="*70)
    
    # Run experiments
    X, y, feature_names = test_new_features()
    correlation_analysis()
    test_alternative_models()
    test_ensemble_methods()
    
    print("\n--- Running Grid Search (may take 2-3 minutes) ---")
    best_params = test_hyperparameter_grid()
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE")
    print("="*70)
