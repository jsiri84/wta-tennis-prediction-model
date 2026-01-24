"""
Tennis Total Games Prediction Model

Predicts the total number of games in a match (over/under).
"""

import json
import numpy as np
import os
import re
import joblib
from collections import defaultdict
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Model save paths
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
TOTALS_MODEL_PATH = os.path.join(MODEL_DIR, 'totals_model.pkl')
DATA_PATH = os.path.join(MODEL_DIR, 'all_players_matches.json')


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
    
    for m in recent:
        parsed = parse_score(m.get('score', ''))
        if parsed:
            totals.append(parsed['total'])
            tiebreaks += parsed['tiebreaks']
            if parsed['is_straight_sets']:
                straight_sets += 1
            elif parsed['sets'] == 3:
                three_sets += 1
    
    # Serve stats
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
    
    n = len(recent)
    n_parsed = len(totals) if totals else 1
    
    # Calculate service hold approximation (higher = fewer breaks = fewer games potentially)
    hold_rate = np.mean(serve_stats['first_won_pct']) * 0.6 + np.mean(serve_stats['second_won_pct']) * 0.4 if serve_stats['first_won_pct'] else 55
    
    # Calculate break rate (higher = more breaks = potentially more games in 3-setters)
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
        
        # Derived
        'hold_rate': hold_rate,
        'break_rate': break_rate,
    }


def build_training_data(data, feature_set='full'):
    """Build training data for total games prediction"""
    
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
    match_info = []
    
    print("Building training data for total games...")
    
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
            
            # Match context
            tourn_level = get_tournament_level(match.get('tournament', ''))
            round_level = get_round_level(match.get('round', 'R32'))
            
            if feature_set == 'basic':
                features = [
                    # ELO gap (closer = more games expected)
                    abs(pdata['elo'] - opp_data['elo']),
                    abs(pdata.get(surface_key, 1500) - opp_data.get(surface_key, 1500)),
                    
                    # Historical total games
                    p1['avg_total_games'] + p2['avg_total_games'],
                    p1['avg_total_games'],
                    p2['avg_total_games'],
                    
                    # Match length tendencies
                    p1['three_set_rate'] + p2['three_set_rate'],
                    p1['tiebreak_rate'] + p2['tiebreak_rate'],
                ]
            
            elif feature_set == 'full':
                features = [
                    # ELO features (3) - smaller gap = closer match = more games
                    abs(pdata['elo'] - opp_data['elo']),
                    abs(pdata.get(surface_key, 1500) - opp_data.get(surface_key, 1500)),
                    (pdata['elo'] + opp_data['elo']) / 2,  # Match quality
                    
                    # Historical total games (5)
                    (p1['avg_total_games'] + p2['avg_total_games']) / 2,  # Combined average
                    p1['avg_total_games'],
                    p2['avg_total_games'],
                    p1['total_games_std'] + p2['total_games_std'],  # Volatility
                    (p1['max_total'] + p2['max_total']) / 2,  # Upper bound tendency
                    
                    # Match type tendencies (4)
                    p1['three_set_rate'] + p2['three_set_rate'],  # 3-set tendency = more games
                    p1['straight_set_rate'] + p2['straight_set_rate'],  # Straight sets = fewer games
                    p1['tiebreak_rate'] + p2['tiebreak_rate'],  # Tiebreaks = more games
                    abs(p1['win_pct'] - p2['win_pct']),  # Win% gap = blowout potential
                    
                    # Serve dominance (4) - higher = fewer breaks = fewer games
                    (p1['hold_rate'] + p2['hold_rate']) / 2,
                    (p1['first_won_pct'] + p2['first_won_pct']) / 2,
                    (p1['ace_pct'] + p2['ace_pct']) / 2,
                    (p1['bp_saved_pct'] + p2['bp_saved_pct']) / 2,
                    
                    # Return/Break (3) - higher = more breaks = potential for 3 sets
                    (p1['break_rate'] + p2['break_rate']) / 2,
                    (p1['rpw_pct'] + p2['rpw_pct']) / 2,
                    (p1['bp_conv_pct'] + p2['bp_conv_pct']) / 2,
                    
                    # Context (3)
                    tourn_level,
                    round_level,
                    tourn_level * round_level,  # Big match indicator
                    
                    # Surface (3) - encoded
                    1 if surface == 'hard' else 0,
                    1 if surface == 'clay' else 0,
                    1 if surface == 'grass' else 0,
                ]
            
            X.append(features)
            y.append(total_games)
            match_info.append({
                'player': player_name,
                'opponent': opp_full,
                'surface': surface,
                'tournament': match.get('tournament', ''),
                'actual_total': total_games
            })
    
    return np.array(X), np.array(y), match_info


def train_and_evaluate():
    """Train and evaluate total games model"""
    
    data = load_data()
    print(f"Loaded {data['player_count']} players with {data['total_matches']} matches")
    
    X, y, match_info = build_training_data(data, 'full')
    print(f"\nBuilt {len(X)} training samples")
    print(f"Average total games: {y.mean():.1f}")
    print(f"Std dev: {y.std():.1f}")
    print(f"Range: {y.min()} to {y.max()}")
    
    # Distribution analysis
    print("\nTotal Games Distribution:")
    for bucket in [(12, 18), (18, 20), (20, 22), (22, 24), (24, 30)]:
        count = np.sum((y >= bucket[0]) & (y < bucket[1]))
        pct = count / len(y) * 100
        print(f"  {bucket[0]}-{bucket[1]} games: {count} ({pct:.1f}%)")
    
    feature_names = [
        'elo_gap', 'surface_elo_gap', 'match_quality',
        'combined_avg_total', 'p1_avg_total', 'p2_avg_total', 'total_volatility', 'upper_bound',
        'combined_3set_rate', 'combined_straight_set', 'combined_tb_rate', 'win_pct_gap',
        'combined_hold', 'combined_first_won', 'combined_ace', 'combined_bp_saved',
        'combined_break', 'combined_rpw', 'combined_bp_conv',
        'tourn_level', 'round_level', 'big_match',
        'is_hard', 'is_clay', 'is_grass'
    ]
    
    X_train, X_test, y_train, y_test, info_train, info_test = train_test_split(
        X, y, match_info, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    print("\n" + "="*70)
    print("MODEL COMPARISON - TOTAL GAMES PREDICTION")
    print("="*70)
    
    models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, subsample=0.8, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    }
    
    best_model = None
    best_mae = float('inf')
    
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        
        train_pred = model.predict(X_train_s)
        test_pred = model.predict(X_test_s)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)
        
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        
        print(f"\n{name}:")
        print(f"  Train MAE: {train_mae:.2f} games")
        print(f"  Test MAE:  {test_mae:.2f} games")
        print(f"  Test RMSE: {test_rmse:.2f} games")
        print(f"  Test R2:   {test_r2:.3f}")
        print(f"  CV MAE:    {cv_mae:.2f} (+/- {cv_scores.std():.2f})")
        
        if test_mae < best_mae:
            best_mae = test_mae
            best_model = model
    
    # Feature importance
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE (Gradient Boosting)")
    print("="*70)
    
    gb = models['Gradient Boosting']
    importances = gb.feature_importances_
    
    print("\nTop 15 Features for predicting total games:")
    for idx in np.argsort(importances)[::-1][:15]:
        print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Correlation analysis
    print("\n" + "="*70)
    print("FEATURE CORRELATIONS WITH TOTAL GAMES")
    print("="*70)
    
    correlations = []
    for i, name in enumerate(feature_names):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        correlations.append((name, corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    print("\nTop correlations:")
    for name, corr in correlations[:15]:
        print(f"  {name}: {corr:+.3f}")
    
    # Prediction analysis
    print("\n" + "="*70)
    print("PREDICTION ANALYSIS")
    print("="*70)
    
    test_pred = best_model.predict(X_test_s)
    
    # Over/Under accuracy
    typical_line = 21.5
    model_over = test_pred > typical_line
    actual_over = y_test > typical_line
    ou_accuracy = np.mean(model_over == actual_over)
    print(f"\nOver/Under {typical_line} accuracy: {ou_accuracy:.1%}")
    
    # By prediction confidence
    print("\nAccuracy by predicted total:")
    for low, high in [(17, 19), (19, 21), (21, 23), (23, 25), (25, 28)]:
        mask = (test_pred >= low) & (test_pred < high)
        if mask.sum() > 10:
            mae = mean_absolute_error(y_test[mask], test_pred[mask])
            print(f"  Predicted {low}-{high} games: MAE={mae:.2f}, n={mask.sum()}")
    
    # Sample predictions
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS")
    print("="*70)
    
    indices = np.random.choice(len(y_test), min(10, len(y_test)), replace=False)
    print(f"\n{'Match':<35} {'Predicted':<12} {'Actual':<10} {'Error'}")
    print("-"*65)
    for i in indices:
        info = info_test[i]
        match_str = f"{info['player'][:15]} vs {info['opponent'][:15]}"
        print(f"{match_str:<35} {test_pred[i]:>6.1f}      {y_test[i]:>5.0f}       {abs(test_pred[i]-y_test[i]):.1f}")
    
    return best_model, scaler, feature_names


class TotalGamesPredictor:
    """Predicts total games in a tennis match"""
    
    MODEL_MAE = 2.5  # Will be updated after training
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.player_data = None
        self.name_lookup = None
    
    def _needs_retrain(self):
        """Check if model needs retraining (data newer than saved model)"""
        if not os.path.exists(TOTALS_MODEL_PATH):
            return True
        if not os.path.exists(DATA_PATH):
            return True
        
        model_time = os.path.getmtime(TOTALS_MODEL_PATH)
        data_time = os.path.getmtime(DATA_PATH)
        
        return data_time > model_time
    
    def _save_model(self):
        """Save trained model to disk"""
        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'player_data': self.player_data,
            'name_lookup': self.name_lookup,
            'model_mae': self.MODEL_MAE
        }
        joblib.dump(save_data, TOTALS_MODEL_PATH)
        print(f"Totals model saved to {TOTALS_MODEL_PATH}")
    
    def _load_model(self):
        """Load trained model from disk"""
        save_data = joblib.load(TOTALS_MODEL_PATH)
        self.model = save_data['model']
        self.scaler = save_data['scaler']
        self.player_data = save_data['player_data']
        self.name_lookup = save_data['name_lookup']
        self.MODEL_MAE = save_data.get('model_mae', 2.5)
        print("Totals model loaded from cache")
        return True
    
    def train(self, force_retrain=False):
        """Train the model (or load from cache)"""
        
        # Try to load cached model
        if not force_retrain and not self._needs_retrain():
            try:
                self._load_model()
                return
            except Exception as e:
                print(f"Could not load cached model: {e}")
        
        # Train fresh
        data = load_data()
        
        self.player_data = {}
        for name, info in data['players'].items():
            self.player_data[name] = {
                'elo': info.get('elo_overall', 1500),
                'elo_hard': info.get('elo_hard', 1500),
                'elo_clay': info.get('elo_clay', 1500),
                'elo_grass': info.get('elo_grass', 1500),
                'matches': sorted(info.get('matches', []), key=lambda x: x.get('date', ''), reverse=True)
            }
        
        self.name_lookup = {}
        for name in self.player_data:
            self.name_lookup[name.lower()] = name
            self.name_lookup[name.lower().replace(' ', '')] = name
            parts = name.split()
            if len(parts) > 1:
                self.name_lookup[parts[-1].lower()] = name
        
        X, y, _ = build_training_data(data, 'full')
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = GradientBoostingRegressor(
            n_estimators=200, 
            max_depth=8, 
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        self.model.fit(X_scaled, y)
        
        # Calculate MAE on training data for reference
        pred = self.model.predict(X_scaled)
        self.MODEL_MAE = mean_absolute_error(y, pred)
        
        print(f"Total games model trained on {len(X)} matches (MAE: {self.MODEL_MAE:.2f})")
        
        # Save for next time
        self._save_model()
    
    def find_player(self, name):
        name_lower = name.lower()
        if name_lower in self.name_lookup:
            return self.name_lookup[name_lower]
        name_nospace = name_lower.replace(' ', '')
        if name_nospace in self.name_lookup:
            return self.name_lookup[name_nospace]
        for k, v in self.name_lookup.items():
            if name_lower in k:
                return v
        return None
    
    def predict(self, player1: str, player2: str, surface: str = 'Hard',
                tournament: str = '', match_round: str = 'R32'):
        """Predict total games"""
        if not self.model:
            raise ValueError("Model not trained")
        
        p1_name = self.find_player(player1)
        p2_name = self.find_player(player2)
        
        if not p1_name:
            raise ValueError(f"Player not found: {player1}")
        if not p2_name:
            raise ValueError(f"Player not found: {player2}")
        
        p1_data = self.player_data[p1_name]
        p2_data = self.player_data[p2_name]
        
        surface = surface.lower()
        surface_key = f'elo_{surface}' if surface in ['hard', 'clay', 'grass'] else 'elo_hard'
        
        p1 = calculate_player_stats(p1_data['matches'])
        p2 = calculate_player_stats(p2_data['matches'])
        
        if not p1 or not p2:
            raise ValueError("Not enough match data")
        
        tourn_level = get_tournament_level(tournament)
        round_level = get_round_level(match_round)
        
        features = [
            abs(p1_data['elo'] - p2_data['elo']),
            abs(p1_data.get(surface_key, 1500) - p2_data.get(surface_key, 1500)),
            (p1_data['elo'] + p2_data['elo']) / 2,
            
            (p1['avg_total_games'] + p2['avg_total_games']) / 2,
            p1['avg_total_games'],
            p2['avg_total_games'],
            p1['total_games_std'] + p2['total_games_std'],
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
        
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        
        total = self.model.predict(X_scaled)[0]
        
        return {
            'player1': p1_name,
            'player2': p2_name,
            'predicted_total': np.round(total, 1),
            'surface': surface.capitalize(),
            'model_range': f"{total - self.MODEL_MAE:.1f} to {total + self.MODEL_MAE:.1f}"
        }
    
    def print_prediction(self, player1: str, player2: str, surface: str = 'Hard',
                         tournament: str = '', match_round: str = 'R32'):
        """Print formatted prediction"""
        result = self.predict(player1, player2, surface, tournament, match_round)
        
        total = result['predicted_total']
        
        print("\n" + "="*60)
        print(f"TOTAL GAMES PREDICTION - {result['surface']} Court")
        print("="*60)
        print(f"\n{result['player1']} vs {result['player2']}")
        print(f"\nPredicted Total: {total:.1f} games")
        print(f"Model Range (+/-MAE): {result['model_range']} games")
        
        # Over/under odds at common lines
        from scipy import stats
        std = 2.8  # Approximate from data
        
        print(f"\n{'-'*60}")
        print("OVER/UNDER PROBABILITIES:")
        print(f"{'-'*60}")
        
        for line in [19.5, 20.5, 21.5, 22.5, 23.5]:
            prob_over = 1 - stats.norm.cdf(line, loc=total, scale=std)
            prob_under = 1 - prob_over
            
            over_odds = self._prob_to_american(prob_over)
            under_odds = self._prob_to_american(prob_under)
            
            marker = " <--" if abs(line - total) < 0.5 else ""
            print(f"  O/U {line}: Over {prob_over*100:>5.1f}% ({over_odds:>+5}) | Under {prob_under*100:>5.1f}% ({under_odds:>+5}){marker}")
        
        print("="*60)
        
        return result
    
    def _prob_to_american(self, prob):
        if prob <= 0.01:
            return 9999
        if prob >= 0.99:
            return -9999
        if prob >= 0.5:
            return int(-100 * prob / (1 - prob))
        else:
            return int(100 * (1 - prob) / prob)


if __name__ == '__main__':
    # Train and evaluate
    model, scaler, feature_names = train_and_evaluate()
    
    # Example predictions
    print("\n" + "="*70)
    print("EXAMPLE TOTAL GAMES PREDICTIONS")
    print("="*70)
    
    predictor = TotalGamesPredictor()
    predictor.train()
    
    examples = [
        ("Aryna Sabalenka", "Iga Swiatek", "Hard", "Australian Open", "SF"),
        ("Coco Gauff", "Naomi Osaka", "Hard", "", "R32"),
        ("Iga Swiatek", "Elena Rybakina", "Clay", "French Open", "F"),
    ]
    
    for p1, p2, surface, tourn, rnd in examples:
        try:
            predictor.print_prediction(p1, p2, surface, tourn, rnd)
        except Exception as e:
            print(f"Error: {e}")
