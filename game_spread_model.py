"""
Tennis Game Spread Prediction Model

Predicts the game spread (difference in games won) between two players.
Uses the same features as the winner prediction model but tuned for regression.
"""

import json
import numpy as np
import os
import re
import joblib
from collections import defaultdict
from datetime import datetime
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Model save paths
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
SPREAD_MODEL_PATH = os.path.join(MODEL_DIR, 'spread_model.pkl')
DATA_PATH = os.path.join(MODEL_DIR, 'all_players_matches.json')


def parse_score(score_str):
    """
    Parse a tennis score string and return total games for each player.
    Returns (player_games, opponent_games) or None if can't parse.
    
    Examples:
        "6-4 6-3" -> (12, 7)
        "7-6(4) 3-6 6-2" -> (16, 14)
        "6-4 4-6 7-5" -> (17, 15)
    """
    if not score_str:
        return None
    
    # Skip retirements, walkovers
    score_upper = score_str.upper()
    if any(x in score_upper for x in ['RET', 'W/O', 'WO', 'DEF', 'ABD']):
        return None
    
    player_games = 0
    opp_games = 0
    
    # Split into sets
    sets = score_str.strip().split()
    
    for s in sets:
        # Remove tiebreak score in parentheses
        s_clean = re.sub(r'\(\d+\)', '', s)
        
        # Parse "X-Y" format
        match = re.match(r'(\d+)-(\d+)', s_clean)
        if match:
            player_games += int(match.group(1))
            opp_games += int(match.group(2))
    
    if player_games == 0 and opp_games == 0:
        return None
    
    return (player_games, opp_games)


def load_data():
    """Load all player match data"""
    with open('all_players_matches.json', 'r') as f:
        return json.load(f)


def calculate_rolling_stats(matches, lookback=15):
    """Calculate rolling stats for a player's recent matches (enhanced for spread prediction)"""
    if not matches:
        return None
    
    recent = matches[:lookback]
    
    wins = sum(1 for m in recent if m['result'] == 'W')
    win_pct = wins / len(recent) if recent else 0.5
    
    # Dominance Ratio
    dr_values = []
    for m in recent:
        serve = m.get('serve', {})
        ret = m.get('return', {})
        spw = (serve.get('first_won_pct') or 0) * 0.6 + (serve.get('second_won_pct') or 0) * 0.4
        rpw = ret.get('rpw_pct') or 0
        if rpw > 0:
            dr = spw / (100 - rpw) if (100 - rpw) > 0 else 1.0
            dr_values.append(dr)
    
    avg_dr = np.mean(dr_values) if dr_values else 1.0
    
    # Serve stats
    serve_stats = defaultdict(list)
    for m in recent:
        serve = m.get('serve', {})
        for key in ['first_in_pct', 'first_won_pct', 'second_won_pct', 'ace_pct', 'df_pct', 'bp_saved_pct']:
            val = serve.get(key)
            if val is not None:
                serve_stats[key].append(val)
    
    # Return stats
    return_stats = defaultdict(list)
    for m in recent:
        ret = m.get('return', {})
        for key in ['rpw_pct', 'bp_conv_pct']:
            val = ret.get(key)
            if val is not None:
                return_stats[key].append(val)
    
    # Game spread stats - enhanced for spread prediction
    spreads = []
    total_games_list = []
    straight_set_count = 0
    three_set_count = 0
    tiebreak_count = 0
    
    for m in recent:
        parsed = parse_score(m.get('score', ''))
        if parsed:
            p_games, o_games = parsed
            spreads.append(p_games - o_games)
            total_games_list.append(p_games + o_games)
            
            # Count tiebreaks from score
            score = m.get('score', '')
            tiebreak_count += score.count('(')
            
            # Count sets
            sets = score.strip().split()
            if len(sets) == 2:
                straight_set_count += 1
            elif len(sets) == 3:
                three_set_count += 1
    
    n_parsed = len(spreads) if spreads else 1
    
    return {
        'win_pct': win_pct,
        'avg_dr': avg_dr,
        'first_in_pct': np.mean(serve_stats['first_in_pct']) if serve_stats['first_in_pct'] else 60,
        'first_won_pct': np.mean(serve_stats['first_won_pct']) if serve_stats['first_won_pct'] else 65,
        'second_won_pct': np.mean(serve_stats['second_won_pct']) if serve_stats['second_won_pct'] else 50,
        'ace_pct': np.mean(serve_stats['ace_pct']) if serve_stats['ace_pct'] else 5,
        'df_pct': np.mean(serve_stats['df_pct']) if serve_stats['df_pct'] else 3,
        'bp_saved_pct': np.mean(serve_stats['bp_saved_pct']) if serve_stats['bp_saved_pct'] else 60,
        'rpw_pct': np.mean(return_stats['rpw_pct']) if return_stats['rpw_pct'] else 35,
        'bp_conv_pct': np.mean(return_stats['bp_conv_pct']) if return_stats['bp_conv_pct'] else 40,
        # Spread-specific stats
        'avg_spread': np.mean(spreads) if spreads else 0,
        'spread_std': np.std(spreads) if len(spreads) > 1 else 3,
        'avg_total_games': np.mean(total_games_list) if total_games_list else 20,
        'straight_set_rate': straight_set_count / n_parsed,
        'three_set_rate': three_set_count / n_parsed,
        'tiebreak_rate': tiebreak_count / n_parsed,
    }


def calculate_surface_form(matches, surface, lookback=15):
    """Calculate form on a specific surface"""
    surface_matches = [m for m in matches if m.get('surface', '').lower() == surface.lower()][:lookback]
    if len(surface_matches) < 3:
        return None
    return calculate_rolling_stats(surface_matches, lookback)


def build_training_data(data):
    """Build training dataset for game spread prediction"""
    
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
    match_info = []
    
    print("Building training data for game spread...")
    
    for player_name, pdata in player_data.items():
        for i, match in enumerate(pdata['matches']):
            if i < 5:  # Need some history
                continue
            
            # Parse the score to get game spread
            parsed = parse_score(match.get('score', ''))
            if not parsed:
                continue
            
            p_games, o_games = parsed
            game_spread = p_games - o_games  # Positive = player won more games
            
            # Get opponent
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
            
            # Get historical matches before this date
            match_date = match.get('date', '')
            historical_matches = [m for m in pdata['matches'][i+1:] if m.get('date', '') < match_date]
            opp_historical = [m for m in opp_data['matches'] if m.get('date', '') < match_date]
            
            if len(historical_matches) < 5 or len(opp_historical) < 5:
                continue
            
            surface = match.get('surface', 'Hard').lower()
            surface_elo_key = f'elo_{surface}' if surface in ['hard', 'clay', 'grass'] else 'elo_hard'
            
            p1_form = calculate_rolling_stats(historical_matches, 15)
            p2_form = calculate_rolling_stats(opp_historical, 15)
            
            if not p1_form or not p2_form:
                continue
            
            p1_surf = calculate_surface_form(historical_matches, surface, 15)
            p2_surf = calculate_surface_form(opp_historical, surface, 15)
            
            # Build feature vector (optimized 23-feature set)
            features = [
                # ELO (3)
                pdata['elo_overall'] - opp_data['elo_overall'],
                pdata.get(surface_elo_key, 1500) - opp_data.get(surface_elo_key, 1500),
                (pdata['elo_overall'] + opp_data['elo_overall']) / 2,  # Match quality
                
                # Historical spread (5)
                p1_form['avg_spread'] - p2_form['avg_spread'],
                p1_form['avg_spread'],
                p2_form['avg_spread'],
                p1_form['spread_std'] + p2_form['spread_std'],  # Combined volatility
                abs(p1_form['avg_spread']) + abs(p2_form['avg_spread']),  # Combined dominance
                
                # Match tendencies (4)
                p1_form['straight_set_rate'] - p2_form['straight_set_rate'],
                p1_form['three_set_rate'] + p2_form['three_set_rate'],  # Combined 3-set tendency
                p1_form['tiebreak_rate'] + p2_form['tiebreak_rate'],  # Combined tiebreak tendency
                p1_form['avg_total_games'] + p2_form['avg_total_games'],  # Expected match length
                
                # Dominance metrics (3)
                p1_form['avg_dr'] - p2_form['avg_dr'],
                p1_form['win_pct'] - p2_form['win_pct'],
                (p1_surf['win_pct'] if p1_surf else 0.5) - (p2_surf['win_pct'] if p2_surf else 0.5),
                
                # Serve power (4)
                p1_form['first_won_pct'] - p2_form['first_won_pct'],
                p1_form['ace_pct'] - p2_form['ace_pct'],
                p1_form['second_won_pct'] - p2_form['second_won_pct'],
                p1_form['df_pct'] - p2_form['df_pct'],
                
                # Defense (2)
                p1_form['bp_saved_pct'] - p2_form['bp_saved_pct'],
                p1_form['rpw_pct'] - p2_form['rpw_pct'],
                
                # Break points (1)
                p1_form['bp_conv_pct'] - p2_form['bp_conv_pct'],
            ]
            
            X.append(features)
            y.append(game_spread)
            match_info.append({
                'player': player_name,
                'opponent': opp_full_name,
                'date': match_date,
                'surface': surface,
                'score': match.get('score'),
                'actual_spread': game_spread
            })
    
    return np.array(X), np.array(y), match_info


def train_and_evaluate():
    """Train game spread model and evaluate performance"""
    
    data = load_data()
    print(f"Loaded {data['player_count']} players with {data['total_matches']} matches")
    
    X, y, match_info = build_training_data(data)
    print(f"\nBuilt {len(X)} training samples")
    print(f"Average game spread: {y.mean():.2f}")
    print(f"Spread std dev: {y.std():.2f}")
    print(f"Spread range: {y.min()} to {y.max()}")
    
    feature_names = [
        # ELO (3)
        'elo_diff', 'surface_elo_diff', 'match_quality',
        # Spread history (5)
        'avg_spread_diff', 'p1_avg_spread', 'p2_avg_spread', 'combined_volatility', 'combined_dominance',
        # Match tendencies (4)
        'straight_set_diff', 'combined_3set_rate', 'combined_tb_rate', 'expected_length',
        # Dominance (3)
        'dr_diff', 'win_pct_diff', 'surface_form_diff',
        # Serve (4)
        'first_won_diff', 'ace_diff', 'second_won_diff', 'df_diff',
        # Defense (2)
        'bp_saved_diff', 'rpw_diff',
        # Break (1)
        'bp_conv_diff'
    ]
    
    # Split data
    X_train, X_test, y_train, y_test, info_train, info_test = train_test_split(
        X, y, match_info, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON - GAME SPREAD PREDICTION")
    print("="*60)
    
    models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    }
    
    best_model = None
    best_mae = float('inf')
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)
        
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"\n{name}:")
        print(f"  Train MAE: {train_mae:.2f} games")
        print(f"  Test MAE:  {test_mae:.2f} games")
        print(f"  Test RMSE: {test_rmse:.2f} games")
        print(f"  Test R2:   {test_r2:.3f}")
        print(f"  CV MAE:    {cv_mae:.2f} (+/- {cv_std:.2f})")
        
        if test_mae < best_mae:
            best_mae = test_mae
            best_model = model
    
    # Feature importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (Gradient Boosting)")
    print("="*60)
    
    gb_model = models['Gradient Boosting']
    importances = gb_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop Features for predicting game spread:")
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Analyze predictions
    print("\n" + "="*60)
    print("PREDICTION ANALYSIS")
    print("="*60)
    
    test_pred = best_model.predict(X_test_scaled)
    
    # How often do we get the direction right?
    direction_correct = np.sum((test_pred > 0) == (y_test > 0))
    direction_accuracy = direction_correct / len(y_test)
    print(f"\nDirection accuracy (who wins more games): {direction_accuracy:.1%}")
    
    # Prediction buckets
    print("\nPrediction accuracy by predicted spread:")
    for low, high in [(0, 2), (2, 4), (4, 6), (6, 10)]:
        mask = (np.abs(test_pred) >= low) & (np.abs(test_pred) < high)
        if mask.sum() > 0:
            mae = mean_absolute_error(y_test[mask], test_pred[mask])
            print(f"  Predicted spread {low}-{high} games: MAE={mae:.2f}, n={mask.sum()}")
    
    # Sample predictions
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS vs ACTUAL")
    print("="*60)
    
    indices = np.random.choice(len(y_test), min(10, len(y_test)), replace=False)
    print(f"\n{'Player vs Opponent':<35} {'Predicted':<12} {'Actual':<10} {'Error'}")
    print("-"*70)
    for i in indices:
        info = info_test[i]
        match_str = f"{info['player'][:15]} vs {info['opponent'][:15]}"
        pred_spread = test_pred[i]
        actual_spread = y_test[i]
        error = abs(pred_spread - actual_spread)
        print(f"{match_str:<35} {pred_spread:>+6.1f} games  {actual_spread:>+5.0f} games  {error:.1f}")
    
    return best_model, scaler, feature_names


class GameSpreadPredictor:
    """Predicts game spread for tennis matches"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.player_data = None
        self.name_lookup = None
    
    def _needs_retrain(self):
        """Check if model needs retraining (data newer than saved model)"""
        if not os.path.exists(SPREAD_MODEL_PATH):
            return True
        if not os.path.exists(DATA_PATH):
            return True
        
        model_time = os.path.getmtime(SPREAD_MODEL_PATH)
        data_time = os.path.getmtime(DATA_PATH)
        
        return data_time > model_time
    
    def _save_model(self):
        """Save trained model to disk"""
        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'player_data': self.player_data,
            'name_lookup': self.name_lookup
        }
        joblib.dump(save_data, SPREAD_MODEL_PATH)
        print(f"Spread model saved to {SPREAD_MODEL_PATH}")
    
    def _load_model(self):
        """Load trained model from disk"""
        save_data = joblib.load(SPREAD_MODEL_PATH)
        self.model = save_data['model']
        self.scaler = save_data['scaler']
        self.player_data = save_data['player_data']
        self.name_lookup = save_data['name_lookup']
        print("Spread model loaded from cache")
        return True
    
    def train(self, force_retrain=False):
        """Train the game spread prediction model (or load from cache)"""
        
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
                'elo_overall': info.get('elo_overall', 1500),
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
        
        X, y, _ = build_training_data(data)
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
        self.model.fit(X_scaled, y)
        
        print(f"Game spread model trained on {len(X)} matches")
        
        # Save for next time
        self._save_model()
    
    def find_player(self, name):
        """Find player in database"""
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
    
    def predict(self, player1: str, player2: str, surface: str = 'Hard'):
        """Predict game spread between two players"""
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
        surface_elo_key = f'elo_{surface}' if surface in ['hard', 'clay', 'grass'] else 'elo_hard'
        
        p1_form = calculate_rolling_stats(p1_data['matches'], 15)
        p2_form = calculate_rolling_stats(p2_data['matches'], 15)
        p1_surf = calculate_surface_form(p1_data['matches'], surface, 15)
        p2_surf = calculate_surface_form(p2_data['matches'], surface, 15)
        
        if not p1_form or not p2_form:
            raise ValueError("Not enough match data")
        
        # Build feature vector (must match training features exactly)
        features = [
            # ELO (3)
            p1_data['elo_overall'] - p2_data['elo_overall'],
            p1_data.get(surface_elo_key, 1500) - p2_data.get(surface_elo_key, 1500),
            (p1_data['elo_overall'] + p2_data['elo_overall']) / 2,
            
            # Historical spread (5)
            p1_form['avg_spread'] - p2_form['avg_spread'],
            p1_form['avg_spread'],
            p2_form['avg_spread'],
            p1_form['spread_std'] + p2_form['spread_std'],
            abs(p1_form['avg_spread']) + abs(p2_form['avg_spread']),
            
            # Match tendencies (4)
            p1_form['straight_set_rate'] - p2_form['straight_set_rate'],
            p1_form['three_set_rate'] + p2_form['three_set_rate'],
            p1_form['tiebreak_rate'] + p2_form['tiebreak_rate'],
            p1_form['avg_total_games'] + p2_form['avg_total_games'],
            
            # Dominance (3)
            p1_form['avg_dr'] - p2_form['avg_dr'],
            p1_form['win_pct'] - p2_form['win_pct'],
            (p1_surf['win_pct'] if p1_surf else 0.5) - (p2_surf['win_pct'] if p2_surf else 0.5),
            
            # Serve (4)
            p1_form['first_won_pct'] - p2_form['first_won_pct'],
            p1_form['ace_pct'] - p2_form['ace_pct'],
            p1_form['second_won_pct'] - p2_form['second_won_pct'],
            p1_form['df_pct'] - p2_form['df_pct'],
            
            # Defense (2)
            p1_form['bp_saved_pct'] - p2_form['bp_saved_pct'],
            p1_form['rpw_pct'] - p2_form['rpw_pct'],
            
            # Break (1)
            p1_form['bp_conv_pct'] - p2_form['bp_conv_pct'],
        ]
        
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        
        spread = self.model.predict(X_scaled)[0]
        
        return {
            'player1': p1_name,
            'player2': p2_name,
            'predicted_spread': round(spread, 1),
            'spread_favors': p1_name if spread > 0 else p2_name,
            'surface': surface.capitalize()
        }
    
    def print_prediction(self, player1: str, player2: str, surface: str = 'Hard'):
        """Print formatted game spread prediction"""
        result = self.predict(player1, player2, surface)
        
        spread = result['predicted_spread']
        spread_abs = abs(spread)
        
        print("\n" + "="*60)
        print(f"GAME SPREAD PREDICTION - {result['surface']} Court")
        print("="*60)
        print(f"\n{result['player1']} vs {result['player2']}")
        print()
        if spread > 0:
            print(f"  {result['player1']} favored by {spread_abs:.1f} games")
            print(f"  Spread: {result['player1']} -{spread_abs:.1f}")
        else:
            print(f"  {result['player2']} favored by {spread_abs:.1f} games")
            print(f"  Spread: {result['player2']} -{spread_abs:.1f}")
        print("="*60)
        
        return result


if __name__ == '__main__':
    # Train and evaluate
    model, scaler, feature_names = train_and_evaluate()
    
    # Example predictions
    print("\n" + "="*60)
    print("EXAMPLE SPREAD PREDICTIONS")
    print("="*60)
    
    predictor = GameSpreadPredictor()
    predictor.train()
    
    examples = [
        ("Aryna Sabalenka", "Iga Swiatek", "Hard"),
        ("Coco Gauff", "Naomi Osaka", "Hard"),
        ("Iga Swiatek", "Elena Rybakina", "Clay"),
    ]
    
    for p1, p2, surface in examples:
        try:
            predictor.print_prediction(p1, p2, surface)
        except Exception as e:
            print(f"Error: {e}")
