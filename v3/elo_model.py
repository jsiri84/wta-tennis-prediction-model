"""
ELO-Based Tennis Prediction Model

Simple prediction model using only ELO and surface ELO ratings.
Clean, interpretable, and proven to be the strongest predictor.
"""

import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(MODEL_DIR, 'player_data.json')
MODEL_PATH = os.path.join(MODEL_DIR, 'elo_model.pkl')


def load_data():
    """Load player data from JSON file."""
    with open(DATA_PATH, 'r') as f:
        return json.load(f)


def expected_win_prob(elo1: float, elo2: float) -> float:
    """Calculate expected win probability from ELO difference."""
    # Standard ELO formula
    return 1 / (1 + 10 ** ((elo2 - elo1) / 400))


def prob_to_american_odds(prob: float) -> int:
    """Convert probability to American odds."""
    if prob >= 0.5:
        return int(-100 * prob / (1 - prob))
    else:
        return int(100 * (1 - prob) / prob)


class EloPredictor:
    """
    Simple ELO-based prediction model.
    
    Uses only two features:
    - ELO difference (overall rating)
    - Surface ELO difference (surface-specific rating)
    
    Historically achieves ~68-69% accuracy on WTA matches.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.player_data = None
        self.name_lookup = None
    
    def _build_name_lookup(self):
        """Build name lookup dictionary for flexible matching."""
        self.name_lookup = {}
        for name in self.player_data:
            # Full name
            self.name_lookup[name.lower()] = name
            # No spaces
            self.name_lookup[name.lower().replace(' ', '')] = name
            # Last name only
            parts = name.split()
            if len(parts) > 1:
                self.name_lookup[parts[-1].lower()] = name
    
    def find_player(self, name: str) -> str:
        """Find player by name with flexible matching."""
        if not self.name_lookup:
            return None
        
        # Try exact match first
        key = name.lower()
        if key in self.name_lookup:
            return self.name_lookup[key]
        
        # Try without spaces
        key = name.lower().replace(' ', '')
        if key in self.name_lookup:
            return self.name_lookup[key]
        
        # Try last name
        parts = name.split()
        if parts:
            key = parts[-1].lower()
            if key in self.name_lookup:
                return self.name_lookup[key]
        
        return None
    
    def train(self, force_retrain: bool = False):
        """Train the ELO prediction model."""
        
        # Load from cache if available
        if not force_retrain and os.path.exists(MODEL_PATH):
            print("Loading model from cache...")
            saved = joblib.load(MODEL_PATH)
            self.model = saved['model']
            self.scaler = saved['scaler']
            self.player_data = saved['player_data']
            self._build_name_lookup()
            return
        
        # Load fresh data
        print("Training ELO model...")
        data = load_data()
        
        self.player_data = {}
        for name, info in data['players'].items():
            self.player_data[name] = {
                'elo': info.get('elo_overall', 1500),
                'elo_hard': info.get('elo_hard', 1500),
                'elo_clay': info.get('elo_clay', 1500),
                'elo_grass': info.get('elo_grass', 1500),
            }
        
        self._build_name_lookup()
        
        # Build training data from match results
        X = []
        y = []
        
        for name, info in data['players'].items():
            p1_data = self.player_data[name]
            
            for match in info.get('matches', []):
                result = match.get('result')
                if result not in ('W', 'L'):
                    continue
                
                # Find opponent
                opp_name = self.find_player(match.get('opponent', ''))
                if not opp_name or opp_name not in self.player_data:
                    continue
                
                p2_data = self.player_data[opp_name]
                
                # Get surface ELO
                surface = match.get('surface', 'Hard').lower()
                surface_key = f'elo_{surface}' if surface in ['hard', 'clay', 'grass'] else 'elo_hard'
                
                # Features: ELO diff, surface ELO diff
                features = [
                    p1_data['elo'] - p2_data['elo'],
                    p1_data.get(surface_key, 1500) - p2_data.get(surface_key, 1500),
                ]
                
                X.append(features)
                y.append(1 if result == 'W' else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train logistic regression
        self.model = LogisticRegression(max_iter=1000, C=1.0)
        self.model.fit(X_scaled, y)
        
        # Evaluate
        scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='accuracy')
        
        print(f"\nTrained on {len(X)} matches")
        print(f"Accuracy: {scores.mean()*100:.1f}% (+/- {scores.std()*100:.1f}%)")
        print(f"\nCoefficients:")
        print(f"  ELO diff:         {self.model.coef_[0][0]:.4f}")
        print(f"  Surface ELO diff: {self.model.coef_[0][1]:.4f}")
        
        # Save model
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'player_data': self.player_data,
        }, MODEL_PATH)
        print(f"\nSaved to {MODEL_PATH}")
    
    def predict(self, player1: str, player2: str, surface: str = 'Hard') -> dict:
        """
        Predict match outcome.
        
        Args:
            player1: First player name
            player2: Second player name
            surface: Court surface (Hard, Clay, Grass)
        
        Returns:
            Dictionary with win probabilities and odds
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Find players
        p1_name = self.find_player(player1)
        p2_name = self.find_player(player2)
        
        if not p1_name:
            raise ValueError(f"Player not found: {player1}")
        if not p2_name:
            raise ValueError(f"Player not found: {player2}")
        
        p1_data = self.player_data[p1_name]
        p2_data = self.player_data[p2_name]
        
        # Get surface ELO
        surface = surface.lower()
        surface_key = f'elo_{surface}' if surface in ['hard', 'clay', 'grass'] else 'elo_hard'
        
        # Build features
        features = np.array([[
            p1_data['elo'] - p2_data['elo'],
            p1_data.get(surface_key, 1500) - p2_data.get(surface_key, 1500),
        ]])
        
        X_scaled = self.scaler.transform(features)
        
        # Predict
        prob = self.model.predict_proba(X_scaled)[0]
        p1_prob = prob[1]
        p2_prob = prob[0]
        
        return {
            'player1': p1_name,
            'player2': p2_name,
            'surface': surface.capitalize(),
            
            # ELO ratings
            'p1_elo': p1_data['elo'],
            'p2_elo': p2_data['elo'],
            'elo_diff': p1_data['elo'] - p2_data['elo'],
            'p1_surface_elo': p1_data.get(surface_key, 1500),
            'p2_surface_elo': p2_data.get(surface_key, 1500),
            
            # Win probabilities
            'p1_win_prob': round(p1_prob * 100, 1),
            'p2_win_prob': round(p2_prob * 100, 1),
            
            # American odds
            'p1_odds': prob_to_american_odds(p1_prob),
            'p2_odds': prob_to_american_odds(p2_prob),
            
            # Decimal odds
            'p1_decimal': round(1 / p1_prob, 2) if p1_prob > 0 else 99.99,
            'p2_decimal': round(1 / p2_prob, 2) if p2_prob > 0 else 99.99,
        }
    
    def print_prediction(self, player1: str, player2: str, surface: str = 'Hard'):
        """Print formatted prediction."""
        r = self.predict(player1, player2, surface)
        
        print(f"\n{'='*50}")
        print(f"{r['player1']} vs {r['player2']}")
        print(f"Surface: {r['surface']}")
        print(f"{'='*50}")
        
        print(f"\nELO Ratings:")
        print(f"  {r['player1']}: {r['p1_elo']} (surface: {r['p1_surface_elo']})")
        print(f"  {r['player2']}: {r['p2_elo']} (surface: {r['p2_surface_elo']})")
        print(f"  Difference: {r['elo_diff']:+d}")
        
        print(f"\nWin Probability:")
        print(f"  {r['player1']}: {r['p1_win_prob']:.1f}% ({r['p1_odds']:+d})")
        print(f"  {r['player2']}: {r['p2_win_prob']:.1f}% ({r['p2_odds']:+d})")
        
        winner = r['player1'] if r['p1_win_prob'] > 50 else r['player2']
        odds = r['p1_odds'] if r['p1_win_prob'] > 50 else r['p2_odds']
        print(f"\n>>> PICK: {winner} ({odds:+d})")


if __name__ == '__main__':
    model = EloPredictor()
    model.train(force_retrain=True)
    
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    
    try:
        model.print_prediction('Aryna Sabalenka', 'Iga Swiatek', 'Hard')
        model.print_prediction('Coco Gauff', 'Jessica Pegula', 'Hard')
        model.print_prediction('Emma Raducanu', 'Karolina Muchova', 'Hard')
    except Exception as e:
        print(f"Error: {e}")
        print("Run 'python fetch_players.py' first to get player data.")
