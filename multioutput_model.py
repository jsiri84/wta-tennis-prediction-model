"""
Multi-Output Tennis Prediction Model

Single model that predicts spread and total simultaneously.
Winner is derived from the spread prediction.
"""

import json
import os
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
import joblib

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(MODEL_DIR, 'all_players_matches.json')
MODEL_PATH = os.path.join(MODEL_DIR, 'multioutput_model.pkl')


def parse_score(score_str):
    """Parse tennis score string into games"""
    if not score_str:
        return None
    
    sets = []
    parts = score_str.replace('/', ' ').split()
    
    for part in parts:
        part = part.strip('[]')
        if '-' in part:
            try:
                games = part.split('-')
                p = int(games[0].split('(')[0])
                o = int(games[1].split('(')[0])
                sets.append((p, o))
            except:
                continue
    
    if not sets:
        return None
    
    p_games = sum(s[0] for s in sets)
    o_games = sum(s[1] for s in sets)
    
    return {
        'p_games': p_games,
        'o_games': o_games,
        'total': p_games + o_games,
        'spread': p_games - o_games,
        'sets': len(sets)
    }


def load_data():
    with open(DATA_PATH, 'r') as f:
        return json.load(f)


def calculate_player_stats(matches, lookback=15):
    """Calculate player statistics from recent matches"""
    if not matches or len(matches) < 3:
        return None
    
    recent = matches[:lookback]
    wins = sum(1 for m in recent if m['result'] == 'W')
    n = len(recent)
    
    spreads = []
    totals = []
    serve_stats = {'first_in': [], 'first_won': [], 'second_won': [], 'bp_saved': []}
    return_stats = {'rpw': [], 'bp_conv': []}
    
    for m in recent:
        parsed = parse_score(m.get('score', ''))
        if parsed:
            spread = parsed['spread'] if m['result'] == 'W' else -parsed['spread']
            spreads.append(spread)
            totals.append(parsed['total'])
        
        serve = m.get('serve', {})
        if serve.get('first_in_pct'):
            serve_stats['first_in'].append(serve['first_in_pct'])
        if serve.get('first_won_pct'):
            serve_stats['first_won'].append(serve['first_won_pct'])
        if serve.get('second_won_pct'):
            serve_stats['second_won'].append(serve['second_won_pct'])
        if serve.get('bp_saved_pct'):
            serve_stats['bp_saved'].append(serve['bp_saved_pct'])
        
        ret = m.get('return', {})
        if ret.get('rpw_pct'):
            return_stats['rpw'].append(ret['rpw_pct'])
        if ret.get('bp_conv_pct'):
            return_stats['bp_conv'].append(ret['bp_conv_pct'])
    
    return {
        'win_pct': wins / n,
        'avg_spread': np.mean(spreads) if spreads else 0,
        'avg_total': np.mean(totals) if totals else 22,
        'spread_std': np.std(spreads) if len(spreads) > 1 else 3,
        'total_std': np.std(totals) if len(totals) > 1 else 4,
        'first_in': np.mean(serve_stats['first_in']) if serve_stats['first_in'] else 60,
        'first_won': np.mean(serve_stats['first_won']) if serve_stats['first_won'] else 65,
        'second_won': np.mean(serve_stats['second_won']) if serve_stats['second_won'] else 50,
        'bp_saved': np.mean(serve_stats['bp_saved']) if serve_stats['bp_saved'] else 60,
        'rpw': np.mean(return_stats['rpw']) if return_stats['rpw'] else 35,
        'bp_conv': np.mean(return_stats['bp_conv']) if return_stats['bp_conv'] else 40,
    }


def calculate_h2h(matches, opponent_name):
    """Calculate head-to-head record"""
    opp_lower = opponent_name.lower()
    opp_last = opponent_name.split()[-1].lower() if opponent_name else ''
    
    h2h_wins = 0
    h2h_losses = 0
    
    for m in matches:
        match_opp = m.get('opponent', '').lower()
        if opp_last in match_opp or opp_lower.replace(' ', '') in match_opp.replace(' ', ''):
            if m['result'] == 'W':
                h2h_wins += 1
            else:
                h2h_losses += 1
    
    total = h2h_wins + h2h_losses
    return {
        'wins': h2h_wins,
        'losses': h2h_losses,
        'total': total,
        'win_pct': h2h_wins / total if total > 0 else 0.5,
        'diff': h2h_wins - h2h_losses
    }


def build_training_data(data):
    """Build training data with unified feature set for multi-output prediction"""
    
    player_data = {}
    for name, info in data['players'].items():
        player_data[name] = {
            'elo': info.get('elo_overall', 1500),
            'elo_hard': info.get('elo_hard', 1500),
            'elo_clay': info.get('elo_clay', 1500),
            'elo_grass': info.get('elo_grass', 1500),
            'matches': sorted(info.get('matches', []), key=lambda x: x.get('date', ''), reverse=True)
        }
    
    # Name lookup for opponent matching
    name_lookup = {}
    for name in player_data:
        name_lookup[name.lower()] = name
        name_lookup[name.lower().replace(' ', '')] = name
        parts = name.split()
        if len(parts) > 1:
            name_lookup[parts[-1].lower()] = name
    
    X = []
    y_spread = []
    y_total = []
    
    print("Building multi-output training data...")
    
    for player_name, pdata in player_data.items():
        for i, match in enumerate(pdata['matches']):
            if i < 5:  # Need history
                continue
            
            parsed = parse_score(match.get('score', ''))
            if not parsed:
                continue
            
            spread = parsed['spread'] if match['result'] == 'W' else -parsed['spread']
            total = parsed['total']
            
            # Find opponent
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
            
            # Get surface
            surface = match.get('surface', 'Hard')
            if surface not in ['Hard', 'Clay', 'Grass']:
                surface = 'Hard'
            
            surface_key = f'elo_{surface.lower()}'
            
            # Calculate stats using matches before this one
            p1_matches = pdata['matches'][i+1:]
            p2_matches = opp_data['matches']
            
            p1_stats = calculate_player_stats(p1_matches)
            p2_stats = calculate_player_stats(p2_matches)
            
            if not p1_stats or not p2_stats:
                continue
            
            # H2H
            h2h = calculate_h2h(p1_matches, opp_full)
            
            # Build feature vector (differences: P1 - P2)
            features = [
                # ELO features
                pdata['elo'] - opp_data['elo'],
                pdata.get(surface_key, 1500) - opp_data.get(surface_key, 1500),
                
                # Form features
                p1_stats['win_pct'] - p2_stats['win_pct'],
                p1_stats['avg_spread'] - p2_stats['avg_spread'],
                
                # Total game tendencies (sum for total prediction)
                p1_stats['avg_total'] + p2_stats['avg_total'],
                (p1_stats['total_std'] + p2_stats['total_std']) / 2,
                
                # Serve features
                p1_stats['first_in'] - p2_stats['first_in'],
                p1_stats['first_won'] - p2_stats['first_won'],
                p1_stats['second_won'] - p2_stats['second_won'],
                p1_stats['bp_saved'] - p2_stats['bp_saved'],
                
                # Return features
                p1_stats['rpw'] - p2_stats['rpw'],
                p1_stats['bp_conv'] - p2_stats['bp_conv'],
                
                # Dominance ratio (serve vs return balance)
                (p1_stats['first_won'] + p1_stats['second_won']) / 2 - p1_stats['rpw'],
                (p2_stats['first_won'] + p2_stats['second_won']) / 2 - p2_stats['rpw'],
                
                # H2H
                h2h['diff'],
                h2h['win_pct'] - 0.5,
                
                # Absolute levels (for total prediction)
                pdata['elo'],
                opp_data['elo'],
                (p1_stats['first_won'] + p2_stats['first_won']) / 2,
                (p1_stats['rpw'] + p2_stats['rpw']) / 2,
            ]
            
            X.append(features)
            y_spread.append(spread)
            y_total.append(total)
    
    return np.array(X), np.array(y_spread), np.array(y_total)


class MultiOutputPredictor:
    """
    Single model predicting spread and total simultaneously.
    Winner derived from spread.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.player_data = None
        self.name_lookup = None
        
        # Feature names for reference
        self.feature_names = [
            'elo_diff', 'surface_elo_diff',
            'win_pct_diff', 'avg_spread_diff',
            'combined_avg_total', 'avg_total_std',
            'first_in_diff', 'first_won_diff', 'second_won_diff', 'bp_saved_diff',
            'rpw_diff', 'bp_conv_diff',
            'p1_dominance', 'p2_dominance',
            'h2h_diff', 'h2h_win_pct_adj',
            'p1_elo', 'p2_elo',
            'avg_first_won', 'avg_rpw',
        ]
    
    def train(self, force_retrain=False):
        """Train multi-output model"""
        
        if not force_retrain and os.path.exists(MODEL_PATH):
            print("Loading multi-output model from cache...")
            saved = joblib.load(MODEL_PATH)
            self.model = saved['model']
            self.scaler = saved['scaler']
            self.player_data = saved['player_data']
            self.name_lookup = saved['name_lookup']
            return
        
        data = load_data()
        X, y_spread, y_total = build_training_data(data)
        
        # Combine targets
        y = np.column_stack([y_spread, y_total])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Multi-output regressor wrapping gradient boosting
        base_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        self.model = MultiOutputRegressor(base_model)
        self.model.fit(X_scaled, y)
        
        # Store player data for predictions
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
        
        # Evaluate
        y_pred = cross_val_predict(self.model, X_scaled, y, cv=5)
        spread_mae = np.mean(np.abs(y_pred[:, 0] - y_spread))
        total_mae = np.mean(np.abs(y_pred[:, 1] - y_total))
        
        print(f"\nMulti-output model trained on {len(X)} matches")
        print(f"  Spread MAE: {spread_mae:.2f} games")
        print(f"  Total MAE:  {total_mae:.2f} games")
        print(f"  Features:   {len(self.feature_names)}")
        
        # Save
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'player_data': self.player_data,
            'name_lookup': self.name_lookup,
        }, MODEL_PATH)
        print(f"  Saved to:   {MODEL_PATH}")
    
    def predict(self, player1: str, player2: str, surface: str = 'Hard'):
        """Generate prediction for a match"""
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Find players
        p1_key = self.name_lookup.get(player1.lower()) or self.name_lookup.get(player1.lower().replace(' ', ''))
        p2_key = self.name_lookup.get(player2.lower()) or self.name_lookup.get(player2.lower().replace(' ', ''))
        
        if not p1_key or p1_key not in self.player_data:
            raise ValueError(f"Player not found: {player1}")
        if not p2_key or p2_key not in self.player_data:
            raise ValueError(f"Player not found: {player2}")
        
        p1_data = self.player_data[p1_key]
        p2_data = self.player_data[p2_key]
        
        surface_key = f'elo_{surface.lower()}'
        
        # Calculate stats
        p1_stats = calculate_player_stats(p1_data['matches'])
        p2_stats = calculate_player_stats(p2_data['matches'])
        
        if not p1_stats or not p2_stats:
            raise ValueError("Insufficient match history")
        
        # H2H
        h2h = calculate_h2h(p1_data['matches'], p2_key)
        
        # Build features
        features = [
            p1_data['elo'] - p2_data['elo'],
            p1_data.get(surface_key, 1500) - p2_data.get(surface_key, 1500),
            p1_stats['win_pct'] - p2_stats['win_pct'],
            p1_stats['avg_spread'] - p2_stats['avg_spread'],
            p1_stats['avg_total'] + p2_stats['avg_total'],
            (p1_stats['total_std'] + p2_stats['total_std']) / 2,
            p1_stats['first_in'] - p2_stats['first_in'],
            p1_stats['first_won'] - p2_stats['first_won'],
            p1_stats['second_won'] - p2_stats['second_won'],
            p1_stats['bp_saved'] - p2_stats['bp_saved'],
            p1_stats['rpw'] - p2_stats['rpw'],
            p1_stats['bp_conv'] - p2_stats['bp_conv'],
            (p1_stats['first_won'] + p1_stats['second_won']) / 2 - p1_stats['rpw'],
            (p2_stats['first_won'] + p2_stats['second_won']) / 2 - p2_stats['rpw'],
            h2h['diff'],
            h2h['win_pct'] - 0.5,
            p1_data['elo'],
            p2_data['elo'],
            (p1_stats['first_won'] + p2_stats['first_won']) / 2,
            (p1_stats['rpw'] + p2_stats['rpw']) / 2,
        ]
        
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Predict both outputs
        pred = self.model.predict(X_scaled)[0]
        spread = pred[0]
        total = pred[1]
        
        # Derive winner probability from spread
        # Using logistic function: prob = 1 / (1 + exp(-k * spread))
        # k calibrated so spread of 4 ≈ 70% win prob
        k = 0.25
        win_prob = 1 / (1 + np.exp(-k * spread))
        
        # Convert to American odds
        if win_prob >= 0.5:
            p1_american = int(-100 * win_prob / (1 - win_prob))
            p2_american = int(100 * (1 - win_prob) / win_prob)
        else:
            p1_american = int(100 * (1 - win_prob) / win_prob)
            p2_american = int(-100 * win_prob / (1 - win_prob))
        
        return {
            'player1': p1_key,
            'player2': p2_key,
            'surface': surface,
            
            # Winner (derived from spread)
            'p1_win_prob': round(win_prob * 100, 1),
            'p2_win_prob': round((1 - win_prob) * 100, 1),
            'p1_odds': p1_american,
            'p2_odds': p2_american,
            
            # Spread
            'spread': round(spread, 1),
            'spread_favors': p1_key if spread > 0 else p2_key,
            
            # Total
            'total': round(total, 1),
            
            # H2H
            'h2h': f"{h2h['wins']}-{h2h['losses']}" if h2h['total'] > 0 else "No H2H",
            
            # ELO
            'p1_elo': p1_data['elo'],
            'p2_elo': p2_data['elo'],
        }
    
    def print_prediction(self, player1: str, player2: str, surface: str = 'Hard'):
        """Print formatted prediction"""
        r = self.predict(player1, player2, surface)
        
        print(f"\n{'='*60}")
        print(f"MULTI-OUTPUT MODEL PREDICTION")
        print(f"{'='*60}")
        print(f"{r['player1']} vs {r['player2']} ({r['surface']})")
        print(f"H2H: {r['h2h']}")
        print(f"\nWinner:  {r['spread_favors']} ({r['p1_odds'] if r['p1_win_prob'] > 50 else r['p2_odds']:+d})")
        print(f"Spread:  {r['spread_favors']} -{abs(r['spread']):.1f}")
        print(f"Total:   {r['total']:.1f} games")
        print(f"\nELO: {r['player1']} {r['p1_elo']} | {r['player2']} {r['p2_elo']}")


if __name__ == '__main__':
    # Train and test
    model = MultiOutputPredictor()
    model.train(force_retrain=True)
    
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    matchups = [
        ('Aryna Sabalenka', 'Iga Swiatek', 'Hard'),
        ('Coco Gauff', 'Jessica Pegula', 'Hard'),
        ('Emma Raducanu', 'Maja Chwalinska', 'Hard'),
    ]
    
    for p1, p2, surface in matchups:
        model.print_prediction(p1, p2, surface)
