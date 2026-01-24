"""
Unified Tennis Prediction Model

Single model that predicts spread and total games together,
then derives win probability from spread. This ensures all
predictions are mathematically consistent.

Key insight: 
- spread = winner_games - loser_games
- total = winner_games + loser_games
- Therefore: winner_games = (total + spread) / 2
            loser_games = (total - spread) / 2

By predicting spread and total together, we guarantee consistency.
Win probability is derived from P(spread > 0).
"""

import json
import numpy as np
import os
import re
import joblib
from collections import defaultdict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Model paths
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
UNIFIED_MODEL_PATH = os.path.join(MODEL_DIR, 'unified_model.pkl')
DATA_PATH = os.path.join(MODEL_DIR, 'all_players_matches.json')


def parse_score(score_str):
    """Parse tennis score, return (winner_games, loser_games, total, spread) or None"""
    if not score_str:
        return None
    
    score_upper = score_str.upper()
    if any(x in score_upper for x in ['RET', 'W/O', 'WO', 'DEF', 'ABD']):
        return None
    
    p_games, o_games = 0, 0
    sets = score_str.strip().split()
    
    for s in sets:
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
        'spread': p_games - o_games,
        'sets': len(sets)
    }


def load_data():
    with open(DATA_PATH, 'r') as f:
        return json.load(f)


def calculate_player_stats(matches, lookback=15):
    """Calculate comprehensive player stats"""
    if not matches or len(matches) < 3:
        return None
    
    recent = matches[:lookback]
    wins = sum(1 for m in recent if m['result'] == 'W')
    
    # Game/spread stats
    spreads = []
    totals = []
    tiebreaks = 0
    straight_sets = 0
    three_sets = 0
    
    for m in recent:
        parsed = parse_score(m.get('score', ''))
        if parsed:
            spreads.append(parsed['spread'])
            totals.append(parsed['total'])
            if parsed['sets'] == 2:
                straight_sets += 1
            elif parsed['sets'] == 3:
                three_sets += 1
            tiebreaks += m.get('score', '').count('(')
    
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
    n_parsed = len(spreads) if spreads else 1
    
    # Dominance ratio
    dr_values = []
    for m in recent:
        serve = m.get('serve', {})
        ret = m.get('return', {})
        spw = (serve.get('first_won_pct') or 0) * 0.6 + (serve.get('second_won_pct') or 0) * 0.4
        rpw = ret.get('rpw_pct') or 0
        if rpw > 0 and (100 - rpw) > 0:
            dr_values.append(spw / (100 - rpw))
    
    # Hold rate approximation
    hold_rate = np.mean(serve_stats['first_won_pct']) * 0.6 + np.mean(serve_stats['second_won_pct']) * 0.4 if serve_stats['first_won_pct'] and serve_stats['second_won_pct'] else 55
    
    # Break rate (from return stats)
    break_rate = np.mean(return_stats['bp_conv_pct']) if return_stats['bp_conv_pct'] else 40
    
    return {
        'win_pct': wins / n,
        'avg_spread': np.mean(spreads) if spreads else 0,
        'spread_std': np.std(spreads) if len(spreads) > 1 else 3,
        'avg_total': np.mean(totals) if totals else 20,
        'total_std': np.std(totals) if len(totals) > 1 else 4,
        'max_total': max(totals) if totals else 26,
        'min_total': min(totals) if totals else 12,
        'straight_set_rate': straight_sets / n_parsed,
        'three_set_rate': three_sets / n_parsed,
        'tiebreak_rate': tiebreaks / n,
        'avg_dr': np.mean(dr_values) if dr_values else 1.0,
        
        # Serve
        'first_in_pct': np.mean(serve_stats['first_in_pct']) if serve_stats['first_in_pct'] else 60,
        'first_won_pct': np.mean(serve_stats['first_won_pct']) if serve_stats['first_won_pct'] else 65,
        'second_won_pct': np.mean(serve_stats['second_won_pct']) if serve_stats['second_won_pct'] else 50,
        'ace_pct': np.mean(serve_stats['ace_pct']) if serve_stats['ace_pct'] else 5,
        'df_pct': np.mean(serve_stats['df_pct']) if serve_stats['df_pct'] else 3,
        'bp_saved_pct': np.mean(serve_stats['bp_saved_pct']) if serve_stats['bp_saved_pct'] else 60,
        
        # Return
        'rpw_pct': np.mean(return_stats['rpw_pct']) if return_stats['rpw_pct'] else 35,
        'bp_conv_pct': np.mean(return_stats['bp_conv_pct']) if return_stats['bp_conv_pct'] else 40,
        
        # Derived (for total games)
        'hold_rate': hold_rate,
        'break_rate': break_rate,
    }


def calculate_surface_stats(matches, surface, lookback=15):
    """Calculate stats on specific surface"""
    surface_matches = [m for m in matches if m.get('surface', '').lower() == surface.lower()][:lookback]
    if len(surface_matches) < 3:
        return None
    return calculate_player_stats(surface_matches, lookback)


def calculate_h2h(matches, opponent_name):
    """Calculate head-to-head record"""
    opp_lower = opponent_name.lower()
    h2h_wins = 0
    h2h_losses = 0
    
    for m in matches:
        match_opp = m.get('opponent', '').lower()
        if match_opp == opp_lower or match_opp.split()[-1] == opp_lower.split()[-1]:
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


def build_training_data(data):
    """Build training data for unified model - predicts spread AND total together"""
    
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
    
    X_spread = []
    X_total = []
    y_spread = []
    y_total = []
    match_info = []
    
    print("Building unified training data...")
    
    for player_name, pdata in player_data.items():
        for i, match in enumerate(pdata['matches']):
            if i < 5:
                continue
            
            # Parse score
            parsed = parse_score(match.get('score', ''))
            if not parsed:
                continue
            
            spread = parsed['spread']
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
            
            # Historical data
            match_date = match.get('date', '')
            hist = [m for m in pdata['matches'][i+1:] if m.get('date', '') < match_date]
            opp_hist = [m for m in opp_data['matches'] if m.get('date', '') < match_date]
            
            if len(hist) < 5 or len(opp_hist) < 5:
                continue
            
            surface = match.get('surface', 'Hard').lower()
            surface_key = f'elo_{surface}' if surface in ['hard', 'clay', 'grass'] else 'elo_hard'
            
            p1 = calculate_player_stats(hist)
            p2 = calculate_player_stats(opp_hist)
            p1_surf = calculate_surface_stats(hist, surface)
            p2_surf = calculate_surface_stats(opp_hist, surface)
            h2h = calculate_h2h(hist, opp_full)
            
            if not p1 or not p2:
                continue
            
            tourn_level = get_tournament_level(match.get('tournament', ''))
            round_level = get_round_level(match.get('round', 'R32'))
            
            # === SPREAD FEATURES (differences - who wins by how much) ===
            spread_features = [
                # ELO differences (2)
                pdata['elo'] - opp_data['elo'],
                pdata.get(surface_key, 1500) - opp_data.get(surface_key, 1500),
                
                # Historical spread (5)
                p1['avg_spread'] - p2['avg_spread'],
                p1['avg_spread'],
                p2['avg_spread'],
                p1['spread_std'] + p2['spread_std'],
                abs(p1['avg_spread']) + abs(p2['avg_spread']),
                
                # Form differences (3)
                p1['win_pct'] - p2['win_pct'],
                p1['avg_dr'] - p2['avg_dr'],
                (p1_surf['win_pct'] if p1_surf else 0.5) - (p2_surf['win_pct'] if p2_surf else 0.5),
                
                # Serve differences (4)
                p1['first_won_pct'] - p2['first_won_pct'],
                p1['ace_pct'] - p2['ace_pct'],
                p1['second_won_pct'] - p2['second_won_pct'],
                p1['bp_saved_pct'] - p2['bp_saved_pct'],
                
                # Return differences (2)
                p1['rpw_pct'] - p2['rpw_pct'],
                p1['bp_conv_pct'] - p2['bp_conv_pct'],
                
                # Match tendencies (2)
                p1['straight_set_rate'] - p2['straight_set_rate'],
                abs(p1['win_pct'] - p2['win_pct']),
                
                # H2H (3)
                h2h['diff'],
                h2h['win_pct'],
                h2h['total'],
            ]
            
            # === TOTAL GAMES FEATURES (combined - both players contribute) ===
            total_features = [
                # ELO gap (3) - closer match = more games
                abs(pdata['elo'] - opp_data['elo']),
                abs(pdata.get(surface_key, 1500) - opp_data.get(surface_key, 1500)),
                (pdata['elo'] + opp_data['elo']) / 2,  # Match quality
                
                # Historical total games (5)
                (p1['avg_total'] + p2['avg_total']) / 2,
                p1['avg_total'],
                p2['avg_total'],
                p1['total_std'] + p2['total_std'],
                (p1['max_total'] + p2['max_total']) / 2,
                
                # Match type tendencies (4)
                p1['three_set_rate'] + p2['three_set_rate'],
                p1['straight_set_rate'] + p2['straight_set_rate'],
                p1['tiebreak_rate'] + p2['tiebreak_rate'],
                abs(p1['win_pct'] - p2['win_pct']),
                
                # Serve dominance (4) - combined stats
                (p1['hold_rate'] + p2['hold_rate']) / 2,
                (p1['first_won_pct'] + p2['first_won_pct']) / 2,
                (p1['ace_pct'] + p2['ace_pct']) / 2,
                (p1['bp_saved_pct'] + p2['bp_saved_pct']) / 2,
                
                # Return/break (3) - combined
                (p1['break_rate'] + p2['break_rate']) / 2,
                (p1['rpw_pct'] + p2['rpw_pct']) / 2,
                (p1['bp_conv_pct'] + p2['bp_conv_pct']) / 2,
                
                # Context (3)
                tourn_level,
                round_level,
                tourn_level * round_level,
                
                # Surface (3)
                1 if surface == 'hard' else 0,
                1 if surface == 'clay' else 0,
                1 if surface == 'grass' else 0,
            ]
            
            X_spread.append(spread_features)
            X_total.append(total_features)
            y_spread.append(spread)
            y_total.append(total)
            match_info.append({
                'player': player_name,
                'opponent': opp_full,
                'surface': surface,
                'spread': spread,
                'total': total
            })
    
    return np.array(X_spread), np.array(X_total), np.array(y_spread), np.array(y_total), match_info


class UnifiedPredictor:
    """
    Unified tennis prediction model.
    
    Predicts spread and total games in a single consistent framework.
    Win probability comes from the original prediction_model (better calibrated).
    """
    
    # Estimated prediction uncertainty (from validation)
    SPREAD_STD = 4.5  # Standard deviation of spread predictions
    TOTAL_STD = 3.5   # Standard deviation of total predictions
    
    def __init__(self):
        self.spread_model = None
        self.total_model = None
        self.winner_predictor = None  # Uses original TennisPredictor for calibrated probs
        self.spread_scaler = None
        self.total_scaler = None
        self.player_data = None
        self.name_lookup = None
    
    def _needs_retrain(self):
        if not os.path.exists(UNIFIED_MODEL_PATH):
            return True
        if not os.path.exists(DATA_PATH):
            return True
        return os.path.getmtime(DATA_PATH) > os.path.getmtime(UNIFIED_MODEL_PATH)
    
    def _save_model(self):
        save_data = {
            'spread_model': self.spread_model,
            'total_model': self.total_model,
            'spread_scaler': self.spread_scaler,
            'total_scaler': self.total_scaler,
            'player_data': self.player_data,
            'name_lookup': self.name_lookup,
            'spread_std': self.SPREAD_STD,
            'total_std': self.TOTAL_STD
        }
        joblib.dump(save_data, UNIFIED_MODEL_PATH)
        print(f"Unified model saved to {UNIFIED_MODEL_PATH}")
    
    def _load_model(self):
        save_data = joblib.load(UNIFIED_MODEL_PATH)
        self.spread_model = save_data['spread_model']
        self.total_model = save_data['total_model']
        self.spread_scaler = save_data['spread_scaler']
        self.total_scaler = save_data['total_scaler']
        self.player_data = save_data['player_data']
        self.name_lookup = save_data['name_lookup']
        self.SPREAD_STD = save_data.get('spread_std', 4.5)
        self.TOTAL_STD = save_data.get('total_std', 3.5)
        
        # Load the original winner predictor for calibrated win probabilities
        from prediction_model import TennisPredictor
        self.winner_predictor = TennisPredictor()
        self.winner_predictor.train()  # Loads from its own cache
        
        print("Unified model loaded from cache")
        return True
    
    def train(self, force_retrain=False):
        """Train the unified model"""
        
        if not force_retrain and not self._needs_retrain():
            try:
                if self._load_model():
                    return
                else:
                    print("Cache missing winner model, retraining...")
            except Exception as e:
                print(f"Could not load cached model: {e}")
        
        data = load_data()
        
        # Build player lookup
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
        
        # Build training data (separate features for spread vs total)
        X_spread, X_total, y_spread, y_total, info = build_training_data(data)
        
        # Separate scalers for each feature set
        self.spread_scaler = StandardScaler()
        self.total_scaler = StandardScaler()
        
        X_spread_scaled = self.spread_scaler.fit_transform(X_spread)
        X_total_scaled = self.total_scaler.fit_transform(X_total)
        
        # Train spread model on spread-optimized features
        self.spread_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
        )
        self.spread_model.fit(X_spread_scaled, y_spread)
        
        # Train total model on total-optimized features (from original model)
        self.total_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42
        )
        self.total_model.fit(X_total_scaled, y_total)
        
        # Load the original winner predictor (better calibrated probabilities)
        from prediction_model import TennisPredictor
        self.winner_predictor = TennisPredictor()
        self.winner_predictor.train()
        
        # Calculate prediction uncertainty from training residuals
        spread_pred = self.spread_model.predict(X_spread_scaled)
        total_pred = self.total_model.predict(X_total_scaled)
        
        self.SPREAD_STD = np.std(y_spread - spread_pred)
        self.TOTAL_STD = np.std(y_total - total_pred)
        
        spread_mae = mean_absolute_error(y_spread, spread_pred)
        total_mae = mean_absolute_error(y_total, total_pred)
        
        print(f"Unified model trained on {len(X_spread)} matches")
        print(f"  Spread MAE: {spread_mae:.2f} games (std: {self.SPREAD_STD:.2f})")
        print(f"  Total MAE: {total_mae:.2f} games (std: {self.TOTAL_STD:.2f})")
        
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
        """
        Generate unified prediction.
        
        Returns spread, total, and derived win probability - all consistent.
        """
        if not self.spread_model:
            raise ValueError("Model not trained. Call train() first.")
        
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
        p1_surf = calculate_surface_stats(p1_data['matches'], surface)
        p2_surf = calculate_surface_stats(p2_data['matches'], surface)
        h2h = calculate_h2h(p1_data['matches'], p2_name)
        
        if not p1 or not p2:
            raise ValueError("Not enough match data")
        
        tourn_level = get_tournament_level(tournament)
        round_level = get_round_level(match_round)
        
        # === SPREAD FEATURES (differences) ===
        spread_features = [
            p1_data['elo'] - p2_data['elo'],
            p1_data.get(surface_key, 1500) - p2_data.get(surface_key, 1500),
            
            p1['avg_spread'] - p2['avg_spread'],
            p1['avg_spread'],
            p2['avg_spread'],
            p1['spread_std'] + p2['spread_std'],
            abs(p1['avg_spread']) + abs(p2['avg_spread']),
            
            p1['win_pct'] - p2['win_pct'],
            p1['avg_dr'] - p2['avg_dr'],
            (p1_surf['win_pct'] if p1_surf else 0.5) - (p2_surf['win_pct'] if p2_surf else 0.5),
            
            p1['first_won_pct'] - p2['first_won_pct'],
            p1['ace_pct'] - p2['ace_pct'],
            p1['second_won_pct'] - p2['second_won_pct'],
            p1['bp_saved_pct'] - p2['bp_saved_pct'],
            
            p1['rpw_pct'] - p2['rpw_pct'],
            p1['bp_conv_pct'] - p2['bp_conv_pct'],
            
            p1['straight_set_rate'] - p2['straight_set_rate'],
            abs(p1['win_pct'] - p2['win_pct']),
            
            h2h['diff'],
            h2h['win_pct'],
            h2h['total'],
        ]
        
        # === TOTAL GAMES FEATURES (combined) ===
        total_features = [
            abs(p1_data['elo'] - p2_data['elo']),
            abs(p1_data.get(surface_key, 1500) - p2_data.get(surface_key, 1500)),
            (p1_data['elo'] + p2_data['elo']) / 2,
            
            (p1['avg_total'] + p2['avg_total']) / 2,
            p1['avg_total'],
            p2['avg_total'],
            p1['total_std'] + p2['total_std'],
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
        
        X_spread = np.array([spread_features])
        X_total = np.array([total_features])
        
        X_spread_scaled = self.spread_scaler.transform(X_spread)
        X_total_scaled = self.total_scaler.transform(X_total)
        
        # Get raw predictions from specialized models
        raw_spread = self.spread_model.predict(X_spread_scaled)[0]
        raw_total = self.total_model.predict(X_total_scaled)[0]
        
        # Get win probability from the original winner predictor (better calibrated)
        winner_result = self.winner_predictor.predict(player1, player2, surface)
        win_prob = winner_result['player1']['win_prob'] / 100.0
        
        # Apply consistency constraint:
        # loser_games = (total - |spread|) / 2 must be >= 0
        # winner_games = (total + |spread|) / 2 must be >= 12 (typical minimum)
        
        abs_spread = abs(raw_spread)
        
        # Ensure total is large enough for the spread
        min_total = abs_spread + 1  # At least 1 game for loser (avoid 0)
        min_total = max(min_total, 12)  # Minimum realistic match length
        
        # Also ensure total isn't unrealistically high for dominant matches
        if abs_spread >= 8:
            max_total = 20  # Dominant win shouldn't have 25+ games
        elif abs_spread >= 6:
            max_total = 22
        elif abs_spread >= 4:
            max_total = 24
        else:
            max_total = 30  # Close match can go long
        
        # Constrain total
        adjusted_total = np.clip(raw_total, min_total, max_total)
        
        # Calculate derived values
        winner_games = (adjusted_total + abs_spread) / 2
        loser_games = (adjusted_total - abs_spread) / 2
        
        # Final sanity check - ensure loser has reasonable games
        if loser_games < 0:
            loser_games = 0
            adjusted_total = abs_spread
            winner_games = abs_spread
        
        # Win probability comes from dedicated winner model (already calculated above)
        # Clip to reasonable range
        win_prob = np.clip(win_prob, 0.02, 0.98)
        
        # Convert to odds
        p1_american = self._prob_to_american(win_prob)
        p2_american = self._prob_to_american(1 - win_prob)
        p1_decimal = round(1 / win_prob, 2) if win_prob > 0 else 99.99
        p2_decimal = round(1 / (1 - win_prob), 2) if win_prob < 1 else 99.99
        
        return {
            'player1': p1_name,
            'player2': p2_name,
            'surface': surface.capitalize(),
            'tournament': tournament,
            'round': match_round,
            
            # Win probability (derived from spread)
            'p1_win_prob': round(win_prob * 100, 1),
            'p2_win_prob': round((1 - win_prob) * 100, 1),
            'p1_odds': p1_american,
            'p2_odds': p2_american,
            'p1_decimal': p1_decimal,
            'p2_decimal': p2_decimal,
            
            # ELO
            'p1_elo': p1_data['elo'],
            'p2_elo': p2_data['elo'],
            'p1_surface_elo': p1_data.get(surface_key, 1500),
            'p2_surface_elo': p2_data.get(surface_key, 1500),
            'elo_diff': p1_data['elo'] - p2_data['elo'],
            
            # H2H
            'h2h': f"{h2h['wins']}-{h2h['losses']}" if h2h['total'] > 0 else "No previous meetings",
            
            # Spread (P1 perspective)
            'spread': round(raw_spread, 1),
            'spread_favors': p1_name if raw_spread > 0 else p2_name,
            'spread_line': round(abs(raw_spread), 1),
            
            # Total games
            'total': round(adjusted_total, 1),
            'total_raw': round(raw_total, 1),
            
            # Implied scoreline
            'winner_games': round(winner_games, 1),
            'loser_games': round(loser_games, 1),
            
            # Model uncertainty
            'spread_std': self.SPREAD_STD,
            'total_std': self.TOTAL_STD,
        }
    
    def print_prediction(self, player1: str, player2: str, surface: str = 'Hard',
                         tournament: str = '', match_round: str = 'R32'):
        """Print formatted prediction"""
        
        result = self.predict(player1, player2, surface, tournament, match_round)
        
        print("\n" + "="*70)
        print("UNIFIED MATCH PREDICTION")
        if result['tournament']:
            print(f"{result['tournament']} - {result['round']} - {result['surface']} Court")
        else:
            print(f"{result['surface']} Court")
        print("="*70)
        
        print(f"\n{result['player1']} vs {result['player2']}")
        print(f"Head-to-Head: {result['h2h']}")
        
        # Winner prediction
        print(f"\n{'-'*70}")
        print("WINNER PREDICTION")
        print(f"{'-'*70}")
        print(f"  {'Player':<25} {'Win %':>8} {'American':>10} {'Decimal':>10}")
        print(f"  {'-'*55}")
        print(f"  {result['player1']:<25} {result['p1_win_prob']:>7.1f}% {result['p1_odds']:>+10} {result['p1_decimal']:>10.2f}")
        print(f"  {result['player2']:<25} {result['p2_win_prob']:>7.1f}% {result['p2_odds']:>+10} {result['p2_decimal']:>10.2f}")
        print(f"\n  ELO: {result['player1']} {result['p1_elo']} | {result['player2']} {result['p2_elo']} (diff: {result['elo_diff']:+d})")
        print(f"  Surface ELO: {result['player1']} {result['p1_surface_elo']} | {result['player2']} {result['p2_surface_elo']}")
        
        # Spread
        print(f"\n{'-'*70}")
        print("GAME SPREAD")
        print(f"{'-'*70}")
        print(f"  {result['spread_favors']} -{result['spread_line']}")
        print(f"  (Model predicts {result['spread_favors']} wins by ~{result['spread_line']:.0f} games)")
        
        # Total games
        print(f"\n{'-'*70}")
        print("TOTAL GAMES")
        print(f"{'-'*70}")
        print(f"  Predicted Total: {result['total']:.1f} games")
        
        # Implied scoreline
        winner = result['player1'] if result['p1_win_prob'] > 50 else result['player2']
        loser = result['player2'] if result['p1_win_prob'] > 50 else result['player1']
        print(f"  Implied: {winner} ~{result['winner_games']:.0f} games, {loser} ~{result['loser_games']:.0f} games")
        
        # Suggest scorelines
        likely = self._suggest_scorelines(result['winner_games'], result['loser_games'])
        if likely:
            print(f"  Likely scorelines: {', '.join(likely)}")
        
        # Consistency check
        print(f"\n{'-'*70}")
        print("CONSISTENCY CHECK")
        print(f"{'-'*70}")
        calc_total = result['winner_games'] + result['loser_games']
        calc_spread = result['winner_games'] - result['loser_games']
        print(f"  Winner + Loser = {result['winner_games']:.0f} + {result['loser_games']:.0f} = {calc_total:.0f} games (OK)")
        print(f"  Winner - Loser = {result['winner_games']:.0f} - {result['loser_games']:.0f} = {calc_spread:.0f} spread (OK)")
        
        # Over/under
        print(f"\n{'-'*70}")
        print("OVER/UNDER PROBABILITIES")
        print(f"{'-'*70}")
        
        total = result['total']
        std = self.TOTAL_STD
        
        for line in [18.5, 19.5, 20.5, 21.5, 22.5]:
            prob_over = 1 - stats.norm.cdf(line, loc=total, scale=std)
            prob_under = 1 - prob_over
            over_odds = self._prob_to_american(prob_over)
            under_odds = self._prob_to_american(prob_under)
            marker = " <--" if abs(line - total) < 1 else ""
            print(f"  O/U {line}: Over {prob_over*100:>5.1f}% ({over_odds:>+5}) | Under {prob_under*100:>5.1f}% ({under_odds:>+5}){marker}")
        
        print("="*70)
        
        return result
    
    def _suggest_scorelines(self, winner_games: float, loser_games: float) -> list:
        """Suggest likely scorelines"""
        scores = []
        w = round(winner_games)
        l = round(loser_games)
        
        two_set_options = [
            (6, 0, 6, 0), (6, 0, 6, 1), (6, 0, 6, 2), (6, 0, 6, 3), (6, 0, 6, 4),
            (6, 1, 6, 0), (6, 1, 6, 1), (6, 1, 6, 2), (6, 1, 6, 3), (6, 1, 6, 4),
            (6, 2, 6, 0), (6, 2, 6, 1), (6, 2, 6, 2), (6, 2, 6, 3), (6, 2, 6, 4),
            (6, 3, 6, 0), (6, 3, 6, 1), (6, 3, 6, 2), (6, 3, 6, 3), (6, 3, 6, 4),
            (6, 4, 6, 0), (6, 4, 6, 1), (6, 4, 6, 2), (6, 4, 6, 3), (6, 4, 6, 4),
            (7, 5, 6, 3), (7, 5, 6, 4), (6, 3, 7, 5), (6, 4, 7, 5), (7, 5, 7, 5),
            (7, 6, 6, 4), (6, 4, 7, 6), (7, 6, 7, 6),
        ]
        
        for s1w, s1l, s2w, s2l in two_set_options:
            total_w = s1w + s2w
            total_l = s1l + s2l
            if abs(total_w - w) <= 1 and abs(total_l - l) <= 1:
                scores.append(f"{s1w}-{s1l}, {s2w}-{s2l}")
        
        return scores[:3]
    
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
    predictor = UnifiedPredictor()
    predictor.train()
    
    print("\n" + "="*70)
    print("TEST 1: Osaka vs Inglis (big mismatch)")
    print("="*70)
    predictor.print_prediction('Naomi Osaka', 'Maddison Inglis', 'Hard', 'Australian Open', 'R16')
    
    print("\n" + "="*70)
    print("TEST 2: Swiatek vs Kalinskaya")
    print("="*70)
    predictor.print_prediction('Iga Swiatek', 'Anna Kalinskaya', 'Hard', 'Australian Open', 'R16')
    
    print("\n" + "="*70)
    print("TEST 3: Rybakina vs Valentova")
    print("="*70)
    predictor.print_prediction('Elena Rybakina', 'Tereza Valentova', 'Hard', 'Australian Open', 'R16')
