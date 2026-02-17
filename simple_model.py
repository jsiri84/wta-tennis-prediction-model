"""
Simple Tennis Prediction Model v2

Predicts: Winner, Spread, Total Games
Features: ELO + Serve/Return + H2H (when available)

Based on analysis:
- ELO: 42% signal (strongest predictor)
- H2H: 9% signal (meaningful when available)
- Momentum: <1% signal (excluded)
"""

import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
import joblib

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(MODEL_DIR, 'all_players_matches.json')
MODEL_PATH = os.path.join(MODEL_DIR, 'simple_model.pkl')


def load_data():
    with open(DATA_PATH, 'r') as f:
        return json.load(f)


def parse_score(score_str):
    """Parse tennis score string into games"""
    if not score_str:
        return None
    
    try:
        sets = score_str.replace('/', ' ').split()
        p_games, o_games = 0, 0
        for s in sets:
            if '-' in s:
                parts = s.split('-')
                p_games += int(parts[0].split('(')[0])
                o_games += int(parts[1].split('(')[0])
        
        if p_games == 0 and o_games == 0:
            return None
            
        return {
            'p_games': p_games,
            'o_games': o_games,
            'total': p_games + o_games,
            'spread': p_games - o_games,
        }
    except:
        return None


def calculate_player_stats(matches, lookback=15):
    """Calculate serve/return stats from recent matches"""
    if not matches or len(matches) < 3:
        return None
    
    recent = matches[:lookback]
    
    serve_stats = {'first_won': [], 'second_won': [], 'bp_saved': [], 'ace': []}
    return_stats = {'rpw': [], 'bp_conv': []}
    spreads = []
    totals = []
    three_setters = 0
    straight_sets = 0
    tiebreaks = 0
    
    for m in recent:
        # Serve stats
        serve = m.get('serve', {})
        if serve.get('first_won_pct'):
            serve_stats['first_won'].append(serve['first_won_pct'])
        if serve.get('second_won_pct'):
            serve_stats['second_won'].append(serve['second_won_pct'])
        if serve.get('bp_saved_pct'):
            serve_stats['bp_saved'].append(serve['bp_saved_pct'])
        if serve.get('ace_pct'):
            serve_stats['ace'].append(serve['ace_pct'])
        
        # Return stats
        ret = m.get('return', {})
        if ret.get('rpw_pct'):
            return_stats['rpw'].append(ret['rpw_pct'])
        if ret.get('bp_conv_pct'):
            return_stats['bp_conv'].append(ret['bp_conv_pct'])
        
        # Game stats
        score = m.get('score', '')
        parsed = parse_score(score)
        if parsed:
            spread = parsed['spread'] if m['result'] == 'W' else -parsed['spread']
            spreads.append(spread)
            totals.append(parsed['total'])
            
            # Match type analysis
            sets = score.replace('/', ' ').split()
            num_sets = len([s for s in sets if '-' in s])
            if num_sets >= 3:
                three_setters += 1
            elif num_sets == 2:
                straight_sets += 1
            if '(' in score:
                tiebreaks += 1
    
    n_matches = len(recent)
    
    # Calculate dominance ratio
    first_won = np.mean(serve_stats['first_won']) if serve_stats['first_won'] else 65
    second_won = np.mean(serve_stats['second_won']) if serve_stats['second_won'] else 50
    rpw = np.mean(return_stats['rpw']) if return_stats['rpw'] else 35
    serve_avg = (first_won + second_won) / 2
    dr = serve_avg / (100 - rpw) if rpw < 100 else 1.0
    
    return {
        # Serve/return
        'first_won': first_won,
        'second_won': second_won,
        'bp_saved': np.mean(serve_stats['bp_saved']) if serve_stats['bp_saved'] else 60,
        'ace': np.mean(serve_stats['ace']) if serve_stats['ace'] else 5,
        'rpw': rpw,
        'bp_conv': np.mean(return_stats['bp_conv']) if return_stats['bp_conv'] else 40,
        'dr': dr,
        
        # Spread/total stats
        'avg_spread': np.mean(spreads) if spreads else 0,
        'spread_std': np.std(spreads) if len(spreads) > 1 else 3,
        'avg_total': np.mean(totals) if totals else 22,
        'total_std': np.std(totals) if len(totals) > 1 else 3,
        'max_total': max(totals) if totals else 26,
        
        # Match tendencies
        'three_set_rate': three_setters / n_matches if n_matches > 0 else 0.3,
        'straight_set_rate': straight_sets / n_matches if n_matches > 0 else 0.5,
        'tiebreak_rate': tiebreaks / n_matches if n_matches > 0 else 0.2,
    }


def calculate_h2h(matches, opponent_name):
    """Calculate head-to-head record"""
    opp_lower = opponent_name.lower()
    opp_last = opponent_name.split()[-1].lower() if opponent_name else ''
    
    wins, losses = 0, 0
    spreads = []
    
    for m in matches:
        match_opp = m.get('opponent', '').lower()
        if opp_last in match_opp or opp_lower.replace(' ', '') in match_opp.replace(' ', ''):
            if m['result'] == 'W':
                wins += 1
                parsed = parse_score(m.get('score', ''))
                if parsed:
                    spreads.append(parsed['spread'])
            else:
                losses += 1
                parsed = parse_score(m.get('score', ''))
                if parsed:
                    spreads.append(-parsed['spread'])
    
    total = wins + losses
    if total == 0:
        return None
    
    return {
        'wins': wins,
        'losses': losses,
        'total': total,
        'win_rate': wins / total,
        'dominance': (wins - losses) / total,
        'avg_spread': np.mean(spreads) if spreads else 0,
    }


def build_training_data(data):
    """Build training data for all 3 models"""
    
    player_data = {}
    for name, info in data['players'].items():
        player_data[name] = {
            'elo': info.get('elo_overall', 1500),
            'elo_hard': info.get('elo_hard', 1500),
            'elo_clay': info.get('elo_clay', 1500),
            'elo_grass': info.get('elo_grass', 1500),
            'matches': sorted(info.get('matches', []), key=lambda x: x.get('date', ''), reverse=True)
        }
    
    # Name lookup
    name_lookup = {}
    for name in player_data:
        name_lookup[name.lower()] = name
        name_lookup[name.lower().replace(' ', '')] = name
        parts = name.split()
        if len(parts) > 1:
            name_lookup[parts[-1].lower()] = name
    
    X = []
    y_winner = []
    y_spread = []
    y_total = []
    
    print("Building simple model training data...")
    
    for player_name, pdata in player_data.items():
        for i, match in enumerate(pdata['matches']):
            if i < 5:
                continue
            
            result = match.get('result')
            if result not in ('W', 'L'):
                continue
            
            # Parse score
            parsed = parse_score(match.get('score', ''))
            if not parsed:
                continue
            
            spread = parsed['spread'] if result == 'W' else -parsed['spread']
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
            
            # Calculate stats
            p1_matches = pdata['matches'][i+1:]
            p2_matches = opp_data['matches']
            
            p1_stats = calculate_player_stats(p1_matches)
            p2_stats = calculate_player_stats(p2_matches)
            
            if not p1_stats or not p2_stats:
                continue
            
            # H2H
            h2h = calculate_h2h(p1_matches, opp_full)
            h2h_dominance = h2h['dominance'] if h2h else 0
            
            # Build features - separate for winner vs spread/total
            elo_diff = pdata['elo'] - opp_data['elo']
            surf_elo_diff = pdata.get(surface_key, 1500) - opp_data.get(surface_key, 1500)
            
            # Winner features (9) - for classification
            winner_features = [
                elo_diff,
                surf_elo_diff,
                p1_stats['first_won'] - p2_stats['first_won'],
                p1_stats['second_won'] - p2_stats['second_won'],
                p1_stats['bp_saved'] - p2_stats['bp_saved'],
                p1_stats['rpw'] - p2_stats['rpw'],
                p1_stats['bp_conv'] - p2_stats['bp_conv'],
                p1_stats['dr'] - p2_stats['dr'],
                h2h_dominance,
            ]
            
            # Spread features (15) - who wins by how much
            spread_features = [
                elo_diff,
                surf_elo_diff,
                # Historical spread patterns
                p1_stats['avg_spread'] - p2_stats['avg_spread'],
                p1_stats['avg_spread'],
                p2_stats['avg_spread'],
                p1_stats['spread_std'] + p2_stats['spread_std'],
                # Dominance indicators
                p1_stats['dr'] - p2_stats['dr'],
                p1_stats['straight_set_rate'] - p2_stats['straight_set_rate'],
                # Serve power
                p1_stats['ace'] - p2_stats['ace'],
                p1_stats['first_won'] - p2_stats['first_won'],
                # Break vulnerability
                p1_stats['bp_saved'] - p2_stats['bp_saved'],
                p1_stats['bp_conv'] - p2_stats['bp_conv'],
                # H2H
                h2h_dominance,
                # Win rate gap (bigger gap = bigger spread)
                abs(p1_stats['straight_set_rate'] - p2_stats['straight_set_rate']),
                abs(elo_diff),
            ]
            
            # Total features (15) - how many games in match
            total_features = [
                # ELO gap - closer matches have more games
                abs(elo_diff),
                abs(surf_elo_diff),
                (pdata['elo'] + opp_data['elo']) / 2,  # Match quality
                # Historical totals
                (p1_stats['avg_total'] + p2_stats['avg_total']) / 2,
                p1_stats['avg_total'],
                p2_stats['avg_total'],
                p1_stats['total_std'] + p2_stats['total_std'],
                (p1_stats['max_total'] + p2_stats['max_total']) / 2,
                # Match type tendencies
                p1_stats['three_set_rate'] + p2_stats['three_set_rate'],
                p1_stats['tiebreak_rate'] + p2_stats['tiebreak_rate'],
                # Serve dominance (high serve = fewer breaks = fewer games)
                (p1_stats['first_won'] + p2_stats['first_won']) / 2,
                (p1_stats['ace'] + p2_stats['ace']) / 2,
                (p1_stats['bp_saved'] + p2_stats['bp_saved']) / 2,
                # Return strength (high return = more breaks = more games)
                (p1_stats['rpw'] + p2_stats['rpw']) / 2,
                (p1_stats['bp_conv'] + p2_stats['bp_conv']) / 2,
            ]
            
            X.append(winner_features + spread_features + total_features)
            y_winner.append(1 if result == 'W' else 0)
            y_spread.append(spread)
            y_total.append(total)
    
    return (np.array(X), np.array(y_winner), np.array(y_spread), 
            np.array(y_total), player_data, name_lookup)


class SimplePredictor:
    """
    Simple unified predictor with Winner, Spread, and Total.
    Includes H2H when available.
    
    Feature layout:
    - Winner: features[0:9] - ELO + serve/return + H2H
    - Spread: features[9:24] - spread-specific features
    - Total:  features[24:39] - total-specific features
    """
    
    N_WINNER_FEATURES = 9
    N_SPREAD_FEATURES = 15
    N_TOTAL_FEATURES = 15
    
    def __init__(self):
        self.winner_model = None
        self.spread_model = None
        self.total_model = None
        self.winner_scaler = StandardScaler()
        self.spread_scaler = StandardScaler()
        self.total_scaler = StandardScaler()
        self.player_data = None
        self.name_lookup = None
    
    def train(self, force_retrain=False):
        """Train all 3 models with specialized feature sets"""
        
        if not force_retrain and os.path.exists(MODEL_PATH):
            print("Loading simple model from cache...")
            saved = joblib.load(MODEL_PATH)
            self.winner_model = saved['winner_model']
            self.spread_model = saved['spread_model']
            self.total_model = saved['total_model']
            self.winner_scaler = saved['winner_scaler']
            self.spread_scaler = saved['spread_scaler']
            self.total_scaler = saved['total_scaler']
            self.player_data = saved['player_data']
            self.name_lookup = saved['name_lookup']
            return
        
        data = load_data()
        X_all, y_winner, y_spread, y_total, self.player_data, self.name_lookup = build_training_data(data)
        
        # Split features by model
        X_winner = X_all[:, :self.N_WINNER_FEATURES]
        X_spread = X_all[:, self.N_WINNER_FEATURES:self.N_WINNER_FEATURES + self.N_SPREAD_FEATURES]
        X_total = X_all[:, self.N_WINNER_FEATURES + self.N_SPREAD_FEATURES:]
        
        # Scale separately
        X_winner_scaled = self.winner_scaler.fit_transform(X_winner)
        X_spread_scaled = self.spread_scaler.fit_transform(X_spread)
        X_total_scaled = self.total_scaler.fit_transform(X_total)
        
        # Winner model
        self.winner_model = LogisticRegression(max_iter=1000, C=0.5)
        self.winner_model.fit(X_winner_scaled, y_winner)
        
        # Spread model - optimized for spread prediction
        self.spread_model = GradientBoostingRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.08, 
            min_samples_split=10, random_state=42
        )
        self.spread_model.fit(X_spread_scaled, y_spread)
        
        # Total model - optimized for total prediction
        self.total_model = GradientBoostingRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            min_samples_split=10, random_state=42
        )
        self.total_model.fit(X_total_scaled, y_total)
        
        # Evaluate with cross-validation
        winner_scores = cross_val_score(self.winner_model, X_winner_scaled, y_winner, cv=5, scoring='accuracy')
        spread_pred = cross_val_predict(self.spread_model, X_spread_scaled, y_spread, cv=5)
        total_pred = cross_val_predict(self.total_model, X_total_scaled, y_total, cv=5)
        
        spread_mae = np.mean(np.abs(spread_pred - y_spread))
        total_mae = np.mean(np.abs(total_pred - y_total))
        
        # Direction accuracy
        spread_direction = np.mean((spread_pred > 0) == (y_spread > 0))
        
        print(f"\nSimple Model v2 trained on {len(X_all)} matches")
        print(f"  Winner features: {self.N_WINNER_FEATURES}")
        print(f"  Spread features: {self.N_SPREAD_FEATURES}")
        print(f"  Total features:  {self.N_TOTAL_FEATURES}")
        print(f"\n  Winner Accuracy: {winner_scores.mean()*100:.1f}% (+/- {winner_scores.std()*100:.1f}%)")
        print(f"  Spread MAE: {spread_mae:.2f} games")
        print(f"  Spread Direction: {spread_direction*100:.1f}%")
        print(f"  Total MAE: {total_mae:.2f} games")
        
        # Save
        joblib.dump({
            'winner_model': self.winner_model,
            'spread_model': self.spread_model,
            'total_model': self.total_model,
            'winner_scaler': self.winner_scaler,
            'spread_scaler': self.spread_scaler,
            'total_scaler': self.total_scaler,
            'player_data': self.player_data,
            'name_lookup': self.name_lookup,
        }, MODEL_PATH)
        print(f"\n  Saved to: {MODEL_PATH}")
    
    def predict(self, player1: str, player2: str, surface: str = 'Hard', h2h_override: tuple = None):
        """
        Predict match outcome.
        
        Args:
            player1: First player name
            player2: Second player name
            surface: Court surface
            h2h_override: Optional (p1_wins, p2_wins) tuple for manual H2H
        
        Returns dict with winner, spread, total predictions
        """
        
        if self.winner_model is None:
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
        
        # H2H - use override or calculate from data
        if h2h_override:
            p1_wins, p2_wins = h2h_override
            total_h2h = p1_wins + p2_wins
            h2h_dominance = (p1_wins - p2_wins) / total_h2h if total_h2h > 0 else 0
            h2h_str = f"{p1_wins}-{p2_wins}"
        else:
            h2h = calculate_h2h(p1_data['matches'], p2_key)
            if h2h:
                h2h_dominance = h2h['dominance']
                h2h_str = f"{h2h['wins']}-{h2h['losses']}"
            else:
                h2h_dominance = 0
                h2h_str = "No H2H"
        
        # Build features - same structure as training
        elo_diff = p1_data['elo'] - p2_data['elo']
        surf_elo_diff = p1_data.get(surface_key, 1500) - p2_data.get(surface_key, 1500)
        
        # Winner features (9)
        winner_features = np.array([[
            elo_diff,
            surf_elo_diff,
            p1_stats['first_won'] - p2_stats['first_won'],
            p1_stats['second_won'] - p2_stats['second_won'],
            p1_stats['bp_saved'] - p2_stats['bp_saved'],
            p1_stats['rpw'] - p2_stats['rpw'],
            p1_stats['bp_conv'] - p2_stats['bp_conv'],
            p1_stats['dr'] - p2_stats['dr'],
            h2h_dominance,
        ]])
        
        # Spread features (15)
        spread_features = np.array([[
            elo_diff,
            surf_elo_diff,
            p1_stats['avg_spread'] - p2_stats['avg_spread'],
            p1_stats['avg_spread'],
            p2_stats['avg_spread'],
            p1_stats['spread_std'] + p2_stats['spread_std'],
            p1_stats['dr'] - p2_stats['dr'],
            p1_stats['straight_set_rate'] - p2_stats['straight_set_rate'],
            p1_stats['ace'] - p2_stats['ace'],
            p1_stats['first_won'] - p2_stats['first_won'],
            p1_stats['bp_saved'] - p2_stats['bp_saved'],
            p1_stats['bp_conv'] - p2_stats['bp_conv'],
            h2h_dominance,
            abs(p1_stats['straight_set_rate'] - p2_stats['straight_set_rate']),
            abs(elo_diff),
        ]])
        
        # Total features (15)
        total_features = np.array([[
            abs(elo_diff),
            abs(surf_elo_diff),
            (p1_data['elo'] + p2_data['elo']) / 2,
            (p1_stats['avg_total'] + p2_stats['avg_total']) / 2,
            p1_stats['avg_total'],
            p2_stats['avg_total'],
            p1_stats['total_std'] + p2_stats['total_std'],
            (p1_stats['max_total'] + p2_stats['max_total']) / 2,
            p1_stats['three_set_rate'] + p2_stats['three_set_rate'],
            p1_stats['tiebreak_rate'] + p2_stats['tiebreak_rate'],
            (p1_stats['first_won'] + p2_stats['first_won']) / 2,
            (p1_stats['ace'] + p2_stats['ace']) / 2,
            (p1_stats['bp_saved'] + p2_stats['bp_saved']) / 2,
            (p1_stats['rpw'] + p2_stats['rpw']) / 2,
            (p1_stats['bp_conv'] + p2_stats['bp_conv']) / 2,
        ]])
        
        # Scale and predict
        X_winner_scaled = self.winner_scaler.transform(winner_features)
        X_spread_scaled = self.spread_scaler.transform(spread_features)
        X_total_scaled = self.total_scaler.transform(total_features)
        
        # Winner prediction
        prob = self.winner_model.predict_proba(X_winner_scaled)[0]
        p1_prob = prob[1]
        
        # H2H adjustment: strong H2H record overrides model
        if h2h_override:
            p1_wins, p2_wins = h2h_override
            total_h2h = p1_wins + p2_wins
            if total_h2h >= 3:
                h2h_wr = p1_wins / total_h2h
                h2h_weight = min(0.3, total_h2h * 0.1)
                p1_prob = p1_prob * (1 - h2h_weight) + h2h_wr * h2h_weight
        
        # Spread prediction
        spread = self.spread_model.predict(X_spread_scaled)[0]
        
        # Total prediction
        total = self.total_model.predict(X_total_scaled)[0]
        
        # Convert to American odds
        if p1_prob >= 0.5:
            p1_odds = int(-100 * p1_prob / (1 - p1_prob))
            p2_odds = int(100 * (1 - p1_prob) / p1_prob)
        else:
            p1_odds = int(100 * (1 - p1_prob) / p1_prob)
            p2_odds = int(-100 * p1_prob / (1 - p1_prob))
        
        return {
            'player1': p1_key,
            'player2': p2_key,
            'surface': surface,
            
            # Winner
            'p1_win_prob': round(p1_prob * 100, 1),
            'p2_win_prob': round((1 - p1_prob) * 100, 1),
            'p1_odds': p1_odds,
            'p2_odds': p2_odds,
            
            # Spread
            'spread': round(spread, 1),
            'spread_favors': p1_key if spread > 0 else p2_key,
            
            # Total
            'total': round(total, 1),
            
            # H2H
            'h2h': h2h_str,
            
            # ELO
            'p1_elo': p1_data['elo'],
            'p2_elo': p2_data['elo'],
        }
    
    def print_prediction(self, player1: str, player2: str, surface: str = 'Hard', h2h_override: tuple = None):
        """Print formatted prediction"""
        r = self.predict(player1, player2, surface, h2h_override)
        
        winner = r['player1'] if r['p1_win_prob'] > 50 else r['player2']
        odds = r['p1_odds'] if r['p1_win_prob'] > 50 else r['p2_odds']
        
        print(f"\n{r['player1']} vs {r['player2']} ({r['surface']})")
        print(f"  H2H: {r['h2h']}")
        print(f"  Pick: {winner} ({odds:+d})")
        print(f"  Spread: {r['spread_favors']} -{abs(r['spread']):.1f}")
        print(f"  Total: {r['total']:.1f} games")


if __name__ == '__main__':
    model = SimplePredictor()
    model.train(force_retrain=True)
    
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    # Regular predictions
    matchups = [
        ('Aryna Sabalenka', 'Iga Swiatek', 'Hard', None),
        ('Coco Gauff', 'Jessica Pegula', 'Hard', None),
        ('Emma Raducanu', 'Maja Chwalinska', 'Hard', None),
    ]
    
    for p1, p2, surface, h2h in matchups:
        model.print_prediction(p1, p2, surface, h2h)
    
    # With H2H override
    print("\n" + "-"*60)
    print("WITH H2H OVERRIDE:")
    print("-"*60)
    model.print_prediction('Katie Volynets', 'Alycia Parks', 'Hard', h2h_override=(0, 3))
