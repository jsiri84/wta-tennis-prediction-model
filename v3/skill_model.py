"""
ServeSkill / ReturnSkill Tennis Prediction Model

Based on the framework:
- ServeSkill: How good is this player at holding serve?
- ReturnSkill: How good is this player at breaking serve?
- Monte Carlo simulation for match winner and total games

Surface-specific skills calculated for Hard, Clay, Grass.
"""

import json
import os
import numpy as np
import random
from collections import defaultdict
import joblib
from datetime import datetime

# Import fatigue module (v2 - advanced)
try:
    from fatigue_v2 import calculate_advanced_fatigue
    FATIGUE_V2_ENABLED = True
except ImportError:
    FATIGUE_V2_ENABLED = False

# Legacy fatigue (fallback)
try:
    from fatigue import calculate_travel_fatigue_only
    FATIGUE_ENABLED = True
except ImportError:
    FATIGUE_ENABLED = False

# Import court speed module
try:
    from court_speed import get_speed_index, get_speed_adjustment, get_speed_category
    COURT_SPEED_ENABLED = True
except ImportError:
    COURT_SPEED_ENABLED = False

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(MODEL_DIR, 'player_data.json')
SKILLS_PATH = os.path.join(MODEL_DIR, 'player_skills.pkl')
HAND_COEFFS_PATH = os.path.join(MODEL_DIR, 'hand_coefficients.json')

# Load handedness coefficients
# Disabled by default - testing showed it hurts overall accuracy (overfitting)
# Set to True to enable hand-based adjustments
HANDEDNESS_ENABLED = False
HAND_COEFFICIENTS = {}
if HANDEDNESS_ENABLED:
    try:
        with open(HAND_COEFFS_PATH, 'r') as f:
            HAND_COEFFICIENTS = json.load(f)
    except:
        HANDEDNESS_ENABLED = False

# Load clutch/overperformance coefficients
CLUTCH_PATH = os.path.join(MODEL_DIR, 'clutch_coefficients.json')
CLUTCH_ENABLED = True
CLUTCH_COEFFICIENTS = {}
try:
    with open(CLUTCH_PATH, 'r') as f:
        CLUTCH_COEFFICIENTS = json.load(f)
except:
    CLUTCH_ENABLED = False

# Head-to-Head adjustment parameters
H2H_ENABLED = True
H2H_MIN_MEETINGS = 2  # Minimum prior meetings to apply H2H
H2H_WEIGHT = 0.10     # Blend weight for H2H (10%)
H2H_DATABASE = None   # Will be populated on first use

# Tunable parameters
K_DF = 0.5              # Double fault penalty strength (reduced)
C_OPP = 0.5             # Opponent adjustment strength (reduced for stability)
EWMA_HALF_LIFE = 8      # Matches for half-life decay

# Tour average baselines (universal performed better than surface-specific in testing)
SPW_AVG = 0.58          # Tour average service points won
RPW_AVG = 0.45          # Tour average return points won


def load_data():
    """Load player data."""
    with open(DATA_PATH, 'r') as f:
        return json.load(f)


def logit(p):
    """Logit transform: ln(p / (1-p))"""
    p = np.clip(p, 0.01, 0.99)  # Avoid infinity
    return np.log(p / (1 - p))


def sigmoid(x):
    """Inverse logit."""
    return 1 / (1 + np.exp(-x))


def ewma(values, half_life=EWMA_HALF_LIFE):
    """Exponentially weighted moving average (most recent first)."""
    if not values:
        return 0.0
    
    alpha = 1 - np.exp(-np.log(2) / half_life)
    result = values[0]
    for i in range(1, len(values)):
        result = alpha * values[i] + (1 - alpha) * result
    return result


def calculate_spw(match):
    """
    Calculate Service Points Won percentage.
    SPW = FSIn × 1SPW + (1-FSIn) × 2SPW
    """
    serve = match.get('serve', {})
    
    fs_in = serve.get('first_in_pct') or 60
    fs_won = serve.get('first_won_pct') or 70
    ss_won = serve.get('second_won_pct') or 50
    
    fs_in = fs_in / 100
    fs_won = fs_won / 100
    ss_won = ss_won / 100
    
    spw = fs_in * fs_won + (1 - fs_in) * ss_won
    return spw


def calculate_spw_star(match):
    """
    Calculate SPW with double fault penalty.
    SPW* = clip(SPW - k × DFp, 0.35, 0.75)
    """
    spw = calculate_spw(match)
    serve = match.get('serve', {})
    df_pct = serve.get('df_pct') or 3
    df_pct = df_pct / 100
    
    spw_star = spw - K_DF * df_pct
    spw_star = np.clip(spw_star, 0.35, 0.75)
    return spw_star


def calculate_rpw(match):
    """Calculate Return Points Won percentage."""
    ret = match.get('return', {})
    rpw = ret.get('rpw_pct')
    if rpw is None:
        rpw = 40  # Default
    rpw = rpw / 100
    return np.clip(rpw, 0.20, 0.60)


def build_h2h_database():
    """
    Build H2H database from player match data.
    Deduplicates matches and stores with dates for temporal filtering.
    
    Returns: dict of {(player1, player2): [(date, p1_won), ...]}
    """
    global H2H_DATABASE
    
    if H2H_DATABASE is not None:
        return H2H_DATABASE
    
    data = load_data()
    players = data['players']
    
    # Use a set to deduplicate
    seen_matches = set()
    h2h_matches = defaultdict(list)
    
    for name, player_info in players.items():
        for match in player_info.get('matches', []):
            opp = match.get('opponent', '')
            result = match.get('result', '')
            date_str = match.get('date', '')
            
            if not opp or not date_str or result not in ('W', 'L'):
                continue
            
            # Parse date
            try:
                if len(date_str) == 8:
                    match_date = datetime.strptime(date_str, '%Y%m%d')
                else:
                    match_date = datetime.strptime(date_str[:10], '%Y-%m-%d')
            except:
                continue
            
            # Create canonical key for deduplication
            winner = name if result == 'W' else opp
            match_key = (match_date.strftime('%Y%m%d'), tuple(sorted([name, opp])), winner)
            
            if match_key in seen_matches:
                continue
            seen_matches.add(match_key)
            
            # Store in h2h_matches
            key = tuple(sorted([name, opp]))
            p1_won = (winner == key[0])
            h2h_matches[key].append((match_date, p1_won))
    
    H2H_DATABASE = dict(h2h_matches)
    return H2H_DATABASE


def get_h2h_record(player1: str, player2: str, before_date: str = None):
    """
    Get H2H record between two players.
    
    Args:
        player1: First player name
        player2: Second player name
        before_date: Optional date string (YYYYMMDD) - only count matches before this
    
    Returns:
        tuple: (player1_wins, player2_wins, total_meetings)
    """
    h2h_db = build_h2h_database()
    
    key = tuple(sorted([player1, player2]))
    matches = h2h_db.get(key, [])
    
    if not matches:
        return 0, 0, 0
    
    # Filter by date if specified
    if before_date:
        try:
            cutoff = datetime.strptime(str(before_date)[:8], '%Y%m%d')
            matches = [(d, won) for d, won in matches if d < cutoff]
        except:
            pass
    
    # Count wins
    p1_wins = sum(1 for d, won in matches if won)
    p2_wins = sum(1 for d, won in matches if not won)
    
    # Return in correct order
    if player1 == key[0]:
        return p1_wins, p2_wins, p1_wins + p2_wins
    else:
        return p2_wins, p1_wins, p1_wins + p2_wins


def get_h2h_probability(player1: str, player2: str, before_date: str = None):
    """
    Calculate win probability for player1 based on H2H record.
    
    Returns:
        tuple: (h2h_prob, total_meetings)
               h2h_prob is None if insufficient meetings
    """
    p1_wins, p2_wins, total = get_h2h_record(player1, player2, before_date)
    
    if total < H2H_MIN_MEETINGS:
        return None, total
    
    # Add small prior to avoid extreme probabilities
    # Laplace smoothing: (wins + 0.5) / (total + 1)
    h2h_prob = (p1_wins + 0.5) / (total + 1)
    
    return h2h_prob, total


class SkillCalculator:
    """Calculate ServeSkill and ReturnSkill for all players."""
    
    def __init__(self):
        self.player_skills = {}
        self.name_lookup = {}
    
    def _build_name_lookup(self, data):
        """Build flexible name lookup."""
        self.name_lookup = {}
        for name in data['players']:
            self.name_lookup[name.lower()] = name
            self.name_lookup[name.lower().replace(' ', '')] = name
            parts = name.split()
            if len(parts) > 1:
                self.name_lookup[parts[-1].lower()] = name
    
    def find_player(self, name):
        """Find player by name."""
        if not name:
            return None
        key = name.lower()
        if key in self.name_lookup:
            return self.name_lookup[key]
        key = name.lower().replace(' ', '')
        if key in self.name_lookup:
            return self.name_lookup[key]
        parts = name.split()
        if parts:
            key = parts[-1].lower()
            if key in self.name_lookup:
                return self.name_lookup[key]
        return None
    
    def calculate_raw_return_skills(self, data):
        """
        Step 1: Calculate raw ReturnSkill for each player (no opponent adjustment).
        Surface-specific.
        """
        raw_return = defaultdict(lambda: {'hard': [], 'clay': [], 'grass': []})
        
        for player_name, player_info in data['players'].items():
            for match in player_info.get('matches', []):
                surface = match.get('surface', 'Hard').lower()
                if surface not in ['hard', 'clay', 'grass']:
                    surface = 'hard'
                
                rpw = calculate_rpw(match)
                if rpw > 0:
                    # ReturnSkill = logit(RPW) - logit(RPW_avg)
                    return_perf = logit(rpw) - logit(RPW_AVG)
                    raw_return[player_name][surface].append(return_perf)
        
        # EWMA for each surface
        return_skills = {}
        for player, surfaces in raw_return.items():
            return_skills[player] = {}
            for surface in ['hard', 'clay', 'grass']:
                if surfaces[surface]:
                    return_skills[player][surface] = ewma(surfaces[surface])
                else:
                    return_skills[player][surface] = 0.0
        
        return return_skills
    
    def calculate_opponent_adjusted_serve_skills(self, data, return_skills):
        """
        Step 2: Calculate opponent-adjusted ServeSkill.
        AdjServePerf = logit(SPW*) - c × ReturnSkill_opponent
        """
        serve_perfs = defaultdict(lambda: {'hard': [], 'clay': [], 'grass': []})
        
        for player_name, player_info in data['players'].items():
            for match in player_info.get('matches', []):
                surface = match.get('surface', 'Hard').lower()
                if surface not in ['hard', 'clay', 'grass']:
                    surface = 'hard'
                
                # Get opponent
                opp_name = self.find_player(match.get('opponent', ''))
                if not opp_name or opp_name not in return_skills:
                    # No opponent data - use raw serve perf
                    opp_return_skill = 0.0
                else:
                    opp_return_skill = return_skills[opp_name].get(surface, 0.0)
                
                # Calculate adjusted serve performance
                spw_star = calculate_spw_star(match)
                serve_perf = logit(spw_star) - logit(SPW_AVG)
                
                # Adjust for opponent return skill
                adj_serve_perf = serve_perf - C_OPP * opp_return_skill
                serve_perfs[player_name][surface].append(adj_serve_perf)
        
        # EWMA for each surface
        serve_skills = {}
        for player, surfaces in serve_perfs.items():
            serve_skills[player] = {}
            for surface in ['hard', 'clay', 'grass']:
                if surfaces[surface]:
                    serve_skills[player][surface] = ewma(surfaces[surface])
                else:
                    serve_skills[player][surface] = 0.0
        
        return serve_skills
    
    def _normalize_skills(self):
        """Normalize skills to be centered at 0 for each surface."""
        for surface in ['hard', 'clay', 'grass']:
            serve_vals = [p['serve'].get(surface, 0) for p in self.player_skills.values()]
            return_vals = [p['return'].get(surface, 0) for p in self.player_skills.values()]
            
            serve_mean = np.mean(serve_vals) if serve_vals else 0
            return_mean = np.mean(return_vals) if return_vals else 0
            
            for player in self.player_skills.values():
                if surface in player['serve']:
                    player['serve'][surface] -= serve_mean
                if surface in player['return']:
                    player['return'][surface] -= return_mean
    
    def calculate_all_skills(self, force_recalc=False):
        """
        Calculate ServeSkill and ReturnSkill for all players.
        
        Returns dict: {player: {
            'serve': {'hard': X, 'clay': Y, 'grass': Z},
            'return': {'hard': X, 'clay': Y, 'grass': Z},
            'elo': int
        }}
        """
        # Check cache
        if not force_recalc and os.path.exists(SKILLS_PATH):
            print("Loading skills from cache...")
            saved = joblib.load(SKILLS_PATH)
            self.player_skills = saved['skills']
            self.name_lookup = saved['name_lookup']
            return self.player_skills
        
        print("Calculating player skills...")
        data = load_data()
        self._build_name_lookup(data)
        
        # Step 1: Raw return skills
        print("  Step 1: Calculating raw ReturnSkill...")
        return_skills = self.calculate_raw_return_skills(data)
        
        # Step 2: Opponent-adjusted serve skills
        print("  Step 2: Calculating opponent-adjusted ServeSkill...")
        serve_skills = self.calculate_opponent_adjusted_serve_skills(data, return_skills)
        
        # Combine into player_skills
        self.player_skills = {}
        for player_name, player_info in data['players'].items():
            self.player_skills[player_name] = {
                'serve': serve_skills.get(player_name, {'hard': 0, 'clay': 0, 'grass': 0}),
                'return': return_skills.get(player_name, {'hard': 0, 'clay': 0, 'grass': 0}),
                'elo': player_info.get('elo_overall', 1500),
                'elo_hard': player_info.get('elo_hard', 1500),
                'elo_clay': player_info.get('elo_clay', 1500),
                'elo_grass': player_info.get('elo_grass', 1500),
            }
        
        # Normalize skills to be centered at 0 for each surface
        self._normalize_skills()
        
        # Step 3: Calculate court speed preferences
        if COURT_SPEED_ENABLED:
            print("  Step 3: Calculating court speed preferences...")
            self._calculate_speed_preferences(data)
        
        # Save cache
        joblib.dump({
            'skills': self.player_skills,
            'name_lookup': self.name_lookup,
        }, SKILLS_PATH)
        print(f"  Saved skills to {SKILLS_PATH}")
        
        return self.player_skills
    
    def _calculate_speed_preferences(self, data):
        """
        Calculate each player's court speed preference coefficient.
        
        Positive = performs better on fast courts
        Negative = performs better on slow courts
        """
        for player_name, player_info in data['players'].items():
            if player_name not in self.player_skills:
                continue
            
            # Collect win rates by speed category
            results_by_speed = {'fast': [], 'medium': [], 'slow': []}
            
            for match in player_info.get('matches', []):
                tournament = match.get('tournament', '')
                surface = match.get('surface', 'Hard')
                result = match.get('result', '')
                
                if result not in ('W', 'L'):
                    continue
                
                category = get_speed_category(tournament, surface)
                won = 1 if result == 'W' else 0
                results_by_speed[category].append(won)
            
            # Calculate speed preference coefficient
            # Need at least 5 matches on fast and slow to be meaningful
            fast_n = len(results_by_speed['fast'])
            slow_n = len(results_by_speed['slow'])
            
            if fast_n >= 5 and slow_n >= 5:
                fast_wr = np.mean(results_by_speed['fast'])
                slow_wr = np.mean(results_by_speed['slow'])
                # Speed preference: positive = fast court player
                speed_pref = (fast_wr - slow_wr) / 2  # Scale to ~±0.2 range
            else:
                speed_pref = 0.0
            
            self.player_skills[player_name]['speed_pref'] = speed_pref
            self.player_skills[player_name]['fast_matches'] = fast_n
            self.player_skills[player_name]['slow_matches'] = slow_n


def simulate_set(hold_prob_a, hold_prob_b):
    """
    Simulate one set.
    Returns: (games_a, games_b, winner)
    """
    games_a, games_b = 0, 0
    serving = 'A'  # A serves first
    
    while True:
        if serving == 'A':
            if random.random() < hold_prob_a:
                games_a += 1
            else:
                games_b += 1
            serving = 'B'
        else:
            if random.random() < hold_prob_b:
                games_b += 1
            else:
                games_a += 1
            serving = 'A'
        
        # Check for set win
        if games_a >= 6 and games_a - games_b >= 2:
            return games_a, games_b, 'A'
        if games_b >= 6 and games_b - games_a >= 2:
            return games_a, games_b, 'B'
        
        # Tiebreak at 6-6
        if games_a == 6 and games_b == 6:
            # Simplified tiebreak - probability based on hold rates
            tb_prob_a = (hold_prob_a + (1 - hold_prob_b)) / 2
            if random.random() < tb_prob_a:
                return 7, 6, 'A'
            else:
                return 6, 7, 'B'


def simulate_match(hold_prob_a, hold_prob_b, best_of=3, n_sims=5000):
    """
    Monte Carlo simulation of a tennis match.
    
    Returns:
        p1_win_prob: Probability player 1 wins
        avg_total_games: Expected total games
        std_total_games: Std deviation of total games
    """
    wins_a = 0
    total_games_list = []
    
    sets_to_win = 2 if best_of == 3 else 3
    
    for _ in range(n_sims):
        sets_a, sets_b = 0, 0
        match_games = 0
        
        while sets_a < sets_to_win and sets_b < sets_to_win:
            ga, gb, winner = simulate_set(hold_prob_a, hold_prob_b)
            match_games += ga + gb
            if winner == 'A':
                sets_a += 1
            else:
                sets_b += 1
        
        if sets_a == sets_to_win:
            wins_a += 1
        total_games_list.append(match_games)
    
    return {
        'p1_win_prob': wins_a / n_sims,
        'avg_total_games': np.mean(total_games_list),
        'std_total_games': np.std(total_games_list),
        'min_total': np.min(total_games_list),
        'max_total': np.max(total_games_list),
    }


class SkillPredictor:
    """
    Main prediction class using ServeSkill and ReturnSkill.
    Optionally blends with ELO for improved accuracy.
    """
    
    def __init__(self, elo_weight: float = 0.5):
        """
        Args:
            elo_weight: Weight for ELO-based prediction (0-1).
                        0 = pure skill model, 1 = pure ELO
        """
        self.skill_calc = SkillCalculator()
        self.player_skills = None
        self.elo_weight = elo_weight
    
    def train(self, force_recalc=False):
        """Calculate all player skills."""
        self.player_skills = self.skill_calc.calculate_all_skills(force_recalc)
        print(f"\nCalculated skills for {len(self.player_skills)} players")
    
    def elo_win_prob(self, elo1: int, elo2: int) -> float:
        """Calculate win probability from ELO difference."""
        return 1 / (1 + 10 ** ((elo2 - elo1) / 400))
    
    def get_hold_probability(self, serve_skill, return_skill_opp):
        """
        Calculate probability of holding serve.
        P(hold) = sigmoid(baseline + ServeSkill - ReturnSkill_opp)
        
        Baseline chosen so average skills give ~62% hold rate (WTA average)
        sigmoid(0.5) = 0.622
        """
        baseline = 0.5  # Gives ~62% hold for average players
        x = baseline + serve_skill - return_skill_opp
        return sigmoid(x)
    
    def predict(self, player1: str, player2: str, surface: str = 'Hard', 
                tournament: str = None, match_date: str = None, match_round: str = None,
                n_sims: int = 5000) -> dict:
        """
        Predict match outcome.
        
        Args:
            player1: First player name
            player2: Second player name
            surface: Court surface (Hard/Clay/Grass)
            tournament: Tournament name (for fatigue calculation)
            match_date: Match date YYYYMMDD (for fatigue calculation, defaults to today)
            n_sims: Number of Monte Carlo simulations
        
        Returns:
            - Winner prediction with probability
            - Total games prediction
            - Hold probabilities
            - Fatigue info (if available)
        """
        if self.player_skills is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Default match_date to today
        if match_date is None:
            match_date = datetime.now().strftime('%Y%m%d')
        
        # Find players
        p1_name = self.skill_calc.find_player(player1)
        p2_name = self.skill_calc.find_player(player2)
        
        if not p1_name or p1_name not in self.player_skills:
            raise ValueError(f"Player not found: {player1}")
        if not p2_name or p2_name not in self.player_skills:
            raise ValueError(f"Player not found: {player2}")
        
        p1 = self.player_skills[p1_name]
        p2 = self.player_skills[p2_name]
        
        surface = surface.lower()
        if surface not in ['hard', 'clay', 'grass']:
            surface = 'hard'
        
        # Get base skills for this surface
        p1_serve = p1['serve'].get(surface, 0)
        p1_return = p1['return'].get(surface, 0)
        p2_serve = p2['serve'].get(surface, 0)
        p2_return = p2['return'].get(surface, 0)
        
        # Store original skills for output
        p1_serve_orig = p1_serve
        p2_serve_orig = p2_serve
        
        # Calculate fatigue and adjust SERVE SKILLS (v2 method)
        p1_fatigue = {'serve_adjustment': 0, 'fatigue': 0, 'details': {}}
        p2_fatigue = {'serve_adjustment': 0, 'fatigue': 0, 'details': {}}
        
        if FATIGUE_V2_ENABLED and tournament:
            try:
                data = load_data()
                if p1_name in data['players']:
                    p1_fatigue = calculate_advanced_fatigue(
                        data['players'][p1_name].get('matches', []),
                        match_date,
                        tournament,
                        match_round
                    )
                    # Apply fatigue to serve skill (not win probability)
                    p1_serve += p1_fatigue['serve_adjustment']
                
                if p2_name in data['players']:
                    p2_fatigue = calculate_advanced_fatigue(
                        data['players'][p2_name].get('matches', []),
                        match_date,
                        tournament,
                        match_round
                    )
                    p2_serve += p2_fatigue['serve_adjustment']
            except Exception as e:
                pass
        
        # Apply court speed adjustment
        p1_speed_adj = 0.0
        p2_speed_adj = 0.0
        court_speed_info = {}
        
        if COURT_SPEED_ENABLED and tournament:
            court_speed_idx = get_speed_index(tournament, surface)
            court_speed_cat = get_speed_category(tournament, surface)
            court_adj = get_speed_adjustment(tournament, surface)
            
            # Get player speed preferences
            p1_speed_pref = p1.get('speed_pref', 0.0)
            p2_speed_pref = p2.get('speed_pref', 0.0)
            
            # On fast courts, fast-court players get boost; on slow courts, slow-court players get boost
            # court_adj is positive for fast courts, negative for slow
            # speed_pref is positive for fast-court players, negative for slow-court players
            # When both align (fast court + fast player OR slow court + slow player), give boost
            p1_speed_adj = court_adj * p1_speed_pref * 0.5  # Scale factor
            p2_speed_adj = court_adj * p2_speed_pref * 0.5
            
            # Apply to serve skill
            p1_serve += p1_speed_adj
            p2_serve += p2_speed_adj
            
            court_speed_info = {
                'speed_index': round(court_speed_idx, 1),
                'category': court_speed_cat,
                'court_adj': round(court_adj, 3),
                'p1_speed_pref': round(p1_speed_pref, 3),
                'p2_speed_pref': round(p2_speed_pref, 3),
                'p1_speed_adj': round(p1_speed_adj, 4),
                'p2_speed_adj': round(p2_speed_adj, 4),
            }
        
        # Calculate hold probabilities with fatigue and speed-adjusted serve skills
        hold_prob_1 = self.get_hold_probability(p1_serve, p2_return)
        hold_prob_2 = self.get_hold_probability(p2_serve, p1_return)
        
        # Run Monte Carlo simulation
        sim_result = simulate_match(hold_prob_1, hold_prob_2, best_of=3, n_sims=n_sims)
        
        skill_win_prob = sim_result['p1_win_prob']
        
        # Get surface-specific ELO
        elo_key = f'elo_{surface}'
        p1_elo = p1.get(elo_key, p1['elo'])
        p2_elo = p2.get(elo_key, p2['elo'])
        elo_win_prob = self.elo_win_prob(p1_elo, p2_elo)
        
        # Blend skill and ELO predictions
        p1_win_prob = (1 - self.elo_weight) * skill_win_prob + self.elo_weight * elo_win_prob
        p1_win_prob = max(0.05, min(0.95, p1_win_prob))
        
        # Apply handedness adjustment
        p1_hand_adj = 0.0
        p2_hand_adj = 0.0
        hand_info = {}
        
        if HANDEDNESS_ENABLED:
            # Get player hands
            data = load_data()
            p1_hand = data['players'].get(p1_name, {}).get('hand', 'U')
            p2_hand = data['players'].get(p2_name, {}).get('hand', 'U')
            
            # Apply player-specific coefficients based on opponent hand
            # Coefficient represents how much worse player does vs lefties
            # Positive coeff = struggles vs lefties, negative = excels vs lefties
            if p2_hand == 'L':  # P1 facing a lefty
                p1_hand_adj = HAND_COEFFICIENTS.get(p1_name, 0.0)
            if p1_hand == 'L':  # P2 facing a lefty
                p2_hand_adj = HAND_COEFFICIENTS.get(p2_name, 0.0)
            
            # Adjust win probability
            # p1_hand_adj positive = p1 struggles vs lefty = reduce p1 win prob
            # p2_hand_adj positive = p2 struggles vs lefty = increase p1 win prob
            net_hand_adj = p2_hand_adj - p1_hand_adj
            p1_win_prob = p1_win_prob + net_hand_adj
            p1_win_prob = max(0.05, min(0.95, p1_win_prob))
            
            hand_info = {
                'p1_hand': p1_hand,
                'p2_hand': p2_hand,
                'p1_hand_adj': round(p1_hand_adj, 4),
                'p2_hand_adj': round(p2_hand_adj, 4),
                'net_hand_adj': round(net_hand_adj, 4),
            }
        
        # Apply clutch/overperformance adjustment
        # Overperformers get negative adjustment (their ELO may be inflated)
        # Underperformers get positive adjustment (their ELO may be deflated)
        p1_clutch_adj = 0.0
        p2_clutch_adj = 0.0
        clutch_info = {}
        
        if CLUTCH_ENABLED:
            p1_coeff = CLUTCH_COEFFICIENTS.get(p1_name, {})
            p2_coeff = CLUTCH_COEFFICIENTS.get(p2_name, {})
            
            p1_clutch_adj = p1_coeff.get('elo_adjustment', 0.0)
            p2_clutch_adj = p2_coeff.get('elo_adjustment', 0.0)
            
            # Net adjustment: p1's adjustment minus p2's adjustment
            # If p1 is overperformer (negative adj), reduce their win prob
            # If p2 is overperformer (negative adj), increase p1's win prob
            net_clutch_adj = p1_clutch_adj - p2_clutch_adj
            p1_win_prob = p1_win_prob + net_clutch_adj
            p1_win_prob = max(0.05, min(0.95, p1_win_prob))
            
            clutch_info = {
                'p1_overperf': round(p1_coeff.get('overperformance', 0) * 100, 1),
                'p2_overperf': round(p2_coeff.get('overperformance', 0) * 100, 1),
                'p1_clutch_adj': round(p1_clutch_adj, 4),
                'p2_clutch_adj': round(p2_clutch_adj, 4),
                'net_clutch_adj': round(net_clutch_adj, 4),
            }
        
        # Apply H2H adjustment
        h2h_info = {}
        h2h_adj = 0.0
        
        if H2H_ENABLED:
            h2h_prob, h2h_meetings = get_h2h_probability(p1_name, p2_name, match_date)
            p1_h2h_wins, p2_h2h_wins, _ = get_h2h_record(p1_name, p2_name, match_date)
            
            if h2h_prob is not None and h2h_meetings >= H2H_MIN_MEETINGS:
                # Blend H2H with current prediction
                # new_prob = (1 - H2H_WEIGHT) * current_prob + H2H_WEIGHT * h2h_prob
                old_prob = p1_win_prob
                p1_win_prob = (1 - H2H_WEIGHT) * p1_win_prob + H2H_WEIGHT * h2h_prob
                p1_win_prob = max(0.05, min(0.95, p1_win_prob))
                h2h_adj = p1_win_prob - old_prob
            
            h2h_info = {
                'p1_h2h_wins': p1_h2h_wins,
                'p2_h2h_wins': p2_h2h_wins,
                'h2h_meetings': h2h_meetings,
                'h2h_prob': round(h2h_prob, 3) if h2h_prob else None,
                'h2h_weight': H2H_WEIGHT if h2h_meetings >= H2H_MIN_MEETINGS else 0,
                'h2h_adj': round(h2h_adj, 4),
            }
        
        # Calculate net fatigue adjustment for output
        p1_serve_adj = p1_fatigue.get('serve_adjustment', 0)
        p2_serve_adj = p2_fatigue.get('serve_adjustment', 0)
        fatigue_adjustment = (p2_serve_adj - p1_serve_adj) * 100  # For display
        
        # Convert to odds
        if p1_win_prob >= 0.5:
            p1_odds = int(-100 * p1_win_prob / (1 - p1_win_prob))
            p2_odds = int(100 * (1 - p1_win_prob) / p1_win_prob)
        else:
            p1_odds = int(100 * (1 - p1_win_prob) / p1_win_prob)
            p2_odds = int(-100 * p1_win_prob / (1 - p1_win_prob))
        
        return {
            'player1': p1_name,
            'player2': p2_name,
            'surface': surface.capitalize(),
            
            # Skills (original, before fatigue adjustment)
            'p1_serve_skill': round(p1_serve_orig, 3),
            'p1_return_skill': round(p1_return, 3),
            'p2_serve_skill': round(p2_serve_orig, 3),
            'p2_return_skill': round(p2_return, 3),
            
            # Hold probabilities (with fatigue-adjusted serve)
            'p1_hold_prob': round(hold_prob_1 * 100, 1),
            'p2_hold_prob': round(hold_prob_2 * 100, 1),
            
            # Component probabilities
            'p1_skill_prob': round(skill_win_prob * 100, 1),
            'p1_elo_prob': round(elo_win_prob * 100, 1),
            
            # Match prediction (blended)
            'p1_win_prob': round(p1_win_prob * 100, 1),
            'p2_win_prob': round((1 - p1_win_prob) * 100, 1),
            'p1_odds': p1_odds,
            'p2_odds': p2_odds,
            
            # Total games
            'total_games': round(sim_result['avg_total_games'], 1),
            'total_std': round(sim_result['std_total_games'], 1),
            
            # ELO
            'p1_elo': p1['elo'],
            'p2_elo': p2['elo'],
            'p1_surface_elo': p1_elo,
            'p2_surface_elo': p2_elo,
            
            # Fatigue (v2 - serve skill adjustments)
            'p1_fatigue': round(p1_fatigue.get('fatigue', 0), 3),
            'p2_fatigue': round(p2_fatigue.get('fatigue', 0), 3),
            'p1_serve_adj': round(p1_serve_adj, 3),
            'p2_serve_adj': round(p2_serve_adj, 3),
            'p1_fatigue_details': p1_fatigue.get('details', {}),
            'p2_fatigue_details': p2_fatigue.get('details', {}),
            'fatigue_adjustment': round(fatigue_adjustment, 1),
            
            # Court speed
            'court_speed': court_speed_info,
            'p1_speed_adj': round(p1_speed_adj, 4),
            'p2_speed_adj': round(p2_speed_adj, 4),
            
            # Handedness
            'hand_info': hand_info,
            'p1_hand_adj': round(p1_hand_adj, 4),
            'p2_hand_adj': round(p2_hand_adj, 4),
            
            # Clutch/Overperformance
            'clutch_info': clutch_info,
            'p1_clutch_adj': round(p1_clutch_adj, 4),
            'p2_clutch_adj': round(p2_clutch_adj, 4),
            
            # Head-to-Head
            'h2h_info': h2h_info,
            'h2h_adj': round(h2h_adj, 4),
        }
    
    def print_prediction(self, player1: str, player2: str, surface: str = 'Hard',
                         tournament: str = None, match_date: str = None):
        """Print formatted prediction."""
        r = self.predict(player1, player2, surface, tournament, match_date)
        
        print(f"\n{'='*60}")
        print(f"{r['player1']} vs {r['player2']} ({r['surface']})")
        if tournament:
            print(f"Tournament: {tournament}")
        print(f"{'='*60}")
        
        print(f"\nSkills ({r['surface']}):")
        print(f"  {r['player1'][:20]:<20} Serve: {r['p1_serve_skill']:+.3f} | Return: {r['p1_return_skill']:+.3f}")
        print(f"  {r['player2'][:20]:<20} Serve: {r['p2_serve_skill']:+.3f} | Return: {r['p2_return_skill']:+.3f}")
        
        print(f"\nHold Probabilities:")
        print(f"  {r['player1'][:20]:<20} {r['p1_hold_prob']:.1f}%")
        print(f"  {r['player2'][:20]:<20} {r['p2_hold_prob']:.1f}%")
        
        print(f"\nComponent Probabilities:")
        print(f"  Skill-based: {r['player1'][:15]} {r['p1_skill_prob']:.1f}%")
        print(f"  ELO-based:   {r['player1'][:15]} {r['p1_elo_prob']:.1f}%")
        
        # Show fatigue if relevant (v2: serve skill adjustments)
        if r['p1_fatigue'] > 0.01 or r['p2_fatigue'] > 0.01:
            print(f"\nFatigue (ServeSkill Adjustment):")
            if r['p1_fatigue'] > 0.01:
                d = r['p1_fatigue_details']
                print(f"  {r['player1'][:20]:<20} Fatigue: {r['p1_fatigue']:.3f} | Serve adj: {r['p1_serve_adj']:+.3f}")
                if d.get('matches_in_window'):
                    print(f"    ({d['matches_in_window']} matches, {d.get('total_points',0)} points)")
                if d.get('travel'):
                    t = d['travel']
                    print(f"    (traveled {t.get('distance_km',0):,}km from {t.get('from','')}, {t.get('timezone_shift',0)}h shift)")
            if r['p2_fatigue'] > 0.01:
                d = r['p2_fatigue_details']
                print(f"  {r['player2'][:20]:<20} Fatigue: {r['p2_fatigue']:.3f} | Serve adj: {r['p2_serve_adj']:+.3f}")
                if d.get('matches_in_window'):
                    print(f"    ({d['matches_in_window']} matches, {d.get('total_points',0)} points)")
                if d.get('travel'):
                    t = d['travel']
                    print(f"    (traveled {t.get('distance_km',0):,}km from {t.get('from','')}, {t.get('timezone_shift',0)}h shift)")
            if abs(r['fatigue_adjustment']) > 0.1:
                beneficiary = r['player1'] if r['fatigue_adjustment'] > 0 else r['player2']
                print(f"  Net serve edge: {abs(r['fatigue_adjustment']):.1f}% toward {beneficiary}")
        
        # Show H2H if relevant
        h2h = r.get('h2h_info', {})
        if h2h.get('h2h_meetings', 0) >= 1:
            print(f"\nHead-to-Head (prior meetings):")
            print(f"  {r['player1'][:20]:<20} {h2h.get('p1_h2h_wins', 0)} wins")
            print(f"  {r['player2'][:20]:<20} {h2h.get('p2_h2h_wins', 0)} wins")
            if h2h.get('h2h_adj', 0) != 0:
                direction = "+" if h2h['h2h_adj'] > 0 else ""
                print(f"  H2H adjustment: {direction}{h2h['h2h_adj']*100:.1f}% for {r['player1']}")
        
        print(f"\nMatch Prediction (blended):")
        print(f"  {r['player1'][:20]:<20} {r['p1_win_prob']:.1f}% ({r['p1_odds']:+d})")
        print(f"  {r['player2'][:20]:<20} {r['p2_win_prob']:.1f}% ({r['p2_odds']:+d})")
        
        winner = r['player1'] if r['p1_win_prob'] > 50 else r['player2']
        odds = r['p1_odds'] if r['p1_win_prob'] > 50 else r['p2_odds']
        print(f"\n>>> PICK: {winner} ({odds:+d})")
        
        print(f"\nTotal Games: {r['total_games']:.1f} (std: {r['total_std']:.1f})")
        print(f"ELO: {r['player1'][:15]} {r['p1_elo']} | {r['player2'][:15]} {r['p2_elo']}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Tennis Skill-Based Predictor')
    parser.add_argument('player1', nargs='?', help='Player 1 name')
    parser.add_argument('player2', nargs='?', help='Player 2 name')
    parser.add_argument('-s', '--surface', default='Hard', help='Surface (Hard/Clay/Grass)')
    parser.add_argument('-t', '--tournament', help='Tournament name (for fatigue calc)')
    parser.add_argument('-d', '--date', help='Match date YYYYMMDD (for fatigue calc)')
    parser.add_argument('-e', '--elo-weight', type=float, default=0.5, help='ELO blend weight (0-1)')
    parser.add_argument('--recalc', action='store_true', help='Force recalculate skills')
    parser.add_argument('--sample', action='store_true', help='Run sample predictions')
    
    args = parser.parse_args()
    
    model = SkillPredictor(elo_weight=args.elo_weight)
    model.train(force_recalc=args.recalc)
    
    if args.sample or (not args.player1 and not args.player2):
        print("\n" + "="*60)
        print("SAMPLE PREDICTIONS")
        print("="*60)
        
        try:
            # Sample without fatigue
            model.print_prediction('Aryna Sabalenka', 'Iga Swiatek', 'Hard')
            
            # Sample WITH fatigue (during Australian Open)
            print("\n" + "-"*60)
            print("EXAMPLE WITH FATIGUE (AO Final scenario)")
            print("-"*60)
            model.print_prediction('Aryna Sabalenka', 'Madison Keys', 'Hard',
                                   tournament='Australian Open', match_date='20260120')
        except Exception as e:
            print(f"Error: {e}")
    elif args.player1 and args.player2:
        model.print_prediction(args.player1, args.player2, args.surface,
                               args.tournament, args.date)
