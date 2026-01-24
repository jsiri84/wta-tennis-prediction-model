"""
Integrated Tennis Prediction Model

Chains all three models together for consistent predictions:
1. Winner model → win probability
2. Spread model (informed by win prob) → game spread
3. Total games model (informed by win prob + spread) → total games with consistency check
"""

import numpy as np
from prediction_model import TennisPredictor
from game_spread_model import GameSpreadPredictor
from total_games_model import TotalGamesPredictor


def spread_to_expected_total(spread: float) -> tuple:
    """
    Convert a game spread to expected total games range.
    Based on tennis math constraints.
    
    Returns (min_total, expected_total, max_total)
    """
    abs_spread = abs(spread)
    
    # Empirical relationship: larger spread = fewer total games
    # In straight sets with spread S: typical total is around 12 + S + some loser games
    
    if abs_spread >= 10:
        # Double bagel territory: 6-0, 6-0 to 6-1, 6-1
        return (12, 13, 15)
    elif abs_spread >= 8:
        # Dominant win: 6-1, 6-1 to 6-2, 6-2
        return (14, 16, 18)
    elif abs_spread >= 6:
        # Comfortable win: 6-2, 6-2 to 6-3, 6-3
        return (16, 17, 19)
    elif abs_spread >= 4:
        # Solid win: 6-3, 6-3 to 6-4, 6-4
        return (17, 19, 21)
    elif abs_spread >= 2:
        # Competitive: 6-4, 6-4 to 7-5, 6-4
        return (19, 21, 24)
    else:
        # Very close / likely 3 sets
        return (20, 23, 30)


def win_prob_to_spread_adjustment(win_prob: float) -> float:
    """
    Convert win probability to an expected spread multiplier.
    Higher win prob = larger expected spread.
    """
    # At 50% win prob, no adjustment
    # At 95% win prob, expect larger spread
    # At 5% win prob, expect negative spread
    
    # Convert probability to a spread factor
    # Using logit-like transformation
    if win_prob >= 0.99:
        return 1.5
    elif win_prob <= 0.01:
        return 0.5
    
    # Sigmoid-ish relationship
    # 50% -> 1.0, 75% -> 1.15, 90% -> 1.3, 95% -> 1.4
    excess_prob = win_prob - 0.5
    adjustment = 1.0 + (excess_prob * 0.8)
    
    return max(0.5, min(1.5, adjustment))


class IntegratedPredictor:
    """
    Integrated tennis match predictor that ensures consistency
    across winner, spread, and total games predictions.
    """
    
    def __init__(self):
        self.winner_model = TennisPredictor()
        self.spread_model = GameSpreadPredictor()
        self.totals_model = TotalGamesPredictor()
        self.is_trained = False
    
    def train(self, force_retrain=False):
        """Load/train all three models"""
        print("Loading models...")
        self.winner_model.train(force_retrain=force_retrain)
        self.spread_model.train(force_retrain=force_retrain)
        self.totals_model.train(force_retrain=force_retrain)
        self.is_trained = True
        print("All models ready.")
    
    def predict(self, player1: str, player2: str, surface: str = 'Hard',
                tournament: str = '', match_round: str = 'R32'):
        """
        Generate consistent predictions across all three models.
        """
        if not self.is_trained:
            raise ValueError("Models not trained. Call train() first.")
        
        # Step 1: Get win probability
        winner_result = self.winner_model.predict(player1, player2, surface)
        win_prob = winner_result['player1']['win_prob'] / 100  # Convert to decimal
        
        # Step 2: Get raw spread prediction
        spread_result = self.spread_model.predict(player1, player2, surface)
        raw_spread = spread_result['predicted_spread']
        
        # Step 3: Adjust spread based on win probability for consistency
        # If win_prob is very high (>90%), spread should be positive and large
        # If win_prob is very low (<10%), spread should be negative and large
        spread_adjustment = win_prob_to_spread_adjustment(win_prob)
        
        # The spread sign should match the win probability
        expected_spread_sign = 1 if win_prob > 0.5 else -1
        actual_spread_sign = 1 if raw_spread > 0 else -1
        
        # If signs disagree, there's an inconsistency - trust win probability more
        if expected_spread_sign != actual_spread_sign and abs(win_prob - 0.5) > 0.2:
            # Flip the spread to match win probability direction
            adjusted_spread = -raw_spread * 0.5  # Dampened flip
        else:
            adjusted_spread = raw_spread * spread_adjustment
        
        # Ensure spread magnitude matches win probability magnitude
        # Very high win prob (>90%) should have spread of at least 4-5
        if win_prob > 0.9 and abs(adjusted_spread) < 4:
            adjusted_spread = 4.0 * expected_spread_sign + (adjusted_spread * 0.3)
        elif win_prob > 0.8 and abs(adjusted_spread) < 2:
            adjusted_spread = 2.0 * expected_spread_sign + (adjusted_spread * 0.5)
        
        # Step 4: Get total games prediction
        totals_result = self.totals_model.predict(player1, player2, surface, tournament, match_round)
        raw_total = totals_result['predicted_total']
        
        # Step 5: Apply consistency constraint based on spread
        min_total, expected_total, max_total = spread_to_expected_total(adjusted_spread)
        
        # Blend raw prediction with spread-implied expectation
        # Weight towards spread-implied when spread is extreme
        spread_magnitude = abs(adjusted_spread)
        
        if spread_magnitude >= 8:
            # Strong mismatch - trust spread-implied total more
            blend_weight = 0.7  # 70% spread-implied, 30% raw model
        elif spread_magnitude >= 5:
            blend_weight = 0.5
        else:
            blend_weight = 0.3  # Trust raw model more for close matches
        
        adjusted_total = (blend_weight * expected_total) + ((1 - blend_weight) * raw_total)
        
        # Clamp to plausible range
        adjusted_total = max(min_total, min(max_total, adjusted_total))
        
        # Step 6: Final consistency check
        # Verify the math works: winner_games = (total + spread) / 2
        winner_games = (adjusted_total + abs(adjusted_spread)) / 2
        loser_games = (adjusted_total - abs(adjusted_spread)) / 2
        
        # Ensure loser has non-negative games
        if loser_games < 0:
            # Adjust total upward
            adjusted_total = abs(adjusted_spread)  # Minimum: loser gets 0
            loser_games = 0
            winner_games = adjusted_total
        
        return {
            'player1': winner_result['player1']['name'],
            'player2': winner_result['player2']['name'],
            'surface': surface.capitalize(),
            'tournament': tournament,
            'round': match_round,
            
            # Winner prediction
            'p1_win_prob': winner_result['player1']['win_prob'],
            'p2_win_prob': winner_result['player2']['win_prob'],
            'p1_odds': winner_result['player1']['american_odds'],
            'p2_odds': winner_result['player2']['american_odds'],
            'p1_decimal': winner_result['player1']['decimal_odds'],
            'p2_decimal': winner_result['player2']['decimal_odds'],
            'p1_elo': winner_result['player1']['elo'],
            'p2_elo': winner_result['player2']['elo'],
            'p1_surface_elo': winner_result['player1']['surface_elo'],
            'p2_surface_elo': winner_result['player2']['surface_elo'],
            'h2h': winner_result['h2h'],
            
            # Spread prediction (adjusted for consistency)
            'spread': round(adjusted_spread, 1),
            'spread_raw': raw_spread,
            'spread_favors': winner_result['player1']['name'] if adjusted_spread > 0 else winner_result['player2']['name'],
            
            # Total games (adjusted for consistency)
            'total_games': round(adjusted_total, 1),
            'total_raw': raw_total,
            'total_range': (round(min_total, 1), round(max_total, 1)),
            
            # Implied scoreline
            'implied_winner_games': round(winner_games, 1),
            'implied_loser_games': round(loser_games, 1),
        }
    
    def print_prediction(self, player1: str, player2: str, surface: str = 'Hard',
                         tournament: str = '', match_round: str = 'R32'):
        """Print formatted integrated prediction"""
        
        result = self.predict(player1, player2, surface, tournament, match_round)
        
        print("\n" + "="*70)
        print(f"INTEGRATED MATCH PREDICTION")
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
        print(f"\n  ELO: {result['player1']} {result['p1_elo']} | {result['player2']} {result['p2_elo']}")
        print(f"  Surface ELO: {result['player1']} {result['p1_surface_elo']} | {result['player2']} {result['p2_surface_elo']}")
        
        # Spread prediction
        print(f"\n{'-'*70}")
        print("GAME SPREAD (Consistency-Adjusted)")
        print(f"{'-'*70}")
        spread_display = f"{result['spread_favors']} {abs(result['spread']):.1f}"
        print(f"  Predicted Spread: {spread_display}")
        if abs(result['spread'] - result['spread_raw']) > 0.5:
            print(f"  (Raw model: {result['spread_raw']:+.1f}, adjusted for win prob consistency)")
        
        # Total games prediction
        print(f"\n{'-'*70}")
        print("TOTAL GAMES (Consistency-Adjusted)")
        print(f"{'-'*70}")
        print(f"  Predicted Total: {result['total_games']:.1f} games")
        print(f"  Plausible Range: {result['total_range'][0]:.0f} - {result['total_range'][1]:.0f} games")
        if abs(result['total_games'] - result['total_raw']) > 1:
            print(f"  (Raw model: {result['total_raw']:.1f}, adjusted for spread consistency)")
        
        # Implied scoreline
        print(f"\n{'-'*70}")
        print("IMPLIED SCORELINE")
        print(f"{'-'*70}")
        winner = result['player1'] if result['p1_win_prob'] > 50 else result['player2']
        loser = result['player2'] if result['p1_win_prob'] > 50 else result['player1']
        print(f"  {winner}: ~{result['implied_winner_games']:.0f} games")
        print(f"  {loser}: ~{result['implied_loser_games']:.0f} games")
        
        # Suggest likely scorelines
        w_games = result['implied_winner_games']
        l_games = result['implied_loser_games']
        
        likely_scores = self._suggest_scorelines(w_games, l_games)
        if likely_scores:
            print(f"  Likely scorelines: {', '.join(likely_scores)}")
        
        # Over/under probabilities
        print(f"\n{'-'*70}")
        print("OVER/UNDER PROBABILITIES")
        print(f"{'-'*70}")
        
        from scipy import stats
        total = result['total_games']
        std = 2.5  # Reduced std for adjusted model
        
        for line in [19.5, 20.5, 21.5, 22.5, 23.5]:
            prob_over = 1 - stats.norm.cdf(line, loc=total, scale=std)
            prob_under = 1 - prob_over
            
            over_odds = self._prob_to_american(prob_over)
            under_odds = self._prob_to_american(prob_under)
            
            marker = " <--" if abs(line - total) < 1 else ""
            print(f"  O/U {line}: Over {prob_over*100:>5.1f}% ({over_odds:>+5}) | Under {prob_under*100:>5.1f}% ({under_odds:>+5}){marker}")
        
        print("="*70)
        
        return result
    
    def _suggest_scorelines(self, winner_games: float, loser_games: float) -> list:
        """Suggest likely scorelines based on game counts"""
        scores = []
        w = round(winner_games)
        l = round(loser_games)
        
        # Two-set possibilities
        two_set_options = [
            (6, 0, 6, 0), (6, 0, 6, 1), (6, 0, 6, 2), (6, 0, 6, 3), (6, 0, 6, 4),
            (6, 1, 6, 0), (6, 1, 6, 1), (6, 1, 6, 2), (6, 1, 6, 3), (6, 1, 6, 4),
            (6, 2, 6, 0), (6, 2, 6, 1), (6, 2, 6, 2), (6, 2, 6, 3), (6, 2, 6, 4),
            (6, 3, 6, 0), (6, 3, 6, 1), (6, 3, 6, 2), (6, 3, 6, 3), (6, 3, 6, 4),
            (6, 4, 6, 0), (6, 4, 6, 1), (6, 4, 6, 2), (6, 4, 6, 3), (6, 4, 6, 4),
            (7, 5, 6, 0), (7, 5, 6, 1), (7, 5, 6, 2), (7, 5, 6, 3), (7, 5, 6, 4),
            (6, 0, 7, 5), (6, 1, 7, 5), (6, 2, 7, 5), (6, 3, 7, 5), (6, 4, 7, 5),
            (7, 5, 7, 5), (7, 6, 6, 4), (6, 4, 7, 6), (7, 6, 7, 5), (7, 5, 7, 6),
            (7, 6, 7, 6),
        ]
        
        for s1w, s1l, s2w, s2l in two_set_options:
            total_w = s1w + s2w
            total_l = s1l + s2l
            if abs(total_w - w) <= 2 and abs(total_l - l) <= 2:
                scores.append(f"{s1w}-{s1l}, {s2w}-{s2l}")
        
        # Return top 3 closest
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
    predictor = IntegratedPredictor()
    predictor.train()
    
    # Test with the problematic matchup
    print("\n" + "="*70)
    print("TEST: Osaka vs Inglis (previously inconsistent)")
    print("="*70)
    predictor.print_prediction('Naomi Osaka', 'Maddison Inglis', 'Hard', 'Australian Open', 'R16')
    
    # Test with a closer matchup
    print("\n" + "="*70)
    print("TEST: Swiatek vs Kalinskaya (closer match)")
    print("="*70)
    predictor.print_prediction('Iga Swiatek', 'Anna Kalinskaya', 'Hard', 'Australian Open', 'R16')
