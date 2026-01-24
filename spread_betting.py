"""
Game Spread Betting Value Calculator

Uses the spread prediction model to identify value bets.
"""

from game_spread_model import GameSpreadPredictor, load_data
import numpy as np


class SpreadBettingAnalyzer:
    """Analyzes spread bets for value"""
    
    # Model's mean absolute error
    MODEL_MAE = 1.96
    
    # Confidence thresholds
    HIGH_CONFIDENCE = 1.0  # Model prediction differs from line by > 1 MAE
    MEDIUM_CONFIDENCE = 0.5  # Model prediction differs from line by > 0.5 MAE
    
    def __init__(self):
        self.predictor = GameSpreadPredictor()
        self.predictor.train()
    
    def analyze_bet(self, player1: str, player2: str, book_spread: float, 
                    surface: str = 'Hard', juice: int = -110):
        """
        Analyze a spread bet for value.
        
        Args:
            player1: First player name
            player2: Second player name
            book_spread: The spread from sportsbook (negative = player1 favored)
                        e.g., -4.5 means player1 must win by 5+ games
            surface: 'Hard', 'Clay', or 'Grass'
            juice: The odds on the bet (default -110)
        
        Returns:
            Analysis dict with recommendation
        """
        # Get model prediction
        result = self.predictor.predict(player1, player2, surface)
        model_spread = result['predicted_spread']
        
        # Calculate margin difference
        # book_spread is negative when P1 is favored (e.g., -3.5)
        # model_spread is positive when P1 wins more games
        model_margin = model_spread  # Positive = P1 wins by this many
        book_margin = -book_spread   # Convert: -3.5 line means P1 expected to win by 3.5
        margin_diff = model_margin - book_margin  # Positive = model expects bigger P1 win
        
        # Determine confidence level
        edge_in_mae = abs(margin_diff) / self.MODEL_MAE
        
        if edge_in_mae >= self.HIGH_CONFIDENCE:
            confidence = 'HIGH'
        elif edge_in_mae >= self.MEDIUM_CONFIDENCE:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        # Determine recommendation
        
        if abs(margin_diff) < self.MODEL_MAE * 0.5:
            recommendation = 'NO BET - Line is fair'
            value_side = None
        elif margin_diff > 0:
            # Model thinks P1 will win by MORE than the book expects
            # Value is on P1 (taking the spread / covering)
            recommendation = f'VALUE ON {result["player1"]} {book_spread}'
            value_side = player1
        else:
            # Model thinks P1 will win by LESS than book expects (or lose)
            # Value is on P2 (taking +spread / underdog)
            recommendation = f'VALUE ON {result["player2"]} +{abs(book_spread)}'
            value_side = player2
        
        # Calculate implied win probability for the spread
        # Using model's prediction and normal distribution approximation
        from scipy import stats
        
        # If book spread is -4.5, P1 needs to win by 5+ games
        # P(spread > 4.5) given model predicts 'model_spread' with std ~2.5
        std_estimate = 2.5  # Approximate from RMSE
        
        if book_spread < 0:
            # P1 favored, need P1 to cover (win by more than |spread|)
            prob_cover = 1 - stats.norm.cdf(abs(book_spread), loc=model_spread, scale=std_estimate)
        else:
            # P2 favored (or P1 underdog), P1 covers if spread > -book_spread
            prob_cover = 1 - stats.norm.cdf(-book_spread, loc=model_spread, scale=std_estimate)
        
        # Calculate expected value
        # At -110, you risk 110 to win 100
        implied_prob = abs(juice) / (abs(juice) + 100) if juice < 0 else 100 / (juice + 100)
        
        if value_side == player1:
            ev = (prob_cover * 100) - ((1 - prob_cover) * abs(juice) if juice < 0 else (1 - prob_cover) * 100)
        elif value_side == player2:
            prob_cover_p2 = 1 - prob_cover
            ev = (prob_cover_p2 * 100) - ((1 - prob_cover_p2) * abs(juice) if juice < 0 else (1 - prob_cover_p2) * 100)
        else:
            ev = 0
        
        # Calculate probabilities for different spread values
        spread_odds = {}
        for spread_val in [-7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5]:
            # P(P1 wins by more than |spread_val|)
            prob = 1 - stats.norm.cdf(abs(spread_val), loc=model_spread, scale=std_estimate)
            if prob > 0.01 and prob < 0.99:  # Only include reasonable probabilities
                spread_odds[spread_val] = {
                    'prob': round(prob * 100, 1),
                    'american': self._prob_to_american(prob),
                    'decimal': round(1 / prob, 2) if prob > 0 else 99.99
                }
        
        return {
            'player1': result['player1'],
            'player2': result['player2'],
            'surface': surface,
            'model_spread': model_spread,
            'book_spread': book_spread,
            'margin_diff': round(margin_diff, 1),
            'edge_mae': round(abs(margin_diff) / self.MODEL_MAE, 2),
            'confidence': confidence,
            'recommendation': recommendation,
            'value_side': value_side,
            'model_prob_p1_covers': round(prob_cover * 100, 1),
            'implied_prob': round(implied_prob * 100, 1),
            'expected_value': round(ev, 1),
            'model_range': f"{model_spread - self.MODEL_MAE:.1f} to {model_spread + self.MODEL_MAE:.1f}",
            'spread_odds': spread_odds
        }
    
    def _prob_to_american(self, prob):
        """Convert probability to American odds"""
        if prob <= 0 or prob >= 1:
            return 0
        if prob >= 0.5:
            return int(-100 * prob / (1 - prob))
        else:
            return int(100 * (1 - prob) / prob)
    
    def print_analysis(self, player1: str, player2: str, book_spread: float,
                       surface: str = 'Hard', juice: int = -110):
        """Print formatted analysis"""
        
        result = self.analyze_bet(player1, player2, book_spread, surface, juice)
        
        print("\n" + "="*70)
        print("SPREAD BETTING ANALYSIS")
        print("="*70)
        print(f"\nMatch: {result['player1']} vs {result['player2']} ({result['surface']})")
        book_str = f"{result['book_spread']:+.1f}" if result['book_spread'] != 0 else "0"
        # Convert model spread to sportsbook convention (negative = favored)
        model_line = -result['model_spread']
        print(f"\nSportsbook Line: {result['player1']} {book_str} ({juice})")
        print(f"Model Prediction: {result['player1']} {model_line:+.1f} (wins by {abs(result['model_spread']):.1f} games)")
        print(f"Model Range (+/-MAE): {result['model_range']} games")
        
        print(f"\n{'-'*70}")
        print("VALUE ANALYSIS:")
        print(f"{'-'*70}")
        print(f"  Model vs Book Margin: {result['margin_diff']:+.1f} games")
        print(f"  Edge (in MAE units): {result['edge_mae']:.2f}")
        print(f"  Confidence: {result['confidence']}")
        
        print(f"\n{'-'*70}")
        print("PROBABILITY ANALYSIS:")
        print(f"{'-'*70}")
        print(f"  Model P({result['player1']} covers {result['book_spread']}): {result['model_prob_p1_covers']}%")
        print(f"  Implied Prob from Odds: {result['implied_prob']}%")
        
        print(f"\n{'-'*70}")
        print("SPREAD ODDS TABLE:")
        print(f"{'-'*70}")
        print(f"  {'Spread':<10} {'Prob':<10} {'American':<12} {'Decimal':<10}")
        print(f"  {'-'*42}")
        
        for spread_val, odds in sorted(result['spread_odds'].items(), reverse=True):
            fav_indicator = " <-- Model" if abs(spread_val - (-result['model_spread'])) < 0.5 else ""
            print(f"  {result['player1'][:12]:<12} {spread_val:+.1f}   {odds['prob']:>5.1f}%   {odds['american']:>+6}   {odds['decimal']:>6.2f}{fav_indicator}")
        
        print(f"\n{'-'*70}")
        print(f"RECOMMENDATION: {result['recommendation']}")
        if result['expected_value'] != 0:
            print(f"Expected Value: {result['expected_value']:+.1f} units per 100 risked")
        print("="*70)
        
        return result


def betting_guide():
    """Print a guide for using the model"""
    
    print("""
======================================================================
                    SPREAD BETTING VALUE GUIDE                        
======================================================================

  HOW TO FIND VALUE:
  ------------------
  1. Get the model's predicted spread
  2. Compare to sportsbook line
  3. If difference > 1.96 games (1 MAE), consider betting

  CONFIDENCE LEVELS:
  ------------------
  HIGH:   Diff > 2.0 games  (> 1 MAE)   = Strong value signal
  MEDIUM: Diff > 1.0 games  (> 0.5 MAE) = Moderate value signal
  LOW:    Diff < 1.0 games              = No significant value

  EXAMPLE:
  --------
  Book: Sabalenka -5.5
  Model: Sabalenka -3.5 (+/- 2 games)
  Diff: 2.0 games = HIGH confidence on underdog (+5.5)

  BANKROLL MANAGEMENT:
  --------------------
  * HIGH confidence: 2-3% of bankroll
  * MEDIUM confidence: 1-2% of bankroll
  * LOW confidence: Pass or 0.5-1%

  REMEMBER:
  ---------
  * Model has ~2 game error on average
  * Even "value" bets lose sometimes
  * Long-term edge, not guaranteed wins
  * Line movement may indicate information you don't have

======================================================================
""")


if __name__ == '__main__':
    # Print guide
    betting_guide()
    
    # Initialize analyzer
    analyzer = SpreadBettingAnalyzer()
    
    # Example analyses
    print("\n" + "="*70)
    print("EXAMPLE BET ANALYSES")
    print("="*70)
    
    # Example 1: Close line
    analyzer.print_analysis("Aryna Sabalenka", "Iga Swiatek", -3.5, "Hard")
    
    # Example 2: Potential value on favorite
    analyzer.print_analysis("Coco Gauff", "Naomi Osaka", -3.5, "Hard")
    
    # Example 3: Potential value on underdog
    analyzer.print_analysis("Iga Swiatek", "Elena Rybakina", -7.5, "Clay")
