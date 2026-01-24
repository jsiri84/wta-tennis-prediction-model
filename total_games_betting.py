"""
Total Games Betting Analyzer

Analyzes over/under betting value based on model predictions.
"""

import numpy as np
from scipy import stats
from total_games_model import TotalGamesPredictor


def prob_to_american(prob):
    """Convert probability to American odds"""
    if prob <= 0.01:
        return 9999
    if prob >= 0.99:
        return -9999
    if prob >= 0.5:
        return int(-100 * prob / (1 - prob))
    else:
        return int(100 * (1 - prob) / prob)


def prob_to_decimal(prob):
    """Convert probability to decimal odds"""
    if prob <= 0.01:
        return 100.0
    return 1 / prob


def american_to_prob(odds):
    """Convert American odds to implied probability"""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


class TotalGamesBettingAnalyzer:
    """Analyzes total games betting opportunities"""
    
    MODEL_MAE = 2.57
    MODEL_STD = 3.6  # Approximate standard deviation (RMSE)
    
    # Confidence thresholds
    HIGH_CONFIDENCE = MODEL_MAE * 0.75  # ~2.6 games
    MEDIUM_CONFIDENCE = MODEL_MAE * 0.5  # ~1.7 games
    
    def __init__(self):
        self.predictor = TotalGamesPredictor()
        self.predictor.train()
        self.MODEL_MAE = self.predictor.MODEL_MAE
        self.MODEL_STD = self.MODEL_MAE * 1.2  # Approximate
    
    def analyze_bet(self, player1: str, player2: str, book_line: float,
                    over_odds: int, under_odds: int = None,
                    surface: str = 'Hard', tournament: str = '', match_round: str = 'R32'):
        """
        Analyze an over/under bet on total games.
        
        Args:
            player1: First player name
            player2: Second player name
            book_line: Sportsbook's total games line (e.g., 21.5)
            over_odds: American odds for over (e.g., -110)
            under_odds: American odds for under (default: calculate from over)
            surface: Playing surface
            tournament: Tournament name
            match_round: Round of the match
        """
        if under_odds is None:
            # Estimate under odds from over odds (opposite side of -110)
            under_odds = over_odds
        
        result = self.predictor.predict(player1, player2, surface, tournament, match_round)
        model_total = result['predicted_total']
        
        # Calculate probabilities
        prob_over = self._calculate_over_probability(model_total, book_line)
        prob_under = 1 - prob_over
        
        # Implied probabilities from book
        implied_over = american_to_prob(over_odds)
        implied_under = american_to_prob(under_odds)
        
        # Edge calculation
        over_edge = prob_over - implied_over
        under_edge = prob_under - implied_under
        
        # Determine best side
        if over_edge > under_edge:
            best_side = "OVER"
            best_edge = over_edge
            best_prob = prob_over
            best_odds = over_odds
        else:
            best_side = "UNDER"
            best_edge = under_edge
            best_prob = prob_under
            best_odds = under_odds
        
        # Gap from model prediction
        gap = abs(model_total - book_line)
        
        # Confidence assessment
        if gap >= self.HIGH_CONFIDENCE:
            confidence = "HIGH"
        elif gap >= self.MEDIUM_CONFIDENCE:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        # Expected value
        if best_odds > 0:
            ev = best_prob * (best_odds / 100) - (1 - best_prob)
        else:
            ev = best_prob * (100 / abs(best_odds)) - (1 - best_prob)
        
        return {
            'player1': result['player1'],
            'player2': result['player2'],
            'surface': result['surface'],
            'model_total': model_total,
            'book_line': book_line,
            'gap': model_total - book_line,
            'prob_over': prob_over,
            'prob_under': prob_under,
            'implied_over': implied_over,
            'implied_under': implied_under,
            'over_edge': over_edge,
            'under_edge': under_edge,
            'over_odds': over_odds,
            'under_odds': under_odds,
            'best_side': best_side,
            'best_edge': best_edge,
            'confidence': confidence,
            'expected_value': ev,
            'model_range': result['model_range']
        }
    
    def _calculate_over_probability(self, model_total, line):
        """Calculate probability that actual total exceeds the line"""
        # Using normal distribution centered at model prediction
        return 1 - stats.norm.cdf(line, loc=model_total, scale=self.MODEL_STD)
    
    def print_analysis(self, player1: str, player2: str, book_line: float,
                       over_odds: int, under_odds: int = None,
                       surface: str = 'Hard', tournament: str = '', match_round: str = 'R32'):
        """Print formatted betting analysis"""
        
        result = self.analyze_bet(player1, player2, book_line, over_odds, under_odds,
                                  surface, tournament, match_round)
        
        print("\n" + "="*65)
        print("TOTAL GAMES BETTING ANALYSIS")
        print("="*65)
        
        print(f"\n{result['player1']} vs {result['player2']}")
        print(f"Surface: {result['surface']}")
        
        print(f"\n{'-'*65}")
        print("MODEL PREDICTION:")
        print(f"{'-'*65}")
        print(f"  Predicted Total: {result['model_total']:.1f} games")
        print(f"  Model Range:     {result['model_range']} games")
        print(f"  Book Line:       {result['book_line']:.1f} games")
        print(f"  Gap:             {result['gap']:+.1f} games")
        
        print(f"\n{'-'*65}")
        print("PROBABILITIES:")
        print(f"{'-'*65}")
        print(f"  Model:")
        print(f"    Over  {result['book_line']}: {result['prob_over']*100:>5.1f}%  ({prob_to_american(result['prob_over']):>+5})")
        print(f"    Under {result['book_line']}: {result['prob_under']*100:>5.1f}%  ({prob_to_american(result['prob_under']):>+5})")
        print(f"  Implied (Book):")
        print(f"    Over  {result['book_line']}: {result['implied_over']*100:>5.1f}%  ({result['over_odds']:>+5})")
        print(f"    Under {result['book_line']}: {result['implied_under']*100:>5.1f}%  ({result['under_odds']:>+5})")
        
        print(f"\n{'-'*65}")
        print("EDGE ANALYSIS:")
        print(f"{'-'*65}")
        print(f"  Over Edge:  {result['over_edge']*100:>+5.1f}%")
        print(f"  Under Edge: {result['under_edge']*100:>+5.1f}%")
        
        print(f"\n{'-'*65}")
        print("RECOMMENDATION:")
        print(f"{'-'*65}")
        
        # Recommendation logic
        if result['best_edge'] > 0.05:  # 5% edge minimum
            if result['confidence'] == "HIGH":
                rec = f"STRONG {result['best_side']}"
                icon = "***"
            elif result['confidence'] == "MEDIUM":
                rec = f"{result['best_side']}"
                icon = "**"
            else:
                rec = f"Lean {result['best_side']}"
                icon = "*"
        elif result['best_edge'] > 0.02:
            rec = f"Slight {result['best_side']}"
            icon = "*"
        else:
            rec = "NO VALUE - Pass"
            icon = ""
        
        print(f"  {icon} {rec} {icon}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Expected Value: {result['expected_value']*100:+.1f}%")
        
        # Over/Under table for different lines
        print(f"\n{'-'*65}")
        print("O/U ODDS TABLE:")
        print(f"{'-'*65}")
        print(f"{'Line':<10} {'Over Prob':<12} {'Over Odds':<12} {'Under Prob':<12} {'Under Odds':<12}")
        print("-"*60)
        
        model_total = result['model_total']
        for line in [19.5, 20.5, 21.5, 22.5, 23.5, 24.5]:
            p_over = self._calculate_over_probability(model_total, line)
            p_under = 1 - p_over
            
            marker = " <--" if abs(line - result['book_line']) < 0.1 else ""
            print(f"{line:<10.1f} {p_over*100:>5.1f}%      {prob_to_american(p_over):>+6}      {p_under*100:>5.1f}%       {prob_to_american(p_under):>+6}{marker}")
        
        print("="*65)
        
        return result


def main():
    """Demo the betting analyzer"""
    analyzer = TotalGamesBettingAnalyzer()
    
    # Example analyses
    print("\n" + "="*65)
    print("TOTAL GAMES BETTING EXAMPLES")
    print("="*65)
    
    examples = [
        # (player1, player2, book_line, over_odds, under_odds, surface, tournament, round)
        ("Aryna Sabalenka", "Iga Swiatek", 21.5, -110, -110, "Hard", "Australian Open", "SF"),
        ("Coco Gauff", "Naomi Osaka", 22.5, -115, -105, "Hard", "", "R32"),
        ("Elena Rybakina", "Madison Keys", 20.5, -105, -115, "Hard", "", "QF"),
    ]
    
    for args in examples:
        try:
            analyzer.print_analysis(*args)
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == '__main__':
    main()
