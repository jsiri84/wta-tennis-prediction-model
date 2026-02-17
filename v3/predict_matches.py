"""
Match Prediction Script - Standard Output Format

Usage:
    python predict_matches.py

Edit the 'matches' list below with:
    (Player1, Player2, P1_Book_Odds, P2_Book_Odds, Surface, Tournament)

If no book odds available, use None for both odds.
"""

from skill_model import SkillPredictor
from datetime import datetime

def american_to_prob(odds):
    """Convert American odds to implied probability."""
    if odds is None:
        return None
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)

def prob_to_american(prob):
    """Convert probability to American odds."""
    if prob >= 0.99:
        return -9999
    if prob <= 0.01:
        return 9999
    if prob >= 0.5:
        return int(-100 * prob / (1 - prob))
    else:
        return int(100 * (1 - prob) / prob)

def format_odds(odds):
    """Format American odds with +/- sign."""
    if odds is None:
        return "N/A"
    if odds > 0:
        return f"+{odds}"
    return str(odds)


# =============================================================================
# EDIT MATCHES HERE
# Format: (Player1, Player2, P1_Book_Odds, P2_Book_Odds, Surface, Tournament)
# Set book odds to None if not available
# =============================================================================

matches = [
    # Example format:
    # ("Aryna Sabalenka", "Iga Swiatek", -150, +130, "Hard", "Dubai"),
]

# =============================================================================

def run_predictions(matches, match_date=None):
    """Run predictions and output in standard format."""
    
    if match_date is None:
        match_date = datetime.now().strftime('%Y%m%d')
    
    if not matches:
        print("No matches defined. Edit the 'matches' list in this file.")
        return
    
    # Initialize model
    model = SkillPredictor()
    model.train()
    
    # Get tournament/surface from first match
    surface = matches[0][4] if len(matches[0]) > 4 else "Hard"
    tournament = matches[0][5] if len(matches[0]) > 5 else None
    
    print("=" * 120)
    if tournament:
        print(f"{tournament.upper()} - {surface} - {match_date}")
    else:
        print(f"PREDICTIONS - {surface} - {match_date}")
    print("=" * 120)
    print()
    
    # Header
    print(f"{'P1':<18} {'P2':<18} {'Pick':<18} {'Conf':>6} {'P1 ML':>8} {'P1 Book':>8} {'P1 CLV':>8} {'P2 ML':>8} {'P2 Book':>8} {'P2 CLV':>8}")
    print("-" * 120)
    
    results = []
    
    for match in matches:
        p1, p2 = match[0], match[1]
        p1_book = match[2] if len(match) > 2 else None
        p2_book = match[3] if len(match) > 3 else None
        surf = match[4] if len(match) > 4 else "Hard"
        tourn = match[5] if len(match) > 5 else None
        
        try:
            pred = model.predict(p1, p2, surface=surf, tournament=tourn, match_date=match_date)
            
            model_p1 = pred['p1_win_prob'] / 100
            model_p2 = pred['p2_win_prob'] / 100
            
            p1_ml = prob_to_american(model_p1)
            p2_ml = prob_to_american(model_p2)
            
            # Calculate CLV if book odds provided
            if p1_book is not None and p2_book is not None:
                book_p1 = american_to_prob(p1_book)
                book_p2 = american_to_prob(p2_book)
                clv_p1 = (model_p1 - book_p1) * 100
                clv_p2 = (model_p2 - book_p2) * 100
                clv_p1_str = f"{clv_p1:+.1f}%" if clv_p1 < 0 else f"**{clv_p1:+.1f}%**"
                clv_p2_str = f"{clv_p2:+.1f}%" if clv_p2 < 0 else f"**{clv_p2:+.1f}%**"
            else:
                clv_p1_str = "N/A"
                clv_p2_str = "N/A"
                clv_p1 = 0
                clv_p2 = 0
            
            # Determine pick
            if model_p1 > model_p2:
                pick = pred['player1']
                conf = pred['p1_win_prob']
            else:
                pick = pred['player2']
                conf = pred['p2_win_prob']
            
            print(f"{pred['player1']:<18} {pred['player2']:<18} **{pick:<16}** {conf:>5.1f}% {format_odds(p1_ml):>8} {format_odds(p1_book):>8} {clv_p1_str:>8} {format_odds(p2_ml):>8} {format_odds(p2_book):>8} {clv_p2_str:>8}")
            
            results.append({
                'p1': pred['player1'],
                'p2': pred['player2'],
                'pick': pick,
                'conf': conf,
                'p1_ml': p1_ml,
                'p2_ml': p2_ml,
                'p1_book': p1_book,
                'p2_book': p2_book,
                'clv_p1': clv_p1,
                'clv_p2': clv_p2,
            })
            
        except Exception as e:
            print(f"{p1:<18} {p2:<18} ERROR: {e}")
    
    print()
    print("**Bold CLV = Positive value bet**")
    print()
    
    # Value bets summary
    if any(r['p1_book'] is not None for r in results):
        value_bets = []
        for r in results:
            if r['clv_p1'] > 0:
                value_bets.append((r['p1'], r['clv_p1'], r['p1_book']))
            if r['clv_p2'] > 0:
                value_bets.append((r['p2'], r['clv_p2'], r['p2_book']))
        
        if value_bets:
            print("=" * 60)
            print("VALUE BETS (Positive CLV)")
            print("=" * 60)
            value_bets.sort(key=lambda x: x[1], reverse=True)
            for name, clv, odds in value_bets:
                print(f"  {name:<25} {format_odds(odds):>8}  CLV: +{clv:.1f}%")
    
    return results


if __name__ == "__main__":
    run_predictions(matches)
