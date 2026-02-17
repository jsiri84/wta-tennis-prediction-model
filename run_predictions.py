"""
Run predictions for WTA 1000 matches and save to file.
"""
from unified_model import UnifiedPredictor
from datetime import datetime

# Today's matchups from WTA 1000 (outdoor hard court)
MATCHES = [
    ("Liudmila Samsonova", "Magdalena Frech", "Hard"),
    ("Laura Siegemund", "Varvara Gracheva", "Hard"),
    ("Peyton Stearns", "Vera Zvonareva", "Hard"),
    ("Diana Shnaider", "Alycia Parks", "Hard"),
    ("Mccartney Kessler", "Elsa Jacquemot", "Hard"),
    ("Anastasia Pavlyuchenkova", "Elise Mertens", "Hard"),
    ("Solana Sierra", "Karolina Pliskova", "Hard"),
    ("Sonay Kartal", "Magda Linette", "Hard"),
    ("Ann Li", "Leylah Fernandez", "Hard"),
    ("Daria Kasatkina", "Moyuka Uchijima", "Hard"),
    ("Marie Bouzkova", "Victoria Mboko", "Hard"),
    ("Jaqueline Cristian", "Karolina Muchova", "Hard"),
]

def main():
    # Initialize and train model
    print("="*70)
    print("INITIALIZING MODEL")
    print("="*70)
    
    predictor = UnifiedPredictor()
    predictor.train(force_retrain=True)
    
    # Run predictions
    results = []
    errors = []
    
    print("\n" + "="*70)
    print("RUNNING PREDICTIONS")
    print("="*70)
    
    for p1, p2, surface in MATCHES:
        try:
            r = predictor.predict(p1, p2, surface, tournament="WTA 1000", match_round="R64")
            results.append(r)
            print(f"  OK: {p1} vs {p2}")
        except Exception as e:
            errors.append((p1, p2, str(e)))
            print(f"  SKIP: {p1} vs {p2}: {e}")
    
    # Generate output
    output_lines = []
    output_lines.append("="*70)
    output_lines.append(f"WTA 1000 PREDICTIONS - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    output_lines.append("Surface: Outdoor Hard Court")
    output_lines.append("="*70)
    output_lines.append("")
    
    for r in results:
        # Determine winner
        if r['p1_win_prob'] > 50:
            pick = r['player1']
            pick_prob = r['p1_win_prob']
            pick_odds = r['p1_odds']
        else:
            pick = r['player2']
            pick_prob = r['p2_win_prob']
            pick_odds = r['p2_odds']
        
        output_lines.append("-"*70)
        output_lines.append(f"{r['player1']} vs {r['player2']}")
        output_lines.append("-"*70)
        output_lines.append(f"  H2H: {r['h2h']}")
        output_lines.append(f"  ELO: {r['player1']} {r['p1_elo']} | {r['player2']} {r['p2_elo']} ({r['elo_diff']:+d})")
        output_lines.append("")
        
        # Moneyline
        output_lines.append("  MONEYLINE:")
        output_lines.append(f"    {r['player1']}: {r['p1_win_prob']:.1f}% ({r['p1_odds']:+d})")
        output_lines.append(f"    {r['player2']}: {r['p2_win_prob']:.1f}% ({r['p2_odds']:+d})")
        output_lines.append(f"    >>> PICK: {pick} ({pick_odds:+d})")
        
        # Top factors for winner
        if r.get('winner_factors'):
            output_lines.append(f"    Why: {r['winner_factors'][0]['explanation']}")
        output_lines.append("")
        
        # Spread
        output_lines.append("  SPREAD:")
        output_lines.append(f"    {r['spread_favors']} -{r['spread_line']}")
        output_lines.append(f"    >>> PICK: {r['spread_favors']} -{r['spread_line']}")
        
        # Top factors for spread
        if r.get('spread_factors'):
            output_lines.append(f"    Why: {r['spread_factors'][0]['explanation']}")
        output_lines.append("")
        
        # Total
        output_lines.append("  TOTAL GAMES:")
        output_lines.append(f"    Predicted: {r['total']:.1f} games")
        output_lines.append(f"    Implied score: {r['winner_games']:.0f}-{r['loser_games']:.0f}")
        output_lines.append(f"    >>> LINE: O/U {r['total']:.1f}")
        
        # Top factors for total
        if r.get('total_factors'):
            output_lines.append(f"    Why: {r['total_factors'][0]['explanation']}")
        output_lines.append("")
    
    # Add any errors
    if errors:
        output_lines.append("\n" + "="*70)
        output_lines.append("MATCHES NOT FOUND IN DATABASE:")
        output_lines.append("="*70)
        for p1, p2, err in errors:
            output_lines.append(f"  {p1} vs {p2}: {err}")
    
    # Summary
    output_lines.append("\n" + "="*70)
    output_lines.append("SUMMARY - ALL PICKS")
    output_lines.append("="*70)
    output_lines.append(f"{'Match':<45} {'ML Pick':<20} {'Spread':<15} {'Total':<10}")
    output_lines.append("-"*90)
    
    for r in results:
        pick = r['player1'] if r['p1_win_prob'] > 50 else r['player2']
        odds = r['p1_odds'] if r['p1_win_prob'] > 50 else r['p2_odds']
        match_str = f"{r['player1'][:18]} vs {r['player2'][:18]}"
        ml_str = f"{pick[:15]} ({odds:+d})"
        spread_str = f"{r['spread_favors'][:10]} -{r['spread_line']}"
        total_str = f"O/U {r['total']:.1f}"
        output_lines.append(f"{match_str:<45} {ml_str:<20} {spread_str:<15} {total_str:<10}")
    
    output_lines.append("="*70)
    
    # Save to file
    output_text = "\n".join(output_lines)
    
    filename = f"predictions_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(output_text)
    
    print(f"\n{'='*70}")
    print(f"PREDICTIONS SAVED TO: {filename}")
    print(f"{'='*70}")
    
    # Also print to console
    print("\n" + output_text)


if __name__ == "__main__":
    main()
