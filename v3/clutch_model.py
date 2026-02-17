"""
Clutch/Overperformance Model

Calculates how much a player over or underperforms relative to their point statistics.
- Overperformers: Win more than point stats suggest (may regress)
- Underperformers: Win less than point stats suggest (may improve)

Used to adjust ELO weight in predictions - reduce trust in ELO for overperformers
since their rating may be inflated by clutch/luck.
"""

import json
import numpy as np
import os

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(MODEL_DIR, 'player_data.json')
CLUTCH_PATH = os.path.join(MODEL_DIR, 'clutch_coefficients.json')


def calculate_expected_win_prob(spw, rpw):
    """
    Calculate expected win probability from serve/return points won.
    
    Based on tennis probability theory:
    - A player winning 50% of total points wins ~50% of matches
    - A player winning 52% of total points wins ~65% of matches
    - A player winning 55% of total points wins ~85% of matches
    
    The relationship is non-linear due to game/set structure.
    """
    if spw is None or rpw is None or spw == 0 or rpw == 0:
        return None
    
    # Total points won percentage
    total_pct = (spw + rpw) / 2
    
    # Non-linear transformation to match probability
    # Using logistic-style transformation centered at 0.5
    if total_pct <= 0.45:
        return 0.1
    elif total_pct >= 0.55:
        return 0.9
    else:
        # Map 0.45-0.55 to roughly 0.1-0.9
        edge = (total_pct - 0.5) / 0.05  # -1 to +1
        return 0.5 + edge * 0.4  # 0.1 to 0.9


def parse_tiebreaks(score, result):
    """Extract tiebreak wins/losses from a score string."""
    tb_won = 0
    tb_lost = 0
    
    if not score:
        return 0, 0
    
    for set_score in score.split():
        if '(' in set_score:  # Tiebreak indicator
            try:
                # Get the games before tiebreak
                games = set_score.split('(')[0]
                if '-' in games:
                    g1, g2 = games.split('-')
                    g1, g2 = int(g1), int(g2)
                else:
                    # Format like "7(5)" means won 7-6
                    g1 = int(games[0]) if len(games) > 0 else 0
                    g2 = 6  # Assume 6 for tiebreak
                
                # Determine if we won or lost this tiebreak
                if result == 'W':
                    if g1 > g2:
                        tb_won += 1
                    else:
                        tb_lost += 1
                else:  # Loss
                    if g1 > g2:
                        tb_lost += 1  # We won this set but lost match
                    else:
                        tb_won += 1   # We lost this set but it's a TB
            except:
                pass
    
    return tb_won, tb_lost


def calculate_clutch_coefficients(min_matches=10):
    """
    Calculate clutch/overperformance coefficients for all players.
    
    Returns dict: {player_name: {
        'overperformance': float (-0.2 to +0.2),
        'tiebreak_wr': float (0-1),
        'elo_adjustment': float (-0.1 to +0.1)
    }}
    """
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    
    coefficients = {}
    
    for name, p in data['players'].items():
        match_results = []
        tb_won = 0
        tb_lost = 0
        
        for m in p.get('matches', []):
            result = m.get('result', '')
            if result not in ('W', 'L'):
                continue
            
            # Get point stats
            serve_raw = m.get('serve_raw', {})
            return_raw = m.get('return_raw', {})
            
            pts = serve_raw.get('pts', 0)
            fwon = serve_raw.get('fwon', 0)
            swon = serve_raw.get('swon', 0)
            
            opts = return_raw.get('opts', 0)
            ofwon = return_raw.get('ofwon', 0)
            oswon = return_raw.get('oswon', 0)
            
            # Calculate SPW and RPW
            spw = (fwon + swon) / pts if pts > 0 else None
            rpw = 1 - (ofwon + oswon) / opts if opts > 0 else None
            
            if spw and rpw:
                expected = calculate_expected_win_prob(spw, rpw)
                actual = 1 if result == 'W' else 0
                if expected is not None:
                    match_results.append({
                        'expected': expected,
                        'actual': actual
                    })
            
            # Count tiebreaks
            tw, tl = parse_tiebreaks(m.get('score', ''), result)
            tb_won += tw
            tb_lost += tl
        
        # Need minimum matches for reliable estimate
        if len(match_results) >= min_matches:
            actual_wr = np.mean([m['actual'] for m in match_results])
            expected_wr = np.mean([m['expected'] for m in match_results])
            overperformance = actual_wr - expected_wr
            
            # Tiebreak win rate
            tb_total = tb_won + tb_lost
            tb_wr = tb_won / tb_total if tb_total >= 3 else 0.5
            
            # Calculate ELO adjustment
            # Overperformers: reduce ELO trust (their ELO may be inflated)
            # Underperformers: increase ELO trust (their ELO may be deflated)
            # Cap at ±0.05 (±5% adjustment to final probability)
            elo_adj = -overperformance * 0.3  # Scale factor
            elo_adj = max(-0.05, min(0.05, elo_adj))
            
            # Tiebreak adjustment (for close matches)
            # Good tiebreak players get small boost, bad ones get penalty
            tb_adj = (tb_wr - 0.5) * 0.02  # Max ±1% adjustment
            
            coefficients[name] = {
                'overperformance': round(overperformance, 4),
                'actual_wr': round(actual_wr, 4),
                'expected_wr': round(expected_wr, 4),
                'n_matches': len(match_results),
                'tb_won': tb_won,
                'tb_lost': tb_lost,
                'tb_wr': round(tb_wr, 4) if tb_total >= 3 else None,
                'elo_adjustment': round(elo_adj, 4),
                'tb_adjustment': round(tb_adj, 4),
            }
    
    return coefficients


def get_clutch_adjustment(player_name, coefficients=None):
    """
    Get the clutch-based probability adjustment for a player.
    
    Returns: adjustment to apply to win probability (negative for overperformers)
    """
    if coefficients is None:
        try:
            with open(CLUTCH_PATH, 'r') as f:
                coefficients = json.load(f)
        except:
            return 0.0
    
    if player_name not in coefficients:
        return 0.0
    
    coeff = coefficients[player_name]
    return coeff.get('elo_adjustment', 0.0)


def save_coefficients():
    """Calculate and save clutch coefficients."""
    coefficients = calculate_clutch_coefficients()
    
    with open(CLUTCH_PATH, 'w') as f:
        json.dump(coefficients, f, indent=2)
    
    print(f"Saved clutch coefficients for {len(coefficients)} players to {CLUTCH_PATH}")
    
    # Print summary
    overs = [c['overperformance'] for c in coefficients.values()]
    print(f"Average overperformance: {np.mean(overs)*100:+.2f}%")
    print(f"Std dev: {np.std(overs)*100:.2f}%")
    
    # Top overperformers
    sorted_players = sorted(coefficients.items(), 
                           key=lambda x: x[1]['overperformance'], reverse=True)
    
    print("\nTop 10 Overperformers (ELO may be inflated):")
    for name, c in sorted_players[:10]:
        print(f"  {name}: {c['overperformance']*100:+.1f}% over, adj={c['elo_adjustment']:+.3f}")
    
    print("\nTop 10 Underperformers (ELO may be deflated):")
    for name, c in sorted_players[-10:]:
        print(f"  {name}: {c['overperformance']*100:+.1f}% over, adj={c['elo_adjustment']:+.3f}")
    
    return coefficients


if __name__ == '__main__':
    save_coefficients()
