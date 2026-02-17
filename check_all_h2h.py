"""Check H2H for all today's matches and factor into predictions"""
import json
from unified_model import UnifiedPredictor

data = json.load(open('all_players_matches.json'))

matches = [
    ('Clara Tauson', 'Mccartney Kessler', 'Hard', 'WTA 500 Abu Dhabi'),
    ('Sonay Kartal', 'Sara Bejlek', 'Hard', 'WTA 500 Abu Dhabi'),
    ('Alexandra Eala', 'Ekaterina Alexandrova', 'Hard', 'WTA 500 Abu Dhabi'),
    ('Liudmila Samsonova', 'Hailey Baptiste', 'Hard', 'WTA 500 Abu Dhabi'),
    ('Xiyu Wang', 'Oleksandra Oliynykova', 'Hard', 'WTA 250 Cluj'),
    ('Emma Raducanu', 'Maja Chwalinska', 'Hard', 'WTA 250 Cluj'),
    ('Anastasia Potapova', 'Sorana Cirstea', 'Hard', 'WTA 250 Cluj'),
    ('Yue Yuan', 'Daria Snigur', 'Hard', 'WTA 250 Cluj'),
    ('Katie Volynets', 'Alycia Parks', 'Hard', 'WTA 250 Ostrava'),
    ('Linda Fruhvirtova', 'Katie Boulter', 'Hard', 'WTA 250 Ostrava'),
    ('Diane Parry', 'Nikola Bartunkova', 'Hard', 'WTA 250 Ostrava'),
    ('Caty Mcnally', 'Tamara Korpatsch', 'Hard', 'WTA 250 Ostrava'),
]

# Known H2H records not in our dataset (manual overrides)
KNOWN_H2H = {
    ('Katie Volynets', 'Alycia Parks'): (0, 3),  # Parks 3-0 vs Volynets
}

def find_h2h(p1_name, p2_name):
    """Find H2H record from match data"""
    
    # Check for manual override first
    if (p1_name, p2_name) in KNOWN_H2H:
        p1_w, p2_w = KNOWN_H2H[(p1_name, p2_name)]
        return p1_w, p2_w, []
    if (p2_name, p1_name) in KNOWN_H2H:
        p2_w, p1_w = KNOWN_H2H[(p2_name, p1_name)]
        return p1_w, p2_w, []
    
    p1_wins = 0
    p2_wins = 0
    matches_found = []
    
    # Search in p1's matches
    p1_data = data['players'].get(p1_name, {})
    for m in p1_data.get('matches', []):
        opp = m.get('opponent', '').lower()
        p2_last = p2_name.split()[-1].lower()
        if p2_last in opp or p2_name.lower().replace(' ', '') in opp.replace(' ', ''):
            if m['result'] == 'W':
                p1_wins += 1
            else:
                p2_wins += 1
            matches_found.append((m['date'], m['result'], m['score']))
    
    # Search in p2's matches (to catch any we missed)
    p2_data = data['players'].get(p2_name, {})
    for m in p2_data.get('matches', []):
        opp = m.get('opponent', '').lower()
        p1_last = p1_name.split()[-1].lower()
        if p1_last in opp or p1_name.lower().replace(' ', '') in opp.replace(' ', ''):
            # Only count if not already found
            date = m['date']
            if not any(d == date for d, _, _ in matches_found):
                if m['result'] == 'W':
                    p2_wins += 1
                else:
                    p1_wins += 1
                matches_found.append((date, 'L' if m['result'] == 'W' else 'W', m['score']))
    
    return p1_wins, p2_wins, sorted(matches_found, reverse=True)

def adjust_prediction(model_prob, p1_h2h_wins, p2_h2h_wins):
    """Adjust model probability based on H2H"""
    total_h2h = p1_h2h_wins + p2_h2h_wins
    if total_h2h == 0:
        return model_prob, 0
    
    # H2H adjustment: each H2H match is worth ~3-5% adjustment
    h2h_diff = p1_h2h_wins - p2_h2h_wins
    adjustment = h2h_diff * 0.05  # 5% per H2H win difference
    
    # Cap adjustment at +/- 20%
    adjustment = max(-0.20, min(0.20, adjustment))
    
    adjusted = model_prob + adjustment
    adjusted = max(0.15, min(0.85, adjusted))  # Keep between 15-85%
    
    return adjusted, adjustment

p = UnifiedPredictor()
p.train()

print('='*80)
print('TODAY\'S PREDICTIONS WITH H2H ADJUSTMENT - January 24, 2026')
print('='*80)

current_tournament = ""
for p1, p2, surface, tournament in matches:
    if tournament != current_tournament:
        print(f"\n{'='*80}")
        print(f"{tournament}")
        print("=" * 80)
        current_tournament = tournament
    
    r = p.predict(p1, p2, surface)
    model_prob = r['p1_win_prob'] / 100
    
    # Get H2H
    p1_h2h, p2_h2h, h2h_matches = find_h2h(p1, p2)
    
    # Adjust probability
    adj_prob, adjustment = adjust_prediction(model_prob, p1_h2h, p2_h2h)
    
    # Determine pick
    if adj_prob > 0.5:
        pick = p1
        pick_prob = adj_prob
    else:
        pick = p2
        pick_prob = 1 - adj_prob
    
    # Convert to American odds
    if pick_prob >= 0.5:
        odds = int(-100 * pick_prob / (1 - pick_prob))
    else:
        odds = int(100 * (1 - pick_prob) / pick_prob)
    
    spread = abs(r['spread'])
    
    print(f"\n{p1} vs {p2}")
    
    # Show H2H if exists
    if p1_h2h + p2_h2h > 0:
        h2h_str = f"H2H: {p1.split()[-1]} {p1_h2h}-{p2_h2h} {p2.split()[-1]}"
        adj_str = f"(adj: {adjustment*100:+.0f}%)" if adjustment != 0 else ""
        print(f"  {h2h_str} {adj_str}")
    
    print(f"  Pick: {pick} ({odds:+d}) | Spread: -{spread:.1f} | Total: {r['total']:.1f}")
