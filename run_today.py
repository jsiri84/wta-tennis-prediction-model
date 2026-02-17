"""Run today's matches through the model"""
from unified_model import UnifiedPredictor

matches = [
    # WTA 500 - Abu Dhabi (Outdoor Hard)
    ("Clara Tauson", "Mccartney Kessler", "Hard", "WTA 500 Abu Dhabi"),
    ("Sonay Kartal", "Sara Bejlek", "Hard", "WTA 500 Abu Dhabi"),
    ("Alexandra Eala", "Ekaterina Alexandrova", "Hard", "WTA 500 Abu Dhabi"),
    ("Liudmila Samsonova", "Hailey Baptiste", "Hard", "WTA 500 Abu Dhabi"),
    
    # WTA 250 - Cluj Napoca (Indoor Hard)
    ("Xiyu Wang", "Oleksandra Oliynykova", "Hard", "WTA 250 Cluj Napoca"),
    ("Emma Raducanu", "Maja Chwalinska", "Hard", "WTA 250 Cluj Napoca"),
    ("Anastasia Potapova", "Sorana Cirstea", "Hard", "WTA 250 Cluj Napoca"),
    ("Yue Yuan", "Daria Snigur", "Hard", "WTA 250 Cluj Napoca"),
    
    # WTA 250 - Ostrava (Indoor Hard)
    ("Katie Volynets", "Alycia Parks", "Hard", "WTA 250 Ostrava"),
    ("Linda Fruhvirtova", "Katie Boulter", "Hard", "WTA 250 Ostrava"),
    ("Diane Parry", "Nikola Bartunkova", "Hard", "WTA 250 Ostrava"),
    ("Caty Mcnally", "Tamara Korpatsch", "Hard", "WTA 250 Ostrava"),
]

p = UnifiedPredictor()
p.train()

print("=" * 80)
print("TODAY'S WTA PREDICTIONS - January 24, 2026")
print("=" * 80)

current_tournament = ""
for p1, p2, surface, tournament in matches:
    if tournament != current_tournament:
        print(f"\n{'='*80}")
        print(f"{tournament} ({surface})")
        print("=" * 80)
        current_tournament = tournament
    
    try:
        r = p.predict(p1, p2, surface)
        winner = p1 if r['p1_win_prob'] > 50 else p2
        prob = max(r['p1_win_prob'], r['p2_win_prob'])
        odds = r['p1_odds'] if r['p1_win_prob'] > 50 else r['p2_odds']
        spread = abs(r['spread'])
        print(f"\n{p1} vs {p2}")
        print(f"  Pick: {winner} ({odds:+d}) | Spread: -{spread:.1f} | Total: {r['total']:.1f}")
    except Exception as e:
        print(f"\n{p1} vs {p2}")
        print(f"  ERROR: {e}")
