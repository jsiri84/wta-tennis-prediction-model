"""Test fatigue scenarios."""
from skill_model import SkillPredictor

model = SkillPredictor()
model.train()

# Test 1: No fatigue (fresh player)
print("="*60)
print("TEST 1: No tournament info (no fatigue calc)")
print("="*60)
r = model.predict('Aryna Sabalenka', 'Madison Keys', 'Hard')
print(f"Sabalenka win prob: {r['p1_win_prob']}%")
print(f"Fatigue P1: {r['p1_fatigue_penalty']}%, P2: {r['p2_fatigue_penalty']}%")

# Test 2: Mid-tournament fatigue
print("\n" + "="*60)
print("TEST 2: Australian Open R16 (after first week)")
print("="*60)
r = model.predict('Aryna Sabalenka', 'Madison Keys', 'Hard',
                  tournament='Australian Open', match_date='20260118')
print(f"Sabalenka win prob: {r['p1_win_prob']}%")
print(f"Fatigue P1: {r['p1_fatigue_penalty']}% ({r['p1_fatigue_details'].get('matches_last_7_days', 0)} matches)")
print(f"Fatigue P2: {r['p2_fatigue_penalty']}% ({r['p2_fatigue_details'].get('matches_last_7_days', 0)} matches)")

# Test 3: Back-to-back tournament with travel
print("\n" + "="*60)
print("TEST 3: Dubai right after Australian Open (long travel)")
print("="*60)
r = model.predict('Aryna Sabalenka', 'Madison Keys', 'Hard',
                  tournament='Dubai', match_date='20260128')
print(f"Sabalenka win prob: {r['p1_win_prob']}%")
print(f"P1 travel: {r['p1_fatigue_details'].get('travel_distance_km')} km from {r['p1_fatigue_details'].get('prev_tournament')}")
print(f"P2 travel: {r['p2_fatigue_details'].get('travel_distance_km')} km from {r['p2_fatigue_details'].get('prev_tournament')}")
print(f"Net fatigue adjustment: {r['fatigue_adjustment']}%")
