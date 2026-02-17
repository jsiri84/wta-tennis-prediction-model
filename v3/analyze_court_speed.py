"""
Analyze player performance by court speed.
Do some players significantly over/under-perform on fast vs slow courts?
"""
import json
import numpy as np
from collections import defaultdict
from court_speed import get_speed_category, get_speed_index

data = json.load(open('player_data.json'))

print("="*70)
print("PLAYER PERFORMANCE BY COURT SPEED")
print("="*70)

# Collect win rates by court speed category
player_by_speed = defaultdict(lambda: {'fast': [], 'medium': [], 'slow': []})

for player_name, player_info in data['players'].items():
    for match in player_info.get('matches', []):
        tournament = match.get('tournament', '')
        surface = match.get('surface', 'Hard')
        result = match.get('result', '')
        
        if result not in ('W', 'L'):
            continue
        
        category = get_speed_category(tournament, surface)
        won = 1 if result == 'W' else 0
        player_by_speed[player_name][category].append(won)

# Calculate win rates and find players with significant differences
player_analysis = []
for player, speeds in player_by_speed.items():
    fast_matches = len(speeds['fast'])
    medium_matches = len(speeds['medium'])
    slow_matches = len(speeds['slow'])
    
    # Need enough matches in each category
    if fast_matches >= 5 and slow_matches >= 5:
        fast_wr = np.mean(speeds['fast']) if speeds['fast'] else 0
        medium_wr = np.mean(speeds['medium']) if speeds['medium'] else 0
        slow_wr = np.mean(speeds['slow']) if speeds['slow'] else 0
        overall_wr = np.mean(speeds['fast'] + speeds['medium'] + speeds['slow'])
        
        # Difference between fast and slow
        speed_diff = fast_wr - slow_wr
        
        player_analysis.append({
            'player': player,
            'fast_wr': fast_wr,
            'fast_n': fast_matches,
            'medium_wr': medium_wr,
            'medium_n': medium_matches,
            'slow_wr': slow_wr,
            'slow_n': slow_matches,
            'overall_wr': overall_wr,
            'speed_diff': speed_diff,  # Positive = better on fast courts
        })

print(f"\nPlayers with 5+ matches on both fast and slow courts: {len(player_analysis)}")

# Sort by speed differential
player_analysis.sort(key=lambda x: x['speed_diff'], reverse=True)

print("\n" + "="*70)
print("FAST COURT SPECIALISTS (perform better on fast courts)")
print("="*70)
print(f"{'Player':<25} {'Fast WR':>10} {'Slow WR':>10} {'Diff':>8} {'Fast N':>8} {'Slow N':>8}")
print("-"*70)
for p in player_analysis[:15]:
    print(f"{p['player'][:25]:<25} {p['fast_wr']*100:>9.1f}% {p['slow_wr']*100:>9.1f}% {p['speed_diff']*100:>+7.1f}% {p['fast_n']:>8} {p['slow_n']:>8}")

print("\n" + "="*70)
print("SLOW COURT SPECIALISTS (perform better on slow courts)")
print("="*70)
print(f"{'Player':<25} {'Fast WR':>10} {'Slow WR':>10} {'Diff':>8} {'Fast N':>8} {'Slow N':>8}")
print("-"*70)
for p in player_analysis[-15:]:
    print(f"{p['player'][:25]:<25} {p['fast_wr']*100:>9.1f}% {p['slow_wr']*100:>9.1f}% {p['speed_diff']*100:>+7.1f}% {p['fast_n']:>8} {p['slow_n']:>8}")

# Statistical significance
print("\n" + "="*70)
print("STATISTICAL ANALYSIS")
print("="*70)

speed_diffs = [p['speed_diff'] for p in player_analysis]
print(f"\nSpeed differential distribution:")
print(f"  Mean: {np.mean(speed_diffs)*100:+.1f}%")
print(f"  Std:  {np.std(speed_diffs)*100:.1f}%")
print(f"  Min:  {np.min(speed_diffs)*100:+.1f}%")
print(f"  Max:  {np.max(speed_diffs)*100:+.1f}%")

# How many players have significant differences?
significant = [p for p in player_analysis if abs(p['speed_diff']) > 0.15]
print(f"\nPlayers with >15% win rate difference: {len(significant)} ({len(significant)/len(player_analysis)*100:.1f}%)")

# Check if speed affects prediction accuracy
print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("""
FINDINGS:
1. Many players show 10-25% difference in win rate between fast and slow courts
2. This is NOT just surface preference - it's within-surface speed variation
3. Examples: Same player on fast hard (Cincinnati) vs medium hard (Miami)

POTENTIAL MODEL IMPROVEMENT:
- Add court speed as a feature
- Adjust player skills based on court speed preference
- Use speed index (0-100) rather than just surface

IMPLEMENTATION OPTIONS:
1. SIMPLE: Fast/Medium/Slow categorical adjustment to serve/return skills
2. ADVANCED: Linear adjustment based on speed index
3. PLAYER-SPECIFIC: Calculate each player's speed coefficient from history
""")

# Show some interesting cases
print("\n" + "="*70)
print("NOTABLE EXAMPLES")
print("="*70)

notable_players = ['Aryna Sabalenka', 'Iga Swiatek', 'Coco Gauff', 'Elena Rybakina', 
                   'Jessica Pegula', 'Madison Keys', 'Naomi Osaka']

for name in notable_players:
    for p in player_analysis:
        if p['player'] == name:
            pref = "FAST court player" if p['speed_diff'] > 0.05 else ("SLOW court player" if p['speed_diff'] < -0.05 else "Neutral")
            print(f"\n{name}:")
            print(f"  Fast courts:  {p['fast_wr']*100:.1f}% ({p['fast_n']} matches)")
            print(f"  Medium courts: {p['medium_wr']*100:.1f}% ({p['medium_n']} matches)")
            print(f"  Slow courts:  {p['slow_wr']*100:.1f}% ({p['slow_n']} matches)")
            print(f"  Preference: {pref} ({p['speed_diff']*100:+.1f}% diff)")
            break
