"""Print the Tennis Abstract dataset in readable format"""

import json

with open('v2/tennis_abstract_elo.json', 'r') as f:
    data = json.load(f)

print("=" * 120)
print("TENNIS ABSTRACT WTA ELO RATINGS - TOP 150 PLAYERS")
print(f"Source: {data['source']}")
print(f"Fetched: {data['fetched_at']}")
print("=" * 120)
print()

# Header
header = f"{'Rank':<5} {'Name':<28} {'Age':<6} {'Overall':<9} {'Hard':<9} {'Clay':<9} {'Grass':<9} {'Peak':<9} {'WTA':<5}"
print(header)
print("-" * 120)

for p in data['players']:
    rank = str(p.get('elo_rank', '-'))
    name = (p.get('name', 'Unknown'))[:27]
    
    age = p.get('age', '-')
    if isinstance(age, float):
        age = f"{age:.1f}"
    else:
        age = str(age) if age else "-"
    
    overall = str(p.get('elo_overall', '-'))
    hard = str(p.get('elo_hard', '-'))
    clay = str(p.get('elo_clay', '-'))
    grass = str(p.get('elo_grass', '-'))
    peak = str(p.get('peak_elo', '-'))
    wta = str(p.get('wta_rank', '-'))
    
    print(f"{rank:<5} {name:<28} {age:<6} {overall:<9} {hard:<9} {clay:<9} {grass:<9} {peak:<9} {wta:<5}")

print("-" * 120)
print(f"Total Players: {len(data['players'])}")
