import json

data = json.load(open('all_players_matches.json'))

# Check Volynets matches vs Parks
volynets = data['players'].get('Katie Volynets', {})
parks = data['players'].get('Alycia Parks', {})

print('VOLYNETS vs PARKS H2H')
print('='*50)

print('\nRecent Volynets opponents:')
for m in volynets.get('matches', [])[:15]:
    print(f"  {m['date']}: vs {m['opponent']} - {m['result']} {m['score']}")

print('\nSearching for Parks in Volynets matches:')
for m in volynets.get('matches', []):
    opp = m.get('opponent', '').lower()
    if 'park' in opp or 'alycia' in opp:
        print(f"  FOUND: {m['date']}: {m['result']} {m['score']} vs {m['opponent']}")

print('\nSearching for Volynets in Parks matches:')
for m in parks.get('matches', []):
    opp = m.get('opponent', '').lower()
    if 'volyn' in opp or 'katie' in opp:
        print(f"  FOUND: {m['date']}: {m['result']} {m['score']} vs {m['opponent']}")

print('\n' + '='*50)
print('PLAYER ELO COMPARISON')
print('='*50)
print(f"\nVolynets: ELO {volynets.get('elo_overall', 'N/A')} | Hard {volynets.get('elo_hard', 'N/A')}")
print(f"Parks:    ELO {parks.get('elo_overall', 'N/A')} | Hard {parks.get('elo_hard', 'N/A')}")
