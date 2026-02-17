"""Check what fatigue-related data we have."""
import json

d = json.load(open('player_data.json'))
p = d['players']['Aryna Sabalenka']
m = p['matches'][0]

print('=== MATCH DATA STRUCTURE ===')
print('Keys:', list(m.keys()))
print()

print('=== SAMPLE MATCH ===')
print('  Date:', m.get('date'))
print('  Tournament:', m.get('tournament'))
print('  Round:', m.get('round'))
print('  Score:', m.get('score'))
print('  Time (mins):', m.get('time_mins'))
print()

print('=== LAST 10 MATCHES (Sabalenka) ===')
for match in p['matches'][:10]:
    date = match.get('date', '?')
    tourn = match.get('tournament', '?')[:25]
    rnd = match.get('round', '?')
    time = match.get('time_mins', '?')
    score = match.get('score', '?')
    print(f"  {date} | {tourn:<25} | {rnd:<5} | {time:>3} mins | {score}")

print()
print('=== FATIGUE DATA AVAILABILITY ===')

# Check what we can derive
has_dates = all(m.get('date') for m in p['matches'][:20])
has_time = sum(1 for m in p['matches'][:20] if m.get('time_mins'))
has_tournament = all(m.get('tournament') for m in p['matches'][:20])

print(f"  Dates available: {'Yes' if has_dates else 'No'}")
print(f"  Match time (mins): {has_time}/20 matches have it")
print(f"  Tournament names: {'Yes' if has_tournament else 'No'}")
print()

# What we CAN compute from this data:
print('=== WHAT WE CAN COMPUTE ===')
print('  [x] Matches in last 7 days (from dates)')
print('  [x] 3-set matches (from score parsing)')
print('  [x] Time on court (from time_mins)')
print('  [x] Back-to-back tournaments (from dates + tournament names)')
print('  [ ] Travel distance (would need tournament location data)')
