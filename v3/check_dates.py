import json
d = json.load(open('player_data.json'))
p = d['players']['Aryna Sabalenka']

# Check dates in AO matches
print('Sabalenka Australian Open matches:')
for m in p['matches'][:10]:
    print(f"  {m['date']} | {m['round']:<5} | {m['tournament'][:20]}")

print("\n\nUnique dates in first 20 matches:")
dates = set(m['date'] for m in p['matches'][:20])
print(sorted(dates))
