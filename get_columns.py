"""Get the full column definitions from Tennis Abstract"""

import requests
import re

headers = {'User-Agent': 'Mozilla/5.0'}
url = 'https://www.tennisabstract.com/cgi-bin/wplayer-classic.cgi?p=NaomiOsaka&f=v1'
r = requests.get(url, headers=headers, timeout=30)

# Find matchhead which defines column mappings
head_pattern = r'var\s+matchhead\s*=\s*\[([^\]]+)\]'
head_match = re.search(head_pattern, r.text)
if head_match:
    head_str = head_match.group(1)
    print('matchhead columns:')
    cols = re.findall(r'"([^"]+)"', head_str)
    for i, col in enumerate(cols):
        print(f'  {i:2}: {col}')

# Also look for how match object maps to these
print("\n\nLooking for match property assignments...")
prop_pattern = r'(mt|match)\.(opts|ofwon|oswon|pts|fwon|swon)\s*='
matches = re.findall(prop_pattern, r.text)
print(f"Found properties: {set(m[1] for m in matches)}")

# Find actual assignment code
assign_pattern = r'match\[matchhead\.indexOf\([^\)]+\)\]'
assigns = re.findall(assign_pattern, r.text)
print(f"\nAssignment patterns: {len(assigns)}")
for a in assigns[:10]:
    print(f"  {a}")
