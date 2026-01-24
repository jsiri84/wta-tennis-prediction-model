"""Quick test to examine Tennis Abstract data structure"""

import requests
import re

url = 'http://www.tennisabstract.com/cgi-bin/wplayer.cgi?p=ArynaSabalenka'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
r = requests.get(url, headers=headers, timeout=15)

print("Status:", r.status_code)
print("Length:", len(r.text))

# Find JavaScript variable assignments
vars_pattern = r"var\s+(\w+)\s*=\s*'([^']+)';"
matches = re.findall(vars_pattern, r.text)

print("\nKey variables found:")
for name, value in matches:
    keywords = ['elo', 'rank', 'hard', 'clay', 'grass', 'surface', 'serve', 'return', 'name', 'country']
    if any(x in name.lower() for x in keywords):
        print(f"  {name} = {value}")

# Also look for direct patterns
print("\nDirect Elo patterns:")
elo_pattern = r"var\s+elo_\w+\s*=\s*'(\d+)';"
for match in re.finditer(elo_pattern, r.text):
    print(f"  Found: {match.group(0)}")

# Look for surface Elo data
print("\nLooking for surface-specific data...")
surface_patterns = [
    r"hard.*?(\d{4})",
    r"clay.*?(\d{4})",  
    r"grass.*?(\d{4})",
    r"hElo['\"]?\s*[:=]\s*['\"]?(\d{4})",
    r"cElo['\"]?\s*[:=]\s*['\"]?(\d{4})",
    r"gElo['\"]?\s*[:=]\s*['\"]?(\d{4})",
]

for pattern in surface_patterns:
    matches = re.findall(pattern, r.text, re.IGNORECASE)
    if matches:
        print(f"  Pattern '{pattern[:30]}...' found: {matches[:5]}")
