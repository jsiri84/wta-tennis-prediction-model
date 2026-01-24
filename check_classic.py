"""Check the classic view pages for better stats data"""

import requests
from bs4 import BeautifulSoup
import re

headers = {'User-Agent': 'Mozilla/5.0'}

# Fetch serve stats (v1)
print("=== SERVE STATS (v1) ===")
url1 = 'https://www.tennisabstract.com/cgi-bin/wplayer-classic.cgi?p=NaomiOsaka&f=v1'
r1 = requests.get(url1, headers=headers, timeout=30)
print(f'Status: {r1.status_code}, Length: {len(r1.text)}')

soup1 = BeautifulSoup(r1.text, 'html.parser')
tables1 = soup1.find_all('table')
print(f'Found {len(tables1)} tables')

for i, table in enumerate(tables1):
    rows = table.find_all('tr')
    print(f'Table {i}: {len(rows)} rows')
    if len(rows) > 0:
        header_cells = rows[0].find_all(['th', 'td'])
        headers_text = [c.get_text(strip=True) for c in header_cells]
        print(f'\nTable {i}: {len(rows)} rows')
        print(f'Headers: {headers_text[:15]}')
        
        # Find Ruzic match
        for row in rows[1:10]:
            cells = row.find_all('td')
            text = ' '.join(c.get_text(strip=True) for c in cells[:8])
            if 'Ruzic' in text or '2026' in text:
                data = [c.get_text(strip=True) for c in cells]
                print(f'Match: {data}')
                break

# Look for JavaScript data - get the actual values
print("\n--- Looking for JS data ---")

# Find statparams
stat_pattern = r"var\s+statparams\s*=\s*['\"]([^'\"]+)['\"]"
stat_match = re.search(stat_pattern, r1.text)
if stat_match:
    print(f'statparams: {stat_match.group(1)}')

# Find matchhead (column headers)
head_pattern = r'var\s+matchhead\s*=\s*(\[[^\]]+\])'
head_match = re.search(head_pattern, r1.text)
if head_match:
    print(f'matchhead: {head_match.group(1)[:200]}')

# Find statrow definitions
statrow_pattern = r"var\s+statrow\s*=\s*\[([^\]]+)\]"
statrow_matches = re.findall(statrow_pattern, r1.text)
print(f'Found {len(statrow_matches)} statrow definitions')
for sr in statrow_matches[:3]:
    print(f'  statrow: {sr[:100]}')

print("\n\n=== RETURN STATS (v1r1) ===")
url2 = 'https://www.tennisabstract.com/cgi-bin/wplayer-classic.cgi?p=NaomiOsaka&f=v1r1'
r2 = requests.get(url2, headers=headers, timeout=30)
print(f'Status: {r2.status_code}, Length: {len(r2.text)}')

soup2 = BeautifulSoup(r2.text, 'html.parser')
tables2 = soup2.find_all('table')
print(f'Found {len(tables2)} tables')

for i, table in enumerate(tables2):
    rows = table.find_all('tr')
    if len(rows) > 5:
        header_cells = rows[0].find_all(['th', 'td'])
        headers_text = [c.get_text(strip=True) for c in header_cells]
        print(f'\nTable {i}: {len(rows)} rows')
        print(f'Headers: {headers_text[:15]}')
        
        # Find Ruzic match
        for row in rows[1:10]:
            cells = row.find_all('td')
            text = ' '.join(c.get_text(strip=True) for c in cells[:8])
            if 'Ruzic' in text or '2026' in text:
                data = [c.get_text(strip=True) for c in cells]
                print(f'Match: {data}')
                break
