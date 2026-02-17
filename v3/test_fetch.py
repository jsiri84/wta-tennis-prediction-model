import requests
import re

r = requests.get('https://www.tennisabstract.com/reports/wta_elo_ratings.html', timeout=30)

# Pattern: wplayer.cgi?p=SLUG">Name</a></td><td>Age</td><td>ELO
pattern = r'wplayer\.cgi\?p=([^"]+)"[^>]*>([^<]+)</a></td><td[^>]*>[\d.]+</td><td[^>]*>([\d.]+)'
matches = re.findall(pattern, r.text)

print(f'Found {len(matches)} matches')
for slug, name, elo in matches[:10]:
    name = name.replace('&nbsp;', ' ')
    print(f'{name}: {elo}')
