"""Fetch match stats directly from Tennis Abstract JS files - no Selenium needed"""

import requests
import re
import json


def fetch_player_matches(player_name: str, max_matches: int = 52) -> list[dict]:
    """Fetch match data directly from Tennis Abstract JS files"""
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    matches = []
    
    # Column mapping from Tennis Abstract (verified against BP Saved and RPW)
    cols = [
        "date", "tourn", "surf", "level", "wl", "rank", "seed", "entry", "round",
        "score", "max", "opp", "orank", "oseed", "oentry", "ohand", "obday",
        "oht", "ocountry", "oactive",
        # Serve stats (positions 20-29)
        "time", "aces", "dfs", "pts", "firsts", "fwon", "swon", 
        "col27",  # extra column (unknown)
        "saved", "chances",  # BP saved, BP faced (positions 28-29)
        # Opponent serve stats (positions 30-37)
        "oaces", "odfs", "opts", "ofirsts", "ofwon", "oswon",
        "osaved", "ochances",  # opponent BP saved, BP faced
        # Additional (positions 38+)
        "obackhand", "chartlink", "pslink", "whserver", "matchid", "col43"
    ]
    
    def parse_js_array(js_text: str) -> list:
        """Parse JS array handling apostrophes in strings like Queen's Club"""
        # Replace JS single quotes with double quotes, but preserve apostrophes in words
        # Pattern: replace 'value' delimiters but not word's apostrophes
        # Step 1: Replace [' with [" and '] with "]
        text = re.sub(r"\['", '["', js_text)
        text = re.sub(r"'\]", '"]', text)
        # Step 2: Replace ', ' with ", "
        text = re.sub(r"',\s*'", '", "', text)
        # Step 3: Replace remaining ', with ",
        text = re.sub(r"',\s*", '", ', text)
        # Step 4: Replace , ' with , "
        text = re.sub(r",\s*'", ', "', text)
        return json.loads(text)
    
    # Fetch career matches
    career_url = f'https://www.tennisabstract.com/jsmatches/{player_name}Career.js'
    r = requests.get(career_url, headers=headers, timeout=30)
    
    if r.status_code == 200:
        pattern = r'var\s+morematchmx\s*=\s*(\[\[[\s\S]*?\]\s*\]);'
        match = re.search(pattern, r.text)
        if match:
            raw = parse_js_array(match.group(1))
            
            for row in raw:
                m = {cols[i]: row[i] if i < len(row) else "" for i in range(len(cols))}
                matches.append(m)
    
    # Fetch recent matches (may have newer data)
    recent_url = f'https://www.tennisabstract.com/jsmatches/{player_name}.js'
    r2 = requests.get(recent_url, headers=headers, timeout=30)
    
    if r2.status_code == 200:
        pattern = r'var\s+matchmx\s*=\s*(\[\[[\s\S]*?\]\s*\]);'
        match = re.search(pattern, r2.text)
        if match:
            raw = parse_js_array(match.group(1))
            for row in raw:
                m = {cols[i]: row[i] if i < len(row) else "" for i in range(len(cols))}
                matches.insert(0, m)  # Recent matches first
    
    # Dedupe and sort by date desc
    seen = set()
    unique = []
    for m in matches:
        key = (m['date'], m['opp'], m['round'])
        if key not in seen and m['wl'] in ('W', 'L') and m['score']:
            seen.add(key)
            unique.append(m)
    
    unique.sort(key=lambda x: x['date'], reverse=True)
    return unique[:max_matches]


def safe_int(val):
    """Convert to int safely"""
    try:
        return int(val) if val else 0
    except:
        return 0

def safe_pct(num, denom):
    """Calculate percentage"""
    n, d = safe_int(num), safe_int(denom)
    return round(100 * n / d, 1) if d > 0 else None


def format_matches(matches: list[dict]) -> list[dict]:
    """Format matches with serve/return sections (both raw and calculated)"""
    
    # Column names for raw_js dump (verified against Tennis Abstract)
    col_names = [
        "date", "tourn", "surf", "level", "wl", "rank", "seed", "entry", "round",
        "score", "max", "opp", "orank", "oseed", "oentry", "ohand", "obday",
        "oht", "ocountry", "oactive",
        "time", "aces", "dfs", "pts", "firsts", "fwon", "swon", 
        "saved", "chances",
        "oaces", "odfs", "col31", "opts", "ofirsts", "ofwon", "oswon",
        "osaved", "ochances",
        "obackhand", "chartlink", "pslink", "whserver", "matchid", "col43"
    ]
    
    formatted = []
    for m in matches:
        # My serve raw values
        aces = safe_int(m.get('aces', 0))
        dfs = safe_int(m.get('dfs', 0))
        pts = safe_int(m.get('pts', 0))  # total service points
        firsts = safe_int(m.get('firsts', 0))  # first serves in
        fwon = safe_int(m.get('fwon', 0))  # first serve points won
        swon = safe_int(m.get('swon', 0))  # second serve points won
        saved = safe_int(m.get('saved', 0))  # break points saved
        chances = safe_int(m.get('chances', 0))  # break points faced
        second_serves = pts - firsts if pts > firsts else 0
        
        # Opponent serve raw values (for my return stats)
        oaces = safe_int(m.get('oaces', 0))
        odfs = safe_int(m.get('odfs', 0))
        opts = safe_int(m.get('opts', 0))  # opponent total service points
        ofirsts = safe_int(m.get('ofirsts', 0))  # opponent first serves in
        ofwon = safe_int(m.get('ofwon', 0))  # opponent first serve won
        oswon = safe_int(m.get('oswon', 0))  # opponent second serve won
        osaved = safe_int(m.get('osaved', 0))  # opponent bp saved
        ochances = safe_int(m.get('ochances', 0))  # opponent bp faced (my conversions)
        
        # Calculate return points won (RPW) - Tennis Abstract formula
        # RPW = 1 - (ofwon + oswon) / opts
        rpw_pct = None
        if opts > 0:
            rpw_pct = round((1 - (ofwon + oswon) / opts) * 100, 1)
        
        # Calculate opponent second serves
        o_second_serves = opts - ofirsts if opts > ofirsts else 0
        
        match = {
            "date": m.get('date', ''),
            "tournament": m.get('tourn', ''),
            "surface": m.get('surf', ''),
            "round": m.get('round', ''),
            "result": m.get('wl', ''),
            "score": m.get('score', ''),
            "rank": m.get('rank', ''),
            "opponent": m.get('opp', ''),
            "opp_rank": m.get('orank', ''),
            "time_mins": m.get('time', ''),
            "serve": {
                "ace_pct": safe_pct(aces, pts),
                "df_pct": safe_pct(dfs, second_serves),
                "first_in_pct": safe_pct(firsts, pts),
                "first_won_pct": safe_pct(fwon, firsts),
                "second_won_pct": safe_pct(swon, second_serves),
                "bp_saved_pct": safe_pct(saved, chances),  # saved / total BP faced
            },
            "serve_raw": {
                "aces": aces,
                "dfs": dfs,
                "pts": pts,
                "firsts": firsts,
                "fwon": fwon,
                "swon": swon,
                "saved": saved,
                "chances": chances,
            },
            "return": {
                "rpw_pct": rpw_pct,
                "v_ace_pct": safe_pct(oaces, opts) if opts else None,
                "v_first_won_pct": safe_pct(ofirsts - ofwon, ofirsts) if ofirsts else None,  # my return pts won on opp 1st
                "v_second_won_pct": safe_pct(o_second_serves - oswon, o_second_serves) if o_second_serves else None,
                "bp_conv_pct": safe_pct(ochances, osaved + ochances) if (osaved + ochances) else None,
            },
            "return_raw": {
                "opts": opts,
                "ofirsts": ofirsts,
                "ofwon": ofwon,
                "oswon": oswon,
                "osaved": osaved,
                "ochances": ochances,
                "oaces": oaces,
                "odfs": odfs,
            },
            "raw_js": {col_names[i]: m.get(col_names[i], "") for i in range(len(col_names))}
        }
        formatted.append(match)
    
    return formatted


if __name__ == '__main__':
    print('Fetching Naomi Osaka matches...')
    raw = fetch_player_matches('NaomiOsaka', 52)
    matches = format_matches(raw)
    
    with open('osaka_matches.json', 'w', encoding='utf-8') as f:
        json.dump(matches, f, indent=2)
    
    print(f'Saved {len(matches)} matches')
    print('\nSample:')
    print(json.dumps(matches[0], indent=2))
