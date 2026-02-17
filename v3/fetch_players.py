"""
WTA Player Data Fetcher
Fetches player ELO ratings and match-level serve/return stats from Tennis Abstract.
"""

import requests
import re
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import os

BASE_URL = "https://www.tennisabstract.com/cgi-bin/wplayer-classic.cgi?p="
RANKINGS_URL = "https://www.tennisabstract.com/reports/wta_elo_ratings.html"
JS_MATCHES_URL = "https://www.tennisabstract.com/jsmatches/"

HEADERS = {'User-Agent': 'Mozilla/5.0'}


def get_top_players(n: int = 200) -> list:
    """Fetch top N players from ELO rankings with surface-specific ELOs."""
    try:
        resp = requests.get(RANKINGS_URL, headers=HEADERS, timeout=30)
        
        # Pattern matches full row with surface ELOs:
        # wplayer.cgi?p=SLUG">Name</a></td><td>Age</td><td>Overall</td><td></td><td>HardRank</td><td>HardElo</td>...
        # Columns: name, age, overall_elo, empty, hard_rank, hard_elo, clay_rank, clay_elo, grass_rank, grass_elo
        pattern = (
            r'wplayer\.cgi\?p=([^"]+)"[^>]*>([^<]+)</a></td>'
            r'<td[^>]*>[\d.]+</td>'  # Age
            r'<td[^>]*>([\d.]+)</td>'  # Overall ELO
            r'<td[^>]*></td>'  # Empty
            r'<td[^>]*>\d*</td>'  # Hard rank
            r'<td[^>]*>([\d.]+)</td>'  # Hard ELO
            r'<td[^>]*>\d*</td>'  # Clay rank  
            r'<td[^>]*>([\d.]+)</td>'  # Clay ELO
            r'<td[^>]*>\d*</td>'  # Grass rank
            r'<td[^>]*>([\d.]+)</td>'  # Grass ELO
        )
        matches = re.findall(pattern, resp.text)
        
        players = []
        for match in matches[:n]:
            slug, name, overall, hard, clay, grass = match
            name = name.replace('&nbsp;', ' ').strip()
            players.append({
                'slug': slug,
                'name': name,
                'elo': int(float(overall)),
                'elo_hard': int(float(hard)),
                'elo_clay': int(float(clay)),
                'elo_grass': int(float(grass)),
            })
        return players
    except Exception as e:
        print(f"Error fetching rankings: {e}")
        return []


def safe_int(val):
    """Convert to int safely."""
    try:
        return int(val) if val else 0
    except:
        return 0


def safe_pct(num, denom):
    """Calculate percentage safely."""
    n, d = safe_int(num), safe_int(denom)
    return round(100 * n / d, 1) if d > 0 else None


def parse_js_array(js_text: str) -> list:
    """Parse JS array handling apostrophes in strings like Queen's Club."""
    text = re.sub(r"\['", '["', js_text)
    text = re.sub(r"'\]", '"]', text)
    text = re.sub(r"',\s*'", '", "', text)
    text = re.sub(r"',\s*", '", ', text)
    text = re.sub(r",\s*'", ', "', text)
    return json.loads(text)


def fetch_player_hand(slug: str) -> str:
    """Fetch player handedness from JS file. Returns 'R', 'L', or 'U' (unknown)."""
    try:
        url = f'{JS_MATCHES_URL}{slug}.js'
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            m = re.search(r"var\s+hand\s*=\s*['\"](\w)['\"]", r.text)
            if m:
                return m.group(1).upper()
    except:
        pass
    return 'U'  # Unknown


def fetch_matches_from_js(slug: str, max_matches: int = 52) -> list:
    """
    Fetch match data directly from Tennis Abstract JS files.
    This gives us detailed per-match serve/return stats.
    """
    cols = [
        "date", "tourn", "surf", "level", "wl", "rank", "seed", "entry", "round",
        "score", "max", "opp", "orank", "oseed", "oentry", "ohand", "obday",
        "oht", "ocountry", "oactive",
        # Serve stats
        "time", "aces", "dfs", "pts", "firsts", "fwon", "swon", 
        "col27", "saved", "chances",
        # Opponent serve stats
        "oaces", "odfs", "opts", "ofirsts", "ofwon", "oswon",
        "osaved", "ochances",
        # Additional
        "obackhand", "chartlink", "pslink", "whserver", "matchid", "col43"
    ]
    
    matches = []
    
    # Fetch career matches
    try:
        career_url = f'{JS_MATCHES_URL}{slug}Career.js'
        r = requests.get(career_url, headers=HEADERS, timeout=30)
        
        if r.status_code == 200:
            pattern = r'var\s+morematchmx\s*=\s*(\[\[[\s\S]*?\]\s*\]);'
            match = re.search(pattern, r.text)
            if match:
                raw = parse_js_array(match.group(1))
                for row in raw:
                    m = {cols[i]: row[i] if i < len(row) else "" for i in range(len(cols))}
                    matches.append(m)
    except Exception as e:
        pass  # Silently skip career file errors
    
    # Fetch recent matches (may have newer data)
    try:
        recent_url = f'{JS_MATCHES_URL}{slug}.js'
        r2 = requests.get(recent_url, headers=HEADERS, timeout=30)
        
        if r2.status_code == 200:
            pattern = r'var\s+matchmx\s*=\s*(\[\[[\s\S]*?\]\s*\]);'
            match = re.search(pattern, r2.text)
            if match:
                raw = parse_js_array(match.group(1))
                for row in raw:
                    m = {cols[i]: row[i] if i < len(row) else "" for i in range(len(cols))}
                    matches.insert(0, m)
    except Exception as e:
        pass  # Silently skip recent file errors
    
    # Dedupe and sort by date desc
    seen = set()
    unique = []
    for m in matches:
        key = (m.get('date', ''), m.get('opp', ''), m.get('round', ''))
        if key not in seen and m.get('wl') in ('W', 'L') and m.get('score'):
            seen.add(key)
            unique.append(m)
    
    unique.sort(key=lambda x: x.get('date', ''), reverse=True)
    return unique[:max_matches]


def format_match(m: dict) -> dict:
    """Format a raw JS match into our structure with serve/return stats."""
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
    o_second_serves = opts - ofirsts if opts > ofirsts else 0
    
    # Calculate return points won (RPW)
    rpw_pct = None
    if opts > 0:
        rpw_pct = round((1 - (ofwon + oswon) / opts) * 100, 1)
    
    # Clean surface
    surface = m.get('surf', 'Hard').strip().capitalize()
    if surface not in ['Hard', 'Clay', 'Grass']:
        surface = 'Hard'
    
    # Get opponent hand (R/L/U)
    ohand = m.get('ohand', '').upper()
    if ohand not in ('R', 'L'):
        ohand = 'U'
    
    return {
        "date": m.get('date', ''),
        "tournament": m.get('tourn', ''),
        "surface": surface,
        "round": m.get('round', ''),
        "result": m.get('wl', ''),
        "score": m.get('score', ''),
        "rank": m.get('rank', ''),
        "opponent": m.get('opp', ''),
        "opp_rank": m.get('orank', ''),
        "opp_hand": ohand,  # Opponent handedness
        "time_mins": m.get('time', ''),
        "serve": {
            "ace_pct": safe_pct(aces, pts),
            "df_pct": safe_pct(dfs, second_serves) if second_serves > 0 else safe_pct(dfs, pts),
            "first_in_pct": safe_pct(firsts, pts),
            "first_won_pct": safe_pct(fwon, firsts),
            "second_won_pct": safe_pct(swon, second_serves),
            "bp_saved_pct": safe_pct(saved, chances),
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
            "v_first_won_pct": safe_pct(ofirsts - ofwon, ofirsts) if ofirsts else None,
            "v_second_won_pct": safe_pct(o_second_serves - oswon, o_second_serves) if o_second_serves else None,
            "bp_conv_pct": safe_pct(ochances - osaved, ochances) if ochances else None,
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
    }


def fetch_player(player: dict, max_matches: int = 52) -> dict:
    """Fetch full data for a single player."""
    try:
        # Get matches from JS files (detailed stats)
        raw_matches = fetch_matches_from_js(player['slug'], max_matches)
        matches = [format_match(m) for m in raw_matches]
        
        # Get handedness
        hand = fetch_player_hand(player['slug'])
        
        # Use surface ELOs from rankings page (already parsed in get_top_players)
        return {
            'name': player['name'],
            'slug': player['slug'],
            'hand': hand,  # 'R' = right, 'L' = left, 'U' = unknown
            'elo_overall': player['elo'],
            'elo_hard': player.get('elo_hard', player['elo']),
            'elo_clay': player.get('elo_clay', player['elo']),
            'elo_grass': player.get('elo_grass', player['elo']),
            'matches': matches,
        }
        
    except Exception as e:
        print(f"  Error fetching {player['name']}: {e}")
        return None


def fetch_all_players(num_players: int = 200, max_matches: int = 52, num_threads: int = 10):
    """Fetch data for top N players using multithreading."""
    
    print(f"Fetching top {num_players} players...")
    players = get_top_players(num_players)
    
    if not players:
        print("No players found!")
        return
    
    print(f"Found {len(players)} players. Fetching details with {num_threads} threads...")
    
    results = {}
    completed = 0
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(fetch_player, p, max_matches): p 
            for p in players
        }
        
        for future in as_completed(futures):
            player = futures[future]
            completed += 1
            
            try:
                data = future.result()
                if data and data.get('matches'):
                    results[data['name']] = data
                    n_matches = len(data['matches'])
                    # Count matches with actual serve stats
                    n_with_stats = sum(1 for m in data['matches'] if m.get('serve', {}).get('first_in_pct'))
                    hand_str = {'R': 'R', 'L': 'L', 'U': '?'}.get(data.get('hand', 'U'), '?')
                    print(f"[{completed}/{len(players)}] {data['name']} ({hand_str}): {n_matches} matches ({n_with_stats} with stats)")
                else:
                    print(f"[{completed}/{len(players)}] {player['name']}: No data")
            except Exception as e:
                print(f"[{completed}/{len(players)}] {player['name']}: Error - {e}")
    
    # Save to file
    output = {
        'fetched_at': datetime.now().isoformat(),
        'num_players': len(results),
        'players': results,
    }
    
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'player_data.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Calculate stats
    total_matches = sum(len(p['matches']) for p in results.values())
    matches_with_stats = sum(
        sum(1 for m in p['matches'] if m.get('serve', {}).get('first_in_pct'))
        for p in results.values()
    )
    
    # Handedness stats
    lefties = [p['name'] for p in results.values() if p.get('hand') == 'L']
    righties = sum(1 for p in results.values() if p.get('hand') == 'R')
    unknown = sum(1 for p in results.values() if p.get('hand') == 'U')
    
    print(f"\nSaved {len(results)} players to {output_file}")
    print(f"Total matches: {total_matches}")
    print(f"Matches with serve stats: {matches_with_stats} ({100*matches_with_stats/total_matches:.1f}%)")
    print(f"\nHandedness: {righties} right, {len(lefties)} left, {unknown} unknown")
    if lefties:
        print(f"Left-handed players: {', '.join(lefties)}")
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch WTA player data')
    parser.add_argument('-n', '--num-players', type=int, default=200,
                        help='Number of top players to fetch (default: 200)')
    parser.add_argument('-m', '--max-matches', type=int, default=52,
                        help='Max matches per player (default: 52)')
    parser.add_argument('-t', '--threads', type=int, default=10,
                        help='Number of threads (default: 10)')
    
    args = parser.parse_args()
    fetch_all_players(args.num_players, args.max_matches, args.threads)
