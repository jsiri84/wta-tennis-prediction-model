"""Fetch recent match history for a player from Tennis Abstract"""

import requests
import re
import json
from datetime import datetime

def fetch_player_matches(player_name: str, max_matches: int = 52) -> list[dict]:
    """
    Fetch recent match history for a player from Tennis Abstract.
    
    Args:
        player_name: Player name formatted for URL (e.g., "NaomiOsaka")
        max_matches: Maximum number of recent matches to return (default 52)
    
    Returns:
        List of match dictionaries with detailed information
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    # Column definitions based on Tennis Abstract's matchhead
    columns = [
        "date", "tourn", "surf", "level", "wl", "rank", "seed", "entry", "round",
        "score", "sets", "opp", "orank", "oseed", "oentry", "ohand", "obday",
        "oht", "ocountry", "oactive",
        # Extended stats (when available)
        "time",      # 20 - Match time in minutes
        "aces",      # 21 - Aces
        "dfs",       # 22 - Double faults
        "svpt",      # 23 - Service points total
        "1stIn",     # 24 - First serves in
        "1stWon",    # 25 - First serve points won
        "2ndWon",    # 26 - Second serve points won
        "bpSaved",   # 27 - Break points saved
        "bpFaced",   # 28 - Break points faced
        # Opponent stats
        "o_aces",    # 29
        "o_dfs",     # 30
        "o_svpt",    # 31
        "o_1stIn",   # 32
        "o_1stWon",  # 33
        "o_2ndWon",  # 34
        "o_bpSaved", # 35
        "o_bpFaced", # 36
        # Return stats
        "rpw",       # 37 - Return points won
        "rptot",     # 38 - Return points total
        "o_rpw",     # 39
        "o_rptot",   # 40
        "col41",     # 41
        "col42",     # 42
        "match_id",  # 43 - Match identifier
    ]
    
    matches = []
    
    # Try to fetch the Career JS file first (contains all matches)
    career_url = f'https://www.tennisabstract.com/jsmatches/{player_name}Career.js'
    
    try:
        r = requests.get(career_url, headers=headers, timeout=30)
        if r.status_code == 200:
            # Parse morematchmx array - it ends with ]]; potentially with whitespace
            pattern = r'var\s+morematchmx\s*=\s*(\[\[[\s\S]*?\]\s*\]);'
            match = re.search(pattern, r.text)
            
            if match:
                # Parse the JavaScript array
                js_array = match.group(1)
                # Convert to valid JSON
                json_str = js_array.replace("'", '"')
                raw_matches = json.loads(json_str)
                
                # Also try to get recent matches from main file
                main_url = f'https://www.tennisabstract.com/jsmatches/{player_name}.js'
                r2 = requests.get(main_url, headers=headers, timeout=30)
                
                if r2.status_code == 200:
                    # Look for matchmx (recent matches not in career file yet)
                    pattern2 = r'var\s+matchmx\s*=\s*(\[\[[\s\S]*?\]\s*\]);'
                    match2 = re.search(pattern2, r2.text)
                    if match2:
                        recent_json = match2.group(1).replace("'", '"')
                        recent_matches = json.loads(recent_json)
                        # Prepend recent matches
                        raw_matches = recent_matches + raw_matches
                
                # Convert to dictionaries and sort by date (most recent first)
                for raw in raw_matches:
                    match_dict = {}
                    for i, col in enumerate(columns):
                        if i < len(raw):
                            match_dict[col] = raw[i]
                        else:
                            match_dict[col] = ""
                    
                    # Store the raw array for any additional data
                    match_dict['_raw'] = raw
                    matches.append(match_dict)
                
                # Sort by date descending (most recent first)
                matches.sort(key=lambda x: x.get('date', ''), reverse=True)
                
                # Remove duplicates (same date + opponent + round)
                # Also filter out upcoming/unplayed matches (wl = 'U' or empty score)
                seen = set()
                unique_matches = []
                for m in matches:
                    key = (m.get('date'), m.get('opp'), m.get('round'))
                    wl = m.get('wl', '')
                    score = m.get('score', '')
                    
                    # Skip if already seen or if match hasn't been played
                    if key in seen:
                        continue
                    if wl not in ('W', 'L'):
                        continue
                    if not score:
                        continue
                        
                    seen.add(key)
                    unique_matches.append(m)
                
                matches = unique_matches[:max_matches]
                
    except Exception as e:
        print(f"Error fetching matches: {e}")
        return []
    
    return matches


def format_date(date_str: str) -> str:
    """Convert YYYYMMDD to readable format"""
    if len(date_str) == 8:
        try:
            dt = datetime.strptime(date_str, "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        except:
            pass
    return date_str




def print_matches(matches: list[dict], player_name: str):
    """Print matches in a formatted table"""
    print(f"\n{'='*100}")
    print(f"Last {len(matches)} matches for {player_name}")
    print(f"{'='*100}")
    
    print(f"\n{'Date':<12} {'W/L':<4} {'Surface':<6} {'Tournament':<25} {'Round':<6} {'Opponent':<25} {'Score':<15}")
    print("-" * 100)
    
    for m in matches:
        date = format_date(m.get('date', ''))
        wl = m.get('wl', '')
        surface = m.get('surf', '')[:6]
        tourn = m.get('tourn', '')[:24]
        round_ = m.get('round', '')
        opp = m.get('opp', '')[:24]
        score = m.get('score', '')[:14]
        
        print(f"{date:<12} {wl:<4} {surface:<6} {tourn:<25} {round_:<6} {opp:<25} {score:<15}")


if __name__ == "__main__":
    # Test with Naomi Osaka
    player = "NaomiOsaka"
    print(f"Fetching match history for {player}...")
    
    matches = fetch_player_matches(player, max_matches=52)
    
    if matches:
        print(f"\nSuccessfully fetched {len(matches)} matches!")
        print_matches(matches, player)
        
        # Show win/loss record
        wins = sum(1 for m in matches if m.get('wl') == 'W')
        losses = sum(1 for m in matches if m.get('wl') == 'L')
        print(f"\nRecord in last {len(matches)} matches: {wins}W - {losses}L")
        
        # Surface breakdown
        surfaces = {}
        for m in matches:
            surf = m.get('surf', 'Unknown')
            if surf not in surfaces:
                surfaces[surf] = {'W': 0, 'L': 0}
            wl = m.get('wl', '')
            if wl in surfaces[surf]:
                surfaces[surf][wl] += 1
        
        print("\nBy surface:")
        for surf, record in surfaces.items():
            print(f"  {surf}: {record['W']}W - {record['L']}L")
    else:
        print("No matches found!")
