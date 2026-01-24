"""Fetch match stats for all 200 WTA players from Tennis Abstract"""

import json
import time
import re
from datetime import datetime
from fetch_match_stats import fetch_player_matches, format_matches


def name_to_url_key(name: str) -> str:
    """Convert player name to Tennis Abstract URL format"""
    # Remove special characters and spaces
    # "Aryna Sabalenka" -> "ArynaSabalenka"
    # "Iga Swiatek" -> "IgaSwiatek"
    clean = re.sub(r'[^a-zA-Z]', '', name.replace(' ', ''))
    return clean


def has_valid_stats(match: dict) -> bool:
    """Check if match has valid stats (not walkover, not missing data)"""
    # Skip if no score
    if not match.get('score'):
        return False
    
    score = match.get('score', '').upper()
    
    # Filter out walkovers
    if 'W/O' in score or 'WO' in score or 'DEF' in score or 'RET' in score:
        return False
    
    # Filter out matches without serve stats
    serve_raw = match.get('serve_raw', {})
    if not serve_raw.get('pts') or serve_raw.get('pts') == 0:
        return False
    
    # Filter out United Cup and similar team events without stats
    tourn = match.get('tournament', '').lower()
    if 'united cup' in tourn and serve_raw.get('pts', 0) == 0:
        return False
    
    return True


def fetch_all_players(max_matches_per_player: int = 52):
    """Fetch matches for all players in the Elo dataset"""
    
    # Load player list
    with open('tennis_abstract_elo.json', 'r') as f:
        elo_data = json.load(f)
    
    players = elo_data['players']
    print(f"Found {len(players)} players to fetch")
    
    all_data = {
        "fetched_at": datetime.now().isoformat(),
        "player_count": 0,
        "total_matches": 0,
        "players": {}
    }
    
    failed = []
    
    for i, player in enumerate(players):
        name = player['name']
        url_key = name_to_url_key(name)
        
        print(f"[{i+1}/{len(players)}] Fetching {name}...", end=" ", flush=True)
        
        try:
            # Fetch raw matches
            raw_matches = fetch_player_matches(url_key, max_matches_per_player)
            
            if not raw_matches:
                print(f"No matches found")
                failed.append(name)
                continue
            
            # Format and filter
            formatted = format_matches(raw_matches)
            valid_matches = [m for m in formatted if has_valid_stats(m)]
            
            # Remove raw_js from final output to save space
            for m in valid_matches:
                if 'raw_js' in m:
                    del m['raw_js']
            
            all_data['players'][name] = {
                "elo_rank": player['elo_rank'],
                "elo_overall": player['elo_overall'],
                "elo_hard": player['elo_hard'],
                "elo_clay": player['elo_clay'],
                "elo_grass": player['elo_grass'],
                "wta_rank": player['wta_rank'],
                "matches": valid_matches,
                "match_count": len(valid_matches)
            }
            
            all_data['player_count'] += 1
            all_data['total_matches'] += len(valid_matches)
            
            print(f"{len(valid_matches)} valid matches")
            
            # Rate limit - be nice to the server
            time.sleep(0.5)
            
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append(name)
            continue
    
    # Save results
    output_file = 'all_players_matches.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Completed!")
    print(f"  Players fetched: {all_data['player_count']}")
    print(f"  Total matches: {all_data['total_matches']}")
    print(f"  Output: {output_file}")
    
    if failed:
        print(f"\nFailed players ({len(failed)}):")
        for name in failed:
            print(f"  - {name}")


if __name__ == '__main__':
    fetch_all_players(52)
