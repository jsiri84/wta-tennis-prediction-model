"""Fetch match stats for WTA players from Tennis Abstract (threaded)"""

import json
import re
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
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


def fetch_single_player(player: dict, max_matches: int) -> tuple[str, dict | None, str | None]:
    """Fetch matches for a single player. Returns (name, data, error)"""
    name = player['name']
    url_key = name_to_url_key(name)
    
    try:
        # Fetch raw matches
        raw_matches = fetch_player_matches(url_key, max_matches)
        
        if not raw_matches:
            return (name, None, "No matches found")
        
        # Format and filter
        formatted = format_matches(raw_matches)
        valid_matches = [m for m in formatted if has_valid_stats(m)]
        
        # Remove raw_js from final output to save space
        for m in valid_matches:
            if 'raw_js' in m:
                del m['raw_js']
        
        player_data = {
            "elo_rank": player['elo_rank'],
            "elo_overall": player['elo_overall'],
            "elo_hard": player['elo_hard'],
            "elo_clay": player['elo_clay'],
            "elo_grass": player['elo_grass'],
            "wta_rank": player['wta_rank'],
            "matches": valid_matches,
            "match_count": len(valid_matches)
        }
        
        return (name, player_data, None)
        
    except Exception as e:
        return (name, None, str(e))


def fetch_all_players(num_players: int = 200, max_matches_per_player: int = 52, num_threads: int = 10):
    """Fetch matches for top N players in the Elo dataset using threading"""
    
    # Load player list
    with open('tennis_abstract_elo.json', 'r') as f:
        elo_data = json.load(f)
    
    all_players = elo_data['players']
    players = all_players[:num_players]
    
    print(f"Fetching top {len(players)} players (of {len(all_players)} available)")
    print(f"Using {num_threads} threads, {max_matches_per_player} matches per player")
    print()
    
    all_data = {
        "fetched_at": datetime.now().isoformat(),
        "player_count": 0,
        "total_matches": 0,
        "players": {}
    }
    
    failed = []
    completed = 0
    lock = Lock()
    
    def progress_callback(name: str, match_count: int | None, error: str | None):
        nonlocal completed
        with lock:
            completed += 1
            if error:
                status = f"ERROR: {error}"
            elif match_count is not None:
                status = f"{match_count} valid matches"
            else:
                status = "No matches"
            print(f"[{completed}/{len(players)}] {name}... {status}")
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        futures = {
            executor.submit(fetch_single_player, player, max_matches_per_player): player
            for player in players
        }
        
        # Process results as they complete
        for future in as_completed(futures):
            name, player_data, error = future.result()
            
            if error:
                failed.append(name)
                progress_callback(name, None, error)
            elif player_data:
                all_data['players'][name] = player_data
                all_data['player_count'] += 1
                all_data['total_matches'] += player_data['match_count']
                progress_callback(name, player_data['match_count'], None)
            else:
                failed.append(name)
                progress_callback(name, None, "No data")
    
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
    
    return all_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch WTA player match data from Tennis Abstract')
    parser.add_argument('-n', '--num-players', type=int, default=200,
                        help='Number of top players to fetch (default: 200)')
    parser.add_argument('-m', '--max-matches', type=int, default=52,
                        help='Max matches per player (default: 52)')
    parser.add_argument('-t', '--threads', type=int, default=10,
                        help='Number of threads (default: 10)')
    
    args = parser.parse_args()
    
    fetch_all_players(
        num_players=args.num_players,
        max_matches_per_player=args.max_matches,
        num_threads=args.threads
    )
