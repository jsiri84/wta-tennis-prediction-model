"""
Fetch WTA Elo Ratings from Tennis Abstract
Extracts overall and surface-specific Elo for top 200 players
"""

import requests
from bs4 import BeautifulSoup
import json
import csv
from pathlib import Path
from datetime import datetime
import time


def fetch_wta_elo_ratings(top_n: int = 150) -> list:
    """
    Fetch WTA Elo ratings from Tennis Abstract
    Returns list of player dictionaries with all Elo ratings
    """
    url = 'https://tennisabstract.com/reports/wta_elo_ratings.html'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    print(f"Fetching WTA Elo ratings from Tennis Abstract...")
    
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the main data table (table with id 'reportable')
    table = soup.find('table', {'id': 'reportable'})
    if not table:
        # Fallback: find table with Elo headers
        tables = soup.find_all('table')
        for t in tables:
            headers = [th.get_text(strip=True) for th in t.find_all('th')]
            if 'Elo' in headers or 'hElo' in headers:
                table = t
                break
    
    if not table:
        print("Could not find Elo ratings table!")
        return []
    
    # Parse all rows
    all_rows = table.find_all('tr')
    print(f"Found {len(all_rows)} rows in table")
    
    # First row is header
    header_row = all_rows[0]
    headers = [th.get_text(strip=True).replace('\xa0', ' ') for th in header_row.find_all(['th', 'td'])]
    print(f"Found columns: {headers}")
    
    # Parse data rows (skip header)
    players = []
    
    for row in all_rows[1:top_n+1]:
        cells = row.find_all(['td', 'th'])
        if len(cells) < 4:
            continue
        
        # Parse each cell by position (known column order)
        # Columns: Elo Rank, Player, Age, Elo, '', hElo Rank, hElo, cElo Rank, cElo, gElo Rank, gElo, '', Peak Elo, Peak Month, '', WTA Rank
        try:
            cell_values = [c.get_text(strip=True).replace('\xa0', ' ') for c in cells]
            
            player_data = {
                'elo_rank': int(cell_values[0]) if cell_values[0].isdigit() else None,
                'name': cell_values[1],
                'age': float(cell_values[2]) if cell_values[2] else None,
                'elo_overall': round(float(cell_values[3])) if cell_values[3] else None,
                'hard_rank': int(cell_values[5]) if len(cell_values) > 5 and cell_values[5].isdigit() else None,
                'elo_hard': round(float(cell_values[6])) if len(cell_values) > 6 and cell_values[6] else None,
                'clay_rank': int(cell_values[7]) if len(cell_values) > 7 and cell_values[7].isdigit() else None,
                'elo_clay': round(float(cell_values[8])) if len(cell_values) > 8 and cell_values[8] else None,
                'grass_rank': int(cell_values[9]) if len(cell_values) > 9 and cell_values[9].isdigit() else None,
                'elo_grass': round(float(cell_values[10])) if len(cell_values) > 10 and cell_values[10] else None,
                'peak_elo': round(float(cell_values[12])) if len(cell_values) > 12 and cell_values[12] else None,
                'wta_rank': int(cell_values[15]) if len(cell_values) > 15 and cell_values[15].isdigit() else None,
            }
            
            # Only add if we have name and overall Elo
            if player_data.get('name') and player_data.get('elo_overall'):
                players.append(player_data)
                
        except (ValueError, IndexError) as e:
            continue
    
    print(f"Parsed {len(players)} players")
    return players


def save_to_json(players: list, filepath: str = "v2/tennis_abstract_elo.json"):
    """Save to JSON file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'source': 'Tennis Abstract',
        'url': 'https://tennisabstract.com/reports/wta_elo_ratings.html',
        'fetched_at': datetime.now().isoformat(),
        'player_count': len(players),
        'players': players
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved to {filepath}")
    return filepath


def save_to_csv(players: list, filepath: str = "v2/tennis_abstract_elo.csv"):
    """Save to CSV file"""
    if not players:
        return
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Get all unique fields
    all_fields = set()
    for p in players:
        all_fields.update(p.keys())
    
    # Order fields logically
    field_order = ['elo_rank', 'name', 'age', 'elo_overall', 'hard_rank', 'elo_hard', 
                   'clay_rank', 'elo_clay', 'grass_rank', 'elo_grass']
    fields = [f for f in field_order if f in all_fields]
    fields.extend([f for f in all_fields if f not in fields])
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(players)
    
    print(f"Saved to {filepath}")
    return filepath


def display_top_players(players: list, n: int = 20):
    """Display top N players by overall Elo"""
    print(f"\n{'='*80}")
    print(f"TOP {n} WTA PLAYERS BY OVERALL ELO")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Name':<25} {'Overall':<8} {'Hard':<8} {'Clay':<8} {'Grass':<8}")
    print(f"{'-'*80}")
    
    for i, p in enumerate(players[:n], 1):
        name = p.get('name', 'Unknown')[:24]
        overall = p.get('elo_overall', '-')
        hard = p.get('elo_hard', '-')
        clay = p.get('elo_clay', '-')
        grass = p.get('elo_grass', '-')
        
        print(f"{i:<6} {name:<25} {overall:<8} {hard:<8} {clay:<8} {grass:<8}")


def display_hard_court_rankings(players: list, n: int = 20):
    """Display top N players by hard court Elo"""
    # Sort by hard court Elo
    sorted_players = sorted(
        [p for p in players if p.get('elo_hard')],
        key=lambda x: x.get('elo_hard', 0),
        reverse=True
    )
    
    print(f"\n{'='*80}")
    print(f"TOP {n} WTA PLAYERS BY HARD COURT ELO (Australian Open)")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Name':<25} {'Hard Elo':<10} {'Overall':<10} {'Diff':<8}")
    print(f"{'-'*80}")
    
    for i, p in enumerate(sorted_players[:n], 1):
        name = p.get('name', 'Unknown')[:24]
        hard = p.get('elo_hard', 0)
        overall = p.get('elo_overall', 0)
        diff = hard - overall if hard and overall else 0
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        
        print(f"{i:<6} {name:<25} {hard:<10} {overall:<10} {diff_str:<8}")


def main():
    """Main function to fetch and display WTA Elo data"""
    print("="*60)
    print("TENNIS ABSTRACT WTA ELO DATA FETCHER")
    print("="*60)
    
    # Fetch top 200 players
    players = fetch_wta_elo_ratings(top_n=200)
    
    if not players:
        print("\nFailed to fetch data!")
        return
    
    # Save to files
    save_to_json(players)
    save_to_csv(players)
    
    # Display results
    display_top_players(players, n=20)
    display_hard_court_rankings(players, n=20)
    
    print(f"\n{'='*60}")
    print(f"Successfully fetched {len(players)} players!")
    print(f"Data saved to:")
    print(f"  - v2/tennis_abstract_elo.json")
    print(f"  - v2/tennis_abstract_elo.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
