"""
Tennis Abstract Data Scraper
Fetches Elo ratings (overall + surface-specific) for WTA players
"""

import requests
from bs4 import BeautifulSoup
import re
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
import csv
from pathlib import Path


@dataclass
class PlayerElo:
    """Player Elo ratings from Tennis Abstract"""
    name: str
    wta_rank: Optional[int] = None
    
    # Elo ratings
    overall_elo: Optional[float] = None
    hard_elo: Optional[float] = None
    clay_elo: Optional[float] = None
    grass_elo: Optional[float] = None
    
    # Peak ratings
    peak_elo: Optional[float] = None
    peak_hard_elo: Optional[float] = None
    
    # Additional stats if available
    age: Optional[int] = None
    country: Optional[str] = None
    
    # Data source
    source: str = "tennis_abstract"
    fetched_at: Optional[str] = None


class TennisAbstractScraper:
    """
    Scraper for Tennis Abstract WTA player data
    
    Tennis Abstract URLs:
    - Player page: http://www.tennisabstract.com/cgi-bin/wplayer.cgi?p=PlayerName
    - Rankings: http://www.tennisabstract.com/reports/wta_elo_ratings.html
    """
    
    BASE_URL = "http://www.tennisabstract.com"
    PLAYER_URL = "http://www.tennisabstract.com/cgi-bin/wplayer.cgi?p={}"
    RANKINGS_URL = "http://www.tennisabstract.com/reports/wta_elo_ratings.html"
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    def __init__(self, cache_dir: str = "cache"):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _normalize_name(self, name: str) -> str:
        """Convert name to URL format (e.g., 'Aryna Sabalenka' -> 'ArynaSabalenka')"""
        # Remove accents/special chars and join
        return name.replace(" ", "").replace("-", "").replace("'", "")
    
    def _get_page(self, url: str, use_cache: bool = True) -> Optional[str]:
        """Fetch a page with caching"""
        cache_file = self.cache_dir / f"{hash(url)}.html"
        
        if use_cache and cache_file.exists():
            return cache_file.read_text(encoding='utf-8')
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            if use_cache:
                cache_file.write_text(response.text, encoding='utf-8')
            
            time.sleep(1)  # Rate limiting
            return response.text
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def fetch_player(self, name: str) -> Optional[PlayerElo]:
        """Fetch Elo data for a single player"""
        url_name = self._normalize_name(name)
        url = self.PLAYER_URL.format(url_name)
        
        html = self._get_page(url)
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        player = PlayerElo(name=name)
        
        # Parse Elo ratings from the page
        # Tennis Abstract displays Elo in tables/text
        text = soup.get_text()
        
        # Look for Elo patterns
        elo_patterns = {
            'overall_elo': r'(?:Overall\s+)?Elo[:\s]+(\d{3,4})',
            'hard_elo': r'Hard\s+(?:Court\s+)?Elo[:\s]+(\d{3,4})',
            'clay_elo': r'Clay\s+(?:Court\s+)?Elo[:\s]+(\d{3,4})',
            'grass_elo': r'Grass\s+(?:Court\s+)?Elo[:\s]+(\d{3,4})',
        }
        
        for field, pattern in elo_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                setattr(player, field, float(match.group(1)))
        
        return player
    
    def fetch_rankings_page(self) -> Optional[List[Dict]]:
        """
        Fetch the WTA Elo rankings page
        Returns list of player dictionaries
        """
        html = self._get_page(self.RANKINGS_URL)
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        players = []
        
        # Tennis Abstract typically has rankings in a table
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            for row in rows[1:]:  # Skip header
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 3:
                    try:
                        player_data = {
                            'rank': int(cells[0].get_text(strip=True)) if cells[0].get_text(strip=True).isdigit() else None,
                            'name': cells[1].get_text(strip=True),
                            'elo': float(cells[2].get_text(strip=True)) if cells[2].get_text(strip=True).replace('.','').isdigit() else None,
                        }
                        if player_data['name'] and player_data['elo']:
                            players.append(player_data)
                    except (ValueError, IndexError):
                        continue
        
        return players
    
    def save_to_csv(self, players: List[PlayerElo], filename: str = "wta_elo_data.csv"):
        """Save player data to CSV"""
        if not players:
            return
        
        filepath = Path(filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=asdict(players[0]).keys())
            writer.writeheader()
            for player in players:
                writer.writerow(asdict(player))
        
        print(f"Saved {len(players)} players to {filepath}")
    
    def save_to_json(self, players: List[PlayerElo], filename: str = "wta_elo_data.json"):
        """Save player data to JSON"""
        if not players:
            return
        
        filepath = Path(filename)
        
        data = [asdict(p) for p in players]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(players)} players to {filepath}")


# Top 150 WTA players (current rankings - Jan 2026)
# This is our target list to fetch
TOP_150_WTA = [
    "Aryna Sabalenka",
    "Iga Swiatek", 
    "Coco Gauff",
    "Jasmine Paolini",
    "Jessica Pegula",
    "Qinwen Zheng",
    "Elena Rybakina",
    "Emma Navarro",
    "Daria Kasatkina",
    "Barbora Krejcikova",
    "Paula Badosa",
    "Diana Shnaider",
    "Mirra Andreeva",
    "Anna Kalinskaya",
    "Marta Kostyuk",
    "Beatriz Haddad Maia",
    "Madison Keys",
    "Donna Vekic",
    "Maria Sakkari",
    "Dayana Yastremska",
    "Karolina Muchova",
    "Elina Svitolina",
    "Leylah Fernandez",
    "Jelena Ostapenko",
    "Linda Noskova",
    "Katie Boulter",
    "Ekaterina Alexandrova",
    "Yulia Putintseva",
    "Danielle Collins",
    "Elise Mertens",
    "Naomi Osaka",
    "Magda Linette",
    "Anastasia Pavlyuchenkova",
    "Anastasia Potapova",
    "Emma Raducanu",
    "Clara Tauson",
    "Marketa Vondrousova",
    "Victoria Azarenka",
    "Caroline Garcia",
    "Liudmila Samsonova",
    "Ons Jabeur",
    "Amanda Anisimova",
    "Belinda Bencic",
    "Sloane Stephens",
    "Venus Williams",
    "Sofia Kenin",
    "Karolina Pliskova",
    "Sorana Cirstea",
    "Ajla Tomljanovic",
    "Anna Blinkova",
    "Anhelina Kalinina",
    "Camila Osorio",
    "Cristina Bucsa",
    "Peyton Stearns",
    "Moyuka Uchijima",
    "Linda Fruhvirtova",
    "Viktorija Golubic",
    "Kaja Juvan",
    "Magdalena Frech",
    "Katie Volynets",
    "Lulu Sun",
    "Jessica Bouzas Maneiro",
    "Yuan Yue",
    "Marie Bouzkova",
    "Elisabetta Cocciaretto",
    "Eva Lys",
    "Aliaksandra Sasnovich",
    "Rebecca Sramkova",
    "Varvara Gracheva",
    "Sara Bejlek",
    "Ashlyn Krueger",
    "Caty McNally",
    "Polina Kudermetova",
    "Dalma Galfi",
    "Anna Bondar",
    "Alycia Parks",
    "Jaqueline Cristian",
    "Elena-Gabriela Ruse",
    "Yuliya Starodubtseva",
    "Oceane Dodin",
    "Wang Xinyu",
    "Wang Yafan",
    "Shuai Zhang",
    "Hailey Baptiste",
    "Robin Montgomery",
    "Brenda Fruhvirtova",
    "Olga Danilovic",
    "Diane Parry",
    "Clara Burel",
    "Nadia Podoroska",
    "Sara Sorribes Tormo",
    "Tatjana Maria",
    "Laura Siegemund",
    "Katerina Siniakova",
    "Bernarda Pera",
    "Petra Martic",
    "Irina-Camelia Begu",
    "Tamara Zidansek",
    "Panna Udvardy",
    "Simona Waltert",
    # Add more to reach 150...
]


def main():
    """Test the scraper"""
    print("Tennis Abstract Data Scraper")
    print("="*50)
    
    scraper = TennisAbstractScraper(cache_dir="v2/cache")
    
    # Try fetching a single player
    print("\nTesting single player fetch...")
    player = scraper.fetch_player("Aryna Sabalenka")
    if player:
        print(f"Found: {player.name}")
        print(f"  Overall Elo: {player.overall_elo}")
        print(f"  Hard Elo: {player.hard_elo}")
        print(f"  Clay Elo: {player.clay_elo}")
    else:
        print("Could not fetch player data")
    
    print("\nNote: Tennis Abstract may require different parsing.")
    print("Run 'python data_scraper.py --fetch-all' to attempt full scrape.")


if __name__ == "__main__":
    main()
