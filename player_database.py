"""
Player Database for Tennis Model v2
Stores Elo ratings and player statistics
"""

import json
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List
from pathlib import Path
from datetime import datetime


@dataclass
class PlayerRecord:
    """Complete player record with all Elo ratings"""
    name: str
    
    # Rankings
    wta_rank: Optional[int] = None
    
    # Overall Elo
    elo_overall: float = 1500.0
    elo_peak: Optional[float] = None
    
    # Surface-specific Elo
    elo_hard: float = 1500.0
    elo_clay: float = 1500.0
    elo_grass: float = 1500.0
    
    # Surface peaks
    elo_hard_peak: Optional[float] = None
    elo_clay_peak: Optional[float] = None
    elo_grass_peak: Optional[float] = None
    
    # Serve statistics (career averages)
    first_serve_pct: Optional[float] = None  # % of first serves in
    first_serve_won: Optional[float] = None  # % of first serve points won
    second_serve_won: Optional[float] = None  # % of second serve points won
    ace_pct: Optional[float] = None  # Aces per service point
    df_pct: Optional[float] = None  # Double faults per service point
    
    # Return statistics
    return_first_won: Optional[float] = None  # % of return points won vs first serve
    return_second_won: Optional[float] = None  # % of return points won vs second serve
    break_point_conversion: Optional[float] = None
    
    # Metadata
    country: Optional[str] = None
    age: Optional[int] = None
    turned_pro: Optional[int] = None
    
    # Data source tracking
    elo_source: str = "estimated"
    stats_source: str = "estimated"
    last_updated: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PlayerRecord':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class PlayerDatabase:
    """
    Database for storing and querying player records
    Supports JSON persistence
    """
    
    def __init__(self, filepath: str = "v2/players.json"):
        self.filepath = Path(filepath)
        self.players: Dict[str, PlayerRecord] = {}
        self._load()
    
    def _load(self):
        """Load database from JSON file"""
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for name, record in data.items():
                        self.players[name] = PlayerRecord.from_dict(record)
                print(f"Loaded {len(self.players)} players from {self.filepath}")
            except Exception as e:
                print(f"Error loading database: {e}")
    
    def save(self):
        """Save database to JSON file"""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {name: player.to_dict() for name, player in self.players.items()}
        
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(self.players)} players to {self.filepath}")
    
    def add_player(self, player: PlayerRecord):
        """Add or update a player"""
        player.last_updated = datetime.now().isoformat()
        self.players[player.name] = player
    
    def get_player(self, name: str) -> Optional[PlayerRecord]:
        """Get player by name (case-insensitive)"""
        # Exact match first
        if name in self.players:
            return self.players[name]
        
        # Case-insensitive search
        name_lower = name.lower()
        for player_name, player in self.players.items():
            if player_name.lower() == name_lower:
                return player
        
        # Partial match (last name)
        for player_name, player in self.players.items():
            if name_lower in player_name.lower():
                return player
        
        return None
    
    def get_top_n(self, n: int = 150) -> List[PlayerRecord]:
        """Get top N players by WTA rank"""
        ranked = [p for p in self.players.values() if p.wta_rank is not None]
        ranked.sort(key=lambda p: p.wta_rank)
        return ranked[:n]
    
    def get_by_elo(self, surface: str = "overall", top_n: int = 150) -> List[PlayerRecord]:
        """Get top N players by Elo rating"""
        elo_field = f"elo_{surface}" if surface != "overall" else "elo_overall"
        
        players_list = list(self.players.values())
        players_list.sort(key=lambda p: getattr(p, elo_field, 1500), reverse=True)
        return players_list[:top_n]
    
    def bulk_update_elo(self, elo_data: Dict[str, Dict[str, float]]):
        """
        Bulk update Elo ratings from dictionary
        Format: {player_name: {overall: X, hard: Y, clay: Z, grass: W}}
        """
        for name, ratings in elo_data.items():
            player = self.get_player(name)
            if not player:
                player = PlayerRecord(name=name)
            
            if 'overall' in ratings:
                player.elo_overall = ratings['overall']
            if 'hard' in ratings:
                player.elo_hard = ratings['hard']
            if 'clay' in ratings:
                player.elo_clay = ratings['clay']
            if 'grass' in ratings:
                player.elo_grass = ratings['grass']
            
            player.elo_source = "tennis_abstract"
            self.add_player(player)
    
    def export_csv(self, filepath: str = "v2/players_export.csv"):
        """Export database to CSV"""
        import csv
        
        if not self.players:
            return
        
        fields = list(asdict(list(self.players.values())[0]).keys())
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for player in self.players.values():
                writer.writerow(player.to_dict())
        
        print(f"Exported to {filepath}")
    
    def import_csv(self, filepath: str):
        """Import players from CSV"""
        import csv
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                for field in ['wta_rank', 'age', 'turned_pro']:
                    if row.get(field):
                        try:
                            row[field] = int(row[field])
                        except ValueError:
                            row[field] = None
                
                for field in ['elo_overall', 'elo_hard', 'elo_clay', 'elo_grass',
                             'elo_peak', 'elo_hard_peak', 'elo_clay_peak', 'elo_grass_peak',
                             'first_serve_pct', 'first_serve_won', 'second_serve_won',
                             'ace_pct', 'df_pct', 'return_first_won', 'return_second_won',
                             'break_point_conversion']:
                    if row.get(field):
                        try:
                            row[field] = float(row[field])
                        except ValueError:
                            row[field] = None
                
                player = PlayerRecord.from_dict(row)
                self.add_player(player)
        
        print(f"Imported from {filepath}")


# Initialize with estimated data based on current WTA rankings
# These will be replaced with real Tennis Abstract data
INITIAL_PLAYER_DATA = {
    # Top 10
    "Aryna Sabalenka": {"rank": 1, "overall": 2150, "hard": 2180, "clay": 2050, "grass": 2000, "country": "BLR"},
    "Iga Swiatek": {"rank": 2, "overall": 2120, "hard": 2050, "clay": 2220, "grass": 1950, "country": "POL"},
    "Coco Gauff": {"rank": 3, "overall": 2000, "hard": 2020, "clay": 1950, "grass": 1920, "country": "USA"},
    "Jasmine Paolini": {"rank": 4, "overall": 1950, "hard": 1920, "clay": 2000, "grass": 1980, "country": "ITA"},
    "Jessica Pegula": {"rank": 5, "overall": 1920, "hard": 1980, "clay": 1850, "grass": 1880, "country": "USA"},
    "Qinwen Zheng": {"rank": 6, "overall": 1940, "hard": 1980, "clay": 1880, "grass": 1850, "country": "CHN"},
    "Elena Rybakina": {"rank": 7, "overall": 1960, "hard": 1970, "clay": 1900, "grass": 2050, "country": "KAZ"},
    "Emma Navarro": {"rank": 8, "overall": 1880, "hard": 1910, "clay": 1820, "grass": 1870, "country": "USA"},
    "Daria Kasatkina": {"rank": 9, "overall": 1850, "hard": 1830, "clay": 1900, "grass": 1780, "country": "RUS"},
    "Barbora Krejcikova": {"rank": 10, "overall": 1870, "hard": 1850, "clay": 1920, "grass": 1900, "country": "CZE"},
    
    # 11-20
    "Paula Badosa": {"rank": 11, "overall": 1860, "hard": 1880, "clay": 1870, "grass": 1800, "country": "ESP"},
    "Diana Shnaider": {"rank": 12, "overall": 1840, "hard": 1870, "clay": 1800, "grass": 1780, "country": "RUS"},
    "Mirra Andreeva": {"rank": 13, "overall": 1850, "hard": 1840, "clay": 1880, "grass": 1800, "country": "RUS"},
    "Anna Kalinskaya": {"rank": 14, "overall": 1820, "hard": 1860, "clay": 1770, "grass": 1780, "country": "RUS"},
    "Marta Kostyuk": {"rank": 15, "overall": 1800, "hard": 1820, "clay": 1780, "grass": 1760, "country": "UKR"},
    "Beatriz Haddad Maia": {"rank": 16, "overall": 1810, "hard": 1780, "clay": 1870, "grass": 1820, "country": "BRA"},
    "Madison Keys": {"rank": 17, "overall": 1870, "hard": 1920, "clay": 1800, "grass": 1850, "country": "USA"},
    "Donna Vekic": {"rank": 18, "overall": 1820, "hard": 1840, "clay": 1800, "grass": 1830, "country": "CRO"},
    "Maria Sakkari": {"rank": 19, "overall": 1830, "hard": 1860, "clay": 1810, "grass": 1780, "country": "GRE"},
    "Dayana Yastremska": {"rank": 20, "overall": 1790, "hard": 1810, "clay": 1750, "grass": 1780, "country": "UKR"},
    
    # 21-30
    "Karolina Muchova": {"rank": 21, "overall": 1860, "hard": 1850, "clay": 1890, "grass": 1840, "country": "CZE"},
    "Elina Svitolina": {"rank": 22, "overall": 1820, "hard": 1830, "clay": 1820, "grass": 1800, "country": "UKR"},
    "Leylah Fernandez": {"rank": 23, "overall": 1780, "hard": 1800, "clay": 1760, "grass": 1750, "country": "CAN"},
    "Jelena Ostapenko": {"rank": 24, "overall": 1800, "hard": 1780, "clay": 1850, "grass": 1780, "country": "LAT"},
    "Linda Noskova": {"rank": 25, "overall": 1790, "hard": 1820, "clay": 1760, "grass": 1750, "country": "CZE"},
    "Katie Boulter": {"rank": 26, "overall": 1770, "hard": 1760, "clay": 1720, "grass": 1840, "country": "GBR"},
    "Ekaterina Alexandrova": {"rank": 27, "overall": 1780, "hard": 1810, "clay": 1740, "grass": 1760, "country": "RUS"},
    "Yulia Putintseva": {"rank": 28, "overall": 1750, "hard": 1770, "clay": 1740, "grass": 1720, "country": "KAZ"},
    "Danielle Collins": {"rank": 29, "overall": 1840, "hard": 1870, "clay": 1800, "grass": 1800, "country": "USA"},
    "Elise Mertens": {"rank": 30, "overall": 1760, "hard": 1780, "clay": 1750, "grass": 1730, "country": "BEL"},
    
    # 31-50
    "Naomi Osaka": {"rank": 31, "overall": 1850, "hard": 1900, "clay": 1750, "grass": 1800, "country": "JPN"},
    "Magda Linette": {"rank": 32, "overall": 1740, "hard": 1760, "clay": 1730, "grass": 1710, "country": "POL"},
    "Anastasia Pavlyuchenkova": {"rank": 33, "overall": 1730, "hard": 1750, "clay": 1720, "grass": 1700, "country": "RUS"},
    "Anastasia Potapova": {"rank": 34, "overall": 1720, "hard": 1740, "clay": 1700, "grass": 1690, "country": "RUS"},
    "Emma Raducanu": {"rank": 35, "overall": 1760, "hard": 1800, "clay": 1700, "grass": 1780, "country": "GBR"},
    "Clara Tauson": {"rank": 36, "overall": 1730, "hard": 1760, "clay": 1700, "grass": 1720, "country": "DEN"},
    "Marketa Vondrousova": {"rank": 37, "overall": 1810, "hard": 1770, "clay": 1850, "grass": 1870, "country": "CZE"},
    "Victoria Azarenka": {"rank": 38, "overall": 1800, "hard": 1830, "clay": 1760, "grass": 1770, "country": "BLR"},
    "Caroline Garcia": {"rank": 39, "overall": 1790, "hard": 1810, "clay": 1780, "grass": 1750, "country": "FRA"},
    "Liudmila Samsonova": {"rank": 40, "overall": 1810, "hard": 1860, "clay": 1750, "grass": 1820, "country": "RUS"},
    "Ons Jabeur": {"rank": 41, "overall": 1830, "hard": 1800, "clay": 1850, "grass": 1900, "country": "TUN"},
    "Amanda Anisimova": {"rank": 42, "overall": 1740, "hard": 1770, "clay": 1710, "grass": 1720, "country": "USA"},
    "Belinda Bencic": {"rank": 43, "overall": 1780, "hard": 1800, "clay": 1760, "grass": 1770, "country": "SUI"},
    "Sloane Stephens": {"rank": 44, "overall": 1720, "hard": 1740, "clay": 1710, "grass": 1690, "country": "USA"},
    "Venus Williams": {"rank": 45, "overall": 1680, "hard": 1700, "clay": 1650, "grass": 1720, "country": "USA"},
    "Sofia Kenin": {"rank": 46, "overall": 1720, "hard": 1760, "clay": 1680, "grass": 1700, "country": "USA"},
    "Karolina Pliskova": {"rank": 47, "overall": 1760, "hard": 1800, "clay": 1720, "grass": 1760, "country": "CZE"},
    "Sorana Cirstea": {"rank": 48, "overall": 1720, "hard": 1740, "clay": 1700, "grass": 1690, "country": "ROU"},
    "Ajla Tomljanovic": {"rank": 49, "overall": 1710, "hard": 1730, "clay": 1690, "grass": 1720, "country": "AUS"},
    "Anna Blinkova": {"rank": 50, "overall": 1700, "hard": 1720, "clay": 1680, "grass": 1670, "country": "RUS"},
}


def initialize_database():
    """Initialize database with estimated player data"""
    db = PlayerDatabase("v2/players.json")
    
    for name, data in INITIAL_PLAYER_DATA.items():
        player = PlayerRecord(
            name=name,
            wta_rank=data.get("rank"),
            elo_overall=data.get("overall", 1500),
            elo_hard=data.get("hard", 1500),
            elo_clay=data.get("clay", 1500),
            elo_grass=data.get("grass", 1500),
            country=data.get("country"),
            elo_source="estimated_from_rankings"
        )
        db.add_player(player)
    
    db.save()
    return db


if __name__ == "__main__":
    print("Initializing Player Database v2")
    print("="*50)
    
    db = initialize_database()
    
    print(f"\nLoaded {len(db.players)} players")
    print("\nTop 10 by Hard Court Elo:")
    for i, player in enumerate(db.get_by_elo("hard", 10), 1):
        print(f"  {i}. {player.name}: {player.elo_hard}")
