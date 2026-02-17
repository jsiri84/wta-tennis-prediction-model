"""
Court Speed Classification based on ATP data (2026 update).
Applied to WTA tournaments - same venues have same physical properties.

Data source: ATP 2026 Tournament Speed Ranking (5-year avg 2021-2025)
Key metric: 1st Serve Points Won %
  - Fast:   >73% (serves dominate)
  - Medium: 70-73% (balanced)
  - Slow:   <70% (returns/rallies dominate)
"""

# Court speed index (1st serve points won %)
# Higher = faster court
# Updated with 2026 ATP data (5-year averages)
COURT_SPEED_DATA = {
    # FAST COURTS (>73%)
    # Grass
    'stuttgart': {'speed': 77.9, 'category': 'fast', 'surface': 'grass', 'altitude': 225},
    'halle': {'speed': 75.4, 'category': 'fast', 'surface': 'grass', 'altitude': 120},
    'queens': {'speed': 74.8, 'category': 'fast', 'surface': 'grass', 'altitude': 25},
    'london': {'speed': 74.8, 'category': 'fast', 'surface': 'grass', 'altitude': 25},
    'mallorca': {'speed': 74.1, 'category': 'fast', 'surface': 'grass', 'altitude': 35},
    's hertogenbosch': {'speed': 73.9, 'category': 'fast', 'surface': 'grass', 'altitude': 5},
    'hertogenbosch': {'speed': 73.9, 'category': 'fast', 'surface': 'grass', 'altitude': 5},
    'wimbledon': {'speed': 73.3, 'category': 'fast', 'surface': 'grass', 'altitude': 25},
    'birmingham': {'speed': 73.5, 'category': 'fast', 'surface': 'grass', 'altitude': 100},  # estimated
    'berlin': {'speed': 73.5, 'category': 'fast', 'surface': 'grass', 'altitude': 35},  # estimated
    'bad homburg': {'speed': 73.5, 'category': 'fast', 'surface': 'grass', 'altitude': 200},  # estimated
    'nottingham': {'speed': 73.5, 'category': 'fast', 'surface': 'grass', 'altitude': 50},  # estimated
    
    # Indoor Hard (generally fast)
    'brussels': {'speed': 76.7, 'category': 'fast', 'surface': 'hard', 'altitude': 55},
    'basel': {'speed': 75.0, 'category': 'fast', 'surface': 'hard', 'altitude': 285},
    'wta finals': {'speed': 74.5, 'category': 'fast', 'surface': 'hard', 'altitude': 0},  # varies by host
    'turin': {'speed': 74.5, 'category': 'fast', 'surface': 'hard', 'altitude': 250},
    'dallas': {'speed': 73.9, 'category': 'fast', 'surface': 'hard', 'altitude': 170},
    'vienna': {'speed': 73.9, 'category': 'fast', 'surface': 'hard', 'altitude': 220},
    'paris': {'speed': 73.6, 'category': 'fast', 'surface': 'hard', 'altitude': 50},  # indoor
    'paris indoor': {'speed': 73.6, 'category': 'fast', 'surface': 'hard', 'altitude': 50},
    'almaty': {'speed': 73.2, 'category': 'fast', 'surface': 'hard', 'altitude': 800},
    'rotterdam': {'speed': 73.0, 'category': 'fast', 'surface': 'hard', 'altitude': 0},
    'linz': {'speed': 73.0, 'category': 'fast', 'surface': 'hard', 'altitude': 260},  # estimated
    'ostrava': {'speed': 73.0, 'category': 'fast', 'surface': 'hard', 'altitude': 210},  # estimated
    'luxembourg': {'speed': 72.5, 'category': 'medium', 'surface': 'hard', 'altitude': 300},  # estimated
    'montpellier': {'speed': 72.3, 'category': 'medium', 'surface': 'hard', 'altitude': 5},
    'stockholm': {'speed': 72.1, 'category': 'medium', 'surface': 'hard', 'altitude': 20},
    'metz': {'speed': 72.0, 'category': 'medium', 'surface': 'hard', 'altitude': 180},  # estimated
    
    # Fast Outdoor Hard
    'brisbane': {'speed': 75.4, 'category': 'fast', 'surface': 'hard', 'altitude': 25},
    'chengdu': {'speed': 74.0, 'category': 'fast', 'surface': 'hard', 'altitude': 500},
    'cincinnati': {'speed': 73.8, 'category': 'fast', 'surface': 'hard', 'altitude': 235},
    'dubai': {'speed': 73.6, 'category': 'fast', 'surface': 'hard', 'altitude': 25},
    'adelaide': {'speed': 73.5, 'category': 'fast', 'surface': 'hard', 'altitude': 30},
    'shanghai': {'speed': 73.0, 'category': 'fast', 'surface': 'hard', 'altitude': 15},
    
    # MEDIUM COURTS (70-73%)
    'hong kong': {'speed': 72.5, 'category': 'medium', 'surface': 'hard', 'altitude': 5},
    'washington': {'speed': 72.5, 'category': 'medium', 'surface': 'hard', 'altitude': 80},
    'hangzhou': {'speed': 72.5, 'category': 'medium', 'surface': 'hard', 'altitude': 40},
    'eastbourne': {'speed': 72.2, 'category': 'medium', 'surface': 'grass', 'altitude': 10},
    'us open': {'speed': 72.2, 'category': 'medium', 'surface': 'hard', 'altitude': 30},
    'new york': {'speed': 72.2, 'category': 'medium', 'surface': 'hard', 'altitude': 30},
    'australian open': {'speed': 72.2, 'category': 'medium', 'surface': 'hard', 'altitude': 5},
    'melbourne': {'speed': 72.2, 'category': 'medium', 'surface': 'hard', 'altitude': 5},
    'miami': {'speed': 72.2, 'category': 'medium', 'surface': 'hard', 'altitude': 15},
    'tokyo': {'speed': 72.2, 'category': 'medium', 'surface': 'hard', 'altitude': 5},
    'toronto': {'speed': 72.0, 'category': 'medium', 'surface': 'hard', 'altitude': 75},
    'montreal': {'speed': 72.0, 'category': 'medium', 'surface': 'hard', 'altitude': 45},
    'winston-salem': {'speed': 71.9, 'category': 'medium', 'surface': 'hard', 'altitude': 240},
    'doha': {'speed': 71.9, 'category': 'medium', 'surface': 'hard', 'altitude': 5},
    'qatar': {'speed': 71.9, 'category': 'medium', 'surface': 'hard', 'altitude': 5},
    'los cabos': {'speed': 71.7, 'category': 'medium', 'surface': 'hard', 'altitude': 35},
    'auckland': {'speed': 71.6, 'category': 'medium', 'surface': 'hard', 'altitude': 20},
    'madrid': {'speed': 71.5, 'category': 'medium', 'surface': 'clay', 'altitude': 565},  # high altitude clay
    'delray beach': {'speed': 71.2, 'category': 'medium', 'surface': 'hard', 'altitude': 5},
    'beijing': {'speed': 70.9, 'category': 'medium', 'surface': 'hard', 'altitude': 50},
    'acapulco': {'speed': 70.6, 'category': 'medium', 'surface': 'hard', 'altitude': 10},
    'san diego': {'speed': 71.0, 'category': 'medium', 'surface': 'hard', 'altitude': 20},  # estimated
    'charleston': {'speed': 70.5, 'category': 'medium', 'surface': 'clay', 'altitude': 5},  # green clay
    'guadalajara': {'speed': 71.0, 'category': 'medium', 'surface': 'hard', 'altitude': 1500},  # estimated
    'indian wells': {'speed': 70.3, 'category': 'medium', 'surface': 'hard', 'altitude': 45},
    
    # MEDIUM-SLOW Clay (altitude helps)
    'gstaad': {'speed': 72.4, 'category': 'medium', 'surface': 'clay', 'altitude': 1050},
    'houston': {'speed': 71.6, 'category': 'medium', 'surface': 'clay', 'altitude': 20},
    'geneva': {'speed': 71.1, 'category': 'medium', 'surface': 'clay', 'altitude': 395},
    'kitzbuhel': {'speed': 70.5, 'category': 'medium', 'surface': 'clay', 'altitude': 760},
    
    # SLOW COURTS (<70%)
    'marrakech': {'speed': 69.1, 'category': 'slow', 'surface': 'clay', 'altitude': 460},
    'rome': {'speed': 69.1, 'category': 'slow', 'surface': 'clay', 'altitude': 20},
    'italian open': {'speed': 69.1, 'category': 'slow', 'surface': 'clay', 'altitude': 20},
    'roland garros': {'speed': 69.0, 'category': 'slow', 'surface': 'clay', 'altitude': 35},
    'french open': {'speed': 69.0, 'category': 'slow', 'surface': 'clay', 'altitude': 35},
    'santiago': {'speed': 68.8, 'category': 'slow', 'surface': 'clay', 'altitude': 560},
    'munich': {'speed': 68.1, 'category': 'slow', 'surface': 'clay', 'altitude': 480},
    'hamburg': {'speed': 68.0, 'category': 'slow', 'surface': 'clay', 'altitude': 20},
    'umag': {'speed': 67.4, 'category': 'slow', 'surface': 'clay', 'altitude': 0},
    'bastad': {'speed': 67.2, 'category': 'slow', 'surface': 'clay', 'altitude': 5},
    'monte carlo': {'speed': 67.1, 'category': 'slow', 'surface': 'clay', 'altitude': 40},
    'monte-carlo': {'speed': 67.1, 'category': 'slow', 'surface': 'clay', 'altitude': 40},
    'estoril': {'speed': 67.1, 'category': 'slow', 'surface': 'clay', 'altitude': 85},
    'rio': {'speed': 66.6, 'category': 'slow', 'surface': 'clay', 'altitude': 20},
    'rio de janeiro': {'speed': 66.6, 'category': 'slow', 'surface': 'clay', 'altitude': 20},
    'barcelona': {'speed': 66.3, 'category': 'slow', 'surface': 'clay', 'altitude': 95},
    'buenos aires': {'speed': 65.1, 'category': 'slow', 'surface': 'clay', 'altitude': 20},
    'bucharest': {'speed': 64.8, 'category': 'slow', 'surface': 'clay', 'altitude': 80},
    
    # WTA-specific tournaments (estimated based on similar venues)
    'bogota': {'speed': 70.0, 'category': 'medium', 'surface': 'clay', 'altitude': 2600},  # very high altitude
    'rabat': {'speed': 67.5, 'category': 'slow', 'surface': 'clay', 'altitude': 75},
    'strasbourg': {'speed': 68.0, 'category': 'slow', 'surface': 'clay', 'altitude': 140},
    'prague': {'speed': 67.5, 'category': 'slow', 'surface': 'clay', 'altitude': 200},
    'palermo': {'speed': 67.0, 'category': 'slow', 'surface': 'clay', 'altitude': 15},
    'parma': {'speed': 67.5, 'category': 'slow', 'surface': 'clay', 'altitude': 55},
    'lausanne': {'speed': 68.0, 'category': 'slow', 'surface': 'clay', 'altitude': 495},
    'hua hin': {'speed': 71.5, 'category': 'medium', 'surface': 'hard', 'altitude': 5},
    'hobart': {'speed': 72.0, 'category': 'medium', 'surface': 'hard', 'altitude': 5},
    'sydney': {'speed': 72.5, 'category': 'medium', 'surface': 'hard', 'altitude': 5},
    'abu dhabi': {'speed': 72.0, 'category': 'medium', 'surface': 'hard', 'altitude': 5},
    'seoul': {'speed': 72.0, 'category': 'medium', 'surface': 'hard', 'altitude': 40},
    'ningbo': {'speed': 72.0, 'category': 'medium', 'surface': 'hard', 'altitude': 5},
    'wuhan': {'speed': 72.5, 'category': 'medium', 'surface': 'hard', 'altitude': 25},
    'zhengzhou': {'speed': 72.0, 'category': 'medium', 'surface': 'hard', 'altitude': 110},
    'singapore': {'speed': 72.0, 'category': 'medium', 'surface': 'hard', 'altitude': 5},
    'san jose': {'speed': 72.0, 'category': 'medium', 'surface': 'hard', 'altitude': 25},
    'cleveland': {'speed': 71.5, 'category': 'medium', 'surface': 'hard', 'altitude': 200},
    'portoroz': {'speed': 71.5, 'category': 'medium', 'surface': 'hard', 'altitude': 5},
    'granby': {'speed': 71.5, 'category': 'medium', 'surface': 'hard', 'altitude': 250},
    'monterrey': {'speed': 71.5, 'category': 'medium', 'surface': 'hard', 'altitude': 540},
}

# Default speeds by surface (when tournament not found)
DEFAULT_SPEEDS = {
    'grass': {'speed': 73.5, 'category': 'fast', 'altitude': 0},
    'hard': {'speed': 71.5, 'category': 'medium', 'altitude': 0},
    'clay': {'speed': 67.5, 'category': 'slow', 'altitude': 0},
    'indoor': {'speed': 73.5, 'category': 'fast', 'altitude': 0},
}

# Speed thresholds
FAST_THRESHOLD = 73.0
SLOW_THRESHOLD = 70.0


def get_court_speed(tournament_name, surface='hard'):
    """
    Get court speed data for a tournament.
    
    Returns: dict with 'speed' (1st serve pts won %), 'category' (fast/medium/slow), 'altitude'
    """
    if not tournament_name:
        return DEFAULT_SPEEDS.get(surface.lower(), DEFAULT_SPEEDS['hard'])
    
    # Normalize tournament name
    name = tournament_name.lower().strip()
    
    # Try exact match first (with common variations)
    if name in COURT_SPEED_DATA:
        return COURT_SPEED_DATA[name]
    
    # Try with suffixes removed
    name_stripped = name
    for suffix in [' open', ' classic', ' championships', ' trophy', ' cup', ' masters']:
        name_stripped = name_stripped.replace(suffix, '')
    
    if name_stripped in COURT_SPEED_DATA:
        return COURT_SPEED_DATA[name_stripped]
    
    # Try partial match - prefer longer key matches to avoid "brussels" matching "us"
    best_match = None
    best_match_len = 0
    for key, data in COURT_SPEED_DATA.items():
        # Key must be a substantial part of name, not just 2 letters
        if len(key) >= 4 and (key in name or name_stripped in key):
            if len(key) > best_match_len:
                best_match = data
                best_match_len = len(key)
    
    if best_match:
        return best_match
    
    # Fall back to surface default
    return DEFAULT_SPEEDS.get(surface.lower(), DEFAULT_SPEEDS['hard'])


def get_speed_category(tournament_name, surface='hard'):
    """Get just the speed category (fast/medium/slow)."""
    return get_court_speed(tournament_name, surface)['category']


def get_speed_index(tournament_name, surface='hard'):
    """
    Get numeric speed index (0-100 scale, higher = faster).
    0 = slowest (64%), 100 = fastest (78%)
    """
    data = get_court_speed(tournament_name, surface)
    speed = data['speed']
    # Normalize to 0-100 scale
    normalized = (speed - 64) / (78 - 64) * 100
    return max(0, min(100, normalized))


def get_speed_adjustment(tournament_name, surface='hard'):
    """
    Get a speed adjustment factor for skills.
    Returns: float between -0.1 (very slow) and +0.1 (very fast)
    Centered at 0 for medium courts (71.5%)
    """
    data = get_court_speed(tournament_name, surface)
    speed = data['speed']
    # Center at 71.5% (medium), scale so ±7% speed = ±0.1 adjustment
    adjustment = (speed - 71.5) / 70  # Gives ~±0.1 range
    return adjustment


# Test
if __name__ == '__main__':
    print("="*70)
    print("COURT SPEED CLASSIFICATION (2026 ATP Data)")
    print("="*70)
    
    test_tournaments = [
        ('Australian Open', 'Hard'),
        ('French Open', 'Clay'),
        ('Wimbledon', 'Grass'),
        ('US Open', 'Hard'),
        ('Miami', 'Hard'),
        ('Madrid', 'Clay'),
        ('Rome', 'Clay'),
        ('Cincinnati', 'Hard'),
        ('Dubai', 'Hard'),
        ('Stuttgart', 'Grass'),
        ('Indian Wells', 'Hard'),
        ('Brisbane', 'Hard'),
    ]
    
    print("\nMajor Tournaments:")
    print("-"*70)
    print(f"{'Tournament':<20} {'Category':<10} {'Speed':>8} {'Index':>8} {'Adj':>8}")
    print("-"*70)
    for tournament, surface in test_tournaments:
        data = get_court_speed(tournament, surface)
        idx = get_speed_index(tournament, surface)
        adj = get_speed_adjustment(tournament, surface)
        print(f"{tournament:<20} {data['category']:<10} {data['speed']:>7.1f}% {idx:>7.0f} {adj:>+7.3f}")
    
    print("\n\nSpeed Categories Summary:")
    print("-"*70)
    fast = [k for k, v in COURT_SPEED_DATA.items() if v['category'] == 'fast']
    medium = [k for k, v in COURT_SPEED_DATA.items() if v['category'] == 'medium']
    slow = [k for k, v in COURT_SPEED_DATA.items() if v['category'] == 'slow']
    
    print(f"  Fast courts (>73%):    {len(fast)} tournaments")
    print(f"  Medium courts (70-73%): {len(medium)} tournaments")
    print(f"  Slow courts (<70%):    {len(slow)} tournaments")
    
    print("\n\nGrand Slam Speed Comparison:")
    print("-"*70)
    slams = ['Australian Open', 'French Open', 'Wimbledon', 'US Open']
    for slam in slams:
        surface = 'Clay' if 'French' in slam else ('Grass' if 'Wimbledon' in slam else 'Hard')
        data = get_court_speed(slam, surface)
        print(f"  {slam:<20} {data['speed']:.1f}% ({data['category']})")
