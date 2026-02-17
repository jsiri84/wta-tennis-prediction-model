"""
Tournament location database for travel distance calculations.
Coordinates are approximate city centers.
"""

from math import radians, sin, cos, sqrt, atan2

# Major WTA tournaments mapped to (latitude, longitude)
# Format: tournament_name_pattern -> (lat, lon, city)
TOURNAMENT_LOCATIONS = {
    # Grand Slams
    'australian open': (-37.8136, 144.9631, 'Melbourne'),
    'french open': (48.8566, 2.3522, 'Paris'),
    'roland garros': (48.8566, 2.3522, 'Paris'),
    'wimbledon': (51.5074, -0.1278, 'London'),
    'us open': (40.7128, -74.0060, 'New York'),
    
    # WTA 1000
    'indian wells': (33.7238, -116.3511, 'Indian Wells'),
    'bnp paribas': (33.7238, -116.3511, 'Indian Wells'),
    'miami': (25.7617, -80.1918, 'Miami'),
    'madrid': (40.4168, -3.7038, 'Madrid'),
    'mutua madrid': (40.4168, -3.7038, 'Madrid'),
    'rome': (41.9028, 12.4964, 'Rome'),
    'italian open': (41.9028, 12.4964, 'Rome'),
    'internazionali': (41.9028, 12.4964, 'Rome'),
    'canada': (43.6532, -79.3832, 'Toronto'),
    'toronto': (43.6532, -79.3832, 'Toronto'),
    'montreal': (45.5017, -73.5673, 'Montreal'),
    'cincinnati': (39.1031, -84.5120, 'Cincinnati'),
    'western & southern': (39.1031, -84.5120, 'Cincinnati'),
    'beijing': (39.9042, 116.4074, 'Beijing'),
    'china open': (39.9042, 116.4074, 'Beijing'),
    'wuhan': (30.5928, 114.3055, 'Wuhan'),
    'wta finals': (22.3193, 114.1694, 'Shenzhen'),  # Location varies
    
    # WTA 500
    'abu dhabi': (24.4539, 54.3773, 'Abu Dhabi'),
    'adelaide': (-34.9285, 138.6007, 'Adelaide'),
    'brisbane': (-27.4698, 153.0251, 'Brisbane'),
    'united cup': (-33.8688, 151.2093, 'Sydney'),  # Multiple cities
    'sydney': (-33.8688, 151.2093, 'Sydney'),
    'doha': (25.2854, 51.5310, 'Doha'),
    'qatar': (25.2854, 51.5310, 'Doha'),
    'dubai': (25.2048, 55.2708, 'Dubai'),
    'san diego': (32.7157, -117.1611, 'San Diego'),
    'charleston': (32.7765, -79.9311, 'Charleston'),
    'stuttgart': (48.7758, 9.1829, 'Stuttgart'),
    'berlin': (52.5200, 13.4050, 'Berlin'),
    'eastbourne': (50.7684, 0.2905, 'Eastbourne'),
    'bad homburg': (50.2267, 8.6186, 'Bad Homburg'),
    's-hertogenbosch': (51.6978, 5.3037, 's-Hertogenbosch'),
    'hertogenbosch': (51.6978, 5.3037, 's-Hertogenbosch'),
    'birmingham': (52.4862, -1.8904, 'Birmingham'),
    'san jose': (37.3382, -121.8863, 'San Jose'),
    'tokyo': (35.6762, 139.6503, 'Tokyo'),
    'pan pacific': (35.6762, 139.6503, 'Tokyo'),
    'seoul': (37.5665, 126.9780, 'Seoul'),
    'korea': (37.5665, 126.9780, 'Seoul'),
    'ostrava': (49.8209, 18.2625, 'Ostrava'),
    'guadalajara': (20.6597, -103.3496, 'Guadalajara'),
    'linz': (48.3069, 14.2858, 'Linz'),
    
    # WTA 250
    'auckland': (-36.8485, 174.7633, 'Auckland'),
    'hobart': (-42.8821, 147.3272, 'Hobart'),
    'shenzhen': (22.5431, 114.0579, 'Shenzhen'),
    'hua hin': (12.5684, 99.9577, 'Hua Hin'),
    'thailand': (12.5684, 99.9577, 'Hua Hin'),
    'monterrey': (25.6866, -100.3161, 'Monterrey'),
    'bogota': (4.7110, -74.0721, 'Bogota'),
    'lyon': (45.7640, 4.8357, 'Lyon'),
    'rabat': (34.0209, -6.8416, 'Rabat'),
    'morocco': (34.0209, -6.8416, 'Rabat'),
    'strasbourg': (48.5734, 7.7521, 'Strasbourg'),
    'parma': (44.8015, 10.3279, 'Parma'),
    'nottingham': (52.9548, -1.1581, 'Nottingham'),
    'palermo': (38.1157, 13.3615, 'Palermo'),
    'lausanne': (46.5197, 6.6323, 'Lausanne'),
    'budapest': (47.4979, 19.0402, 'Budapest'),
    'prague': (50.0755, 14.4378, 'Prague'),
    'washington': (38.9072, -77.0369, 'Washington'),
    'cleveland': (41.4993, -81.6944, 'Cleveland'),
    'granby': (45.4001, -72.7334, 'Granby'),
    'chicago': (41.8781, -87.6298, 'Chicago'),
    'portoroz': (45.5167, 13.5833, 'Portoroz'),
    'slovenia': (45.5167, 13.5833, 'Portoroz'),
    'monastir': (35.7643, 10.8113, 'Monastir'),
    'tunisia': (35.7643, 10.8113, 'Monastir'),
    'ningbo': (29.8683, 121.5440, 'Ningbo'),
    'zhengzhou': (34.7466, 113.6254, 'Zhengzhou'),
    'nanchang': (28.6820, 115.8579, 'Nanchang'),
    'jiujiang': (29.7051, 116.0019, 'Jiujiang'),
    'guangzhou': (23.1291, 113.2644, 'Guangzhou'),
    'hong kong': (22.3193, 114.1694, 'Hong Kong'),
    'taipei': (25.0330, 121.5654, 'Taipei'),
    'tenerife': (28.2916, -16.6291, 'Tenerife'),
    'merida': (20.9674, -89.5926, 'Merida'),
    'courmayeur': (45.7967, 6.9686, 'Courmayeur'),
    'cluj': (46.7712, 23.6236, 'Cluj'),
    'transylvania': (46.7712, 23.6236, 'Cluj'),
    'moscow': (55.7558, 37.6173, 'Moscow'),
    
    # ITF / Challengers (common ones)
    'indian harbour': (28.1461, -80.6712, 'Indian Harbour Beach'),
    'midland': (31.9973, -102.0779, 'Midland'),
    'tyler': (32.3513, -95.3011, 'Tyler'),
    'antalya': (36.8969, 30.7133, 'Antalya'),
    'cairo': (30.0444, 31.2357, 'Cairo'),
    'canberra': (-35.2809, 149.1300, 'Canberra'),
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth (in km).
    """
    R = 6371  # Earth's radius in km
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


def get_tournament_location(tournament_name: str) -> tuple:
    """
    Get location for a tournament.
    Returns (lat, lon, city) or None if not found.
    """
    if not tournament_name:
        return None
    
    name_lower = tournament_name.lower().strip()
    
    # Try exact match first
    if name_lower in TOURNAMENT_LOCATIONS:
        return TOURNAMENT_LOCATIONS[name_lower]
    
    # Try partial match
    for key, location in TOURNAMENT_LOCATIONS.items():
        if key in name_lower or name_lower in key:
            return location
    
    return None


def calculate_travel_distance(tournament1: str, tournament2: str) -> float:
    """
    Calculate travel distance between two tournaments in km.
    Returns None if either location is unknown.
    """
    loc1 = get_tournament_location(tournament1)
    loc2 = get_tournament_location(tournament2)
    
    if loc1 is None or loc2 is None:
        return None
    
    return haversine_distance(loc1[0], loc1[1], loc2[0], loc2[1])


def get_travel_fatigue_factor(distance_km: float) -> float:
    """
    Convert travel distance to a fatigue factor.
    
    Returns a value 0-1 representing fatigue impact:
    - 0: No travel fatigue
    - 1: Maximum travel fatigue
    
    Based on research:
    - Short haul (<1000km): Minimal impact
    - Medium (1000-5000km): Moderate impact
    - Long haul (5000-10000km): Significant impact  
    - Ultra long (>10000km): Major impact + jet lag
    """
    if distance_km is None:
        return 0.0
    
    if distance_km < 500:
        return 0.0  # Same region, negligible
    elif distance_km < 1500:
        return 0.1  # Short flight
    elif distance_km < 4000:
        return 0.2  # Medium haul
    elif distance_km < 8000:
        return 0.4  # Long haul
    elif distance_km < 12000:
        return 0.6  # Very long haul (e.g., Europe to Asia)
    else:
        return 0.8  # Ultra long (e.g., Australia to Europe)


if __name__ == '__main__':
    # Test some distances
    print("=== TRAVEL DISTANCE TESTS ===")
    
    tests = [
        ('Australian Open', 'Dubai'),
        ('Miami', 'Indian Wells'),
        ('French Open', 'Wimbledon'),
        ('US Open', 'Beijing'),
        ('Brisbane', 'Australian Open'),
        ('Madrid', 'Rome'),
    ]
    
    for t1, t2 in tests:
        dist = calculate_travel_distance(t1, t2)
        if dist:
            fatigue = get_travel_fatigue_factor(dist)
            loc1 = get_tournament_location(t1)
            loc2 = get_tournament_location(t2)
            print(f"{loc1[2]:15} -> {loc2[2]:15}: {dist:,.0f} km (fatigue: {fatigue:.1f})")
        else:
            print(f"{t1} -> {t2}: Unknown location")
