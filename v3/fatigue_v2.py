"""
Advanced Fatigue Model v2
Based on professional tennis modeling approaches.

Key improvements:
1. Fatigue as STATE VARIABLE with decay
2. Load based on POINTS played (not just minutes)
3. Time zone penalties for travel
4. Layoff/rust penalty for extended breaks
5. Returns fatigue adjustment for ServeSkill (not win prob)
"""

from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
import numpy as np

# ============================================
# TOURNAMENT LOCATIONS (with timezones)
# ============================================

# Format: tournament_name -> (lat, lon, timezone_offset_utc, city)
TOURNAMENT_LOCATIONS = {
    # Grand Slams
    'australian open': (-37.8136, 144.9631, 11, 'Melbourne'),
    'french open': (48.8566, 2.3522, 2, 'Paris'),
    'roland garros': (48.8566, 2.3522, 2, 'Paris'),
    'wimbledon': (51.5074, -0.1278, 1, 'London'),
    'us open': (40.7128, -74.0060, -4, 'New York'),
    
    # Middle East / Asia
    'dubai': (25.2048, 55.2708, 4, 'Dubai'),
    'doha': (25.2854, 51.5310, 3, 'Doha'),
    'qatar': (25.2854, 51.5310, 3, 'Doha'),
    'abu dhabi': (24.4539, 54.3773, 4, 'Abu Dhabi'),
    'beijing': (39.9042, 116.4074, 8, 'Beijing'),
    'china open': (39.9042, 116.4074, 8, 'Beijing'),
    'wuhan': (30.5928, 114.3055, 8, 'Wuhan'),
    'tokyo': (35.6762, 139.6503, 9, 'Tokyo'),
    'seoul': (37.5665, 126.9780, 9, 'Seoul'),
    'hong kong': (22.3193, 114.1694, 8, 'Hong Kong'),
    'singapore': (1.3521, 103.8198, 8, 'Singapore'),
    
    # Australia
    'brisbane': (-27.4698, 153.0251, 10, 'Brisbane'),
    'adelaide': (-34.9285, 138.6007, 10.5, 'Adelaide'),
    'sydney': (-33.8688, 151.2093, 11, 'Sydney'),
    'hobart': (-42.8821, 147.3272, 11, 'Hobart'),
    
    # North America
    'indian wells': (33.7238, -116.3511, -7, 'Indian Wells'),
    'bnp paribas': (33.7238, -116.3511, -7, 'Indian Wells'),
    'miami': (25.7617, -80.1918, -4, 'Miami'),
    'charleston': (32.7765, -79.9311, -4, 'Charleston'),
    'san diego': (32.7157, -117.1611, -7, 'San Diego'),
    'cincinnati': (39.1031, -84.5120, -4, 'Cincinnati'),
    'toronto': (43.6532, -79.3832, -4, 'Toronto'),
    'montreal': (45.5017, -73.5673, -4, 'Montreal'),
    'washington': (38.9072, -77.0369, -4, 'Washington'),
    'san jose': (37.3382, -121.8863, -7, 'San Jose'),
    'guadalajara': (20.6597, -103.3496, -5, 'Guadalajara'),
    
    # Europe
    'madrid': (40.4168, -3.7038, 2, 'Madrid'),
    'rome': (41.9028, 12.4964, 2, 'Rome'),
    'italian open': (41.9028, 12.4964, 2, 'Rome'),
    'stuttgart': (48.7758, 9.1829, 2, 'Stuttgart'),
    'berlin': (52.5200, 13.4050, 2, 'Berlin'),
    'eastbourne': (50.7684, 0.2905, 1, 'Eastbourne'),
    'birmingham': (52.4862, -1.8904, 1, 'Birmingham'),
    'strasbourg': (48.5734, 7.7521, 2, 'Strasbourg'),
    'prague': (50.0755, 14.4378, 2, 'Prague'),
    'ostrava': (49.8209, 18.2625, 2, 'Ostrava'),
    'linz': (48.3069, 14.2858, 2, 'Linz'),
}

# ============================================
# PARAMETERS (tunable)
# ============================================

# Decay factor (how quickly fatigue recovers per day)
FATIGUE_DECAY = 0.6  # 60% of fatigue remains each day (faster recovery)

# Load weights (tuned for ~0.1-0.3 fatigue range)
POINTS_WEIGHT = 0.0005     # Per point played (reduced)
THREE_SET_BONUS = 0.1      # Extra load for 3-setter (reduced)
TRAVEL_WEIGHT = 0.00002    # Per km traveled (reduced)
TIMEZONE_WEIGHT = 0.05     # Per timezone crossed (increased for jet lag emphasis)

# Rest/Layoff
OPTIMAL_REST_DAYS = 3      # Ideal rest between matches
RUST_THRESHOLD = 30        # Days before rust kicks in
RUST_PENALTY_PER_DAY = 0.01  # Penalty per day over threshold

# Maximum fatigue (clamped)
MAX_FATIGUE = 1.0

# ServeSkill adjustment factor
FATIGUE_TO_SERVE_FACTOR = 0.3  # How much fatigue reduces ServeSkill (gives ~0.03-0.09 adj)


# ============================================
# HELPER FUNCTIONS
# ============================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


def get_tournament_info(name):
    """Get tournament location and timezone."""
    if not name:
        return None
    name_lower = name.lower().strip()
    
    # Try exact match
    if name_lower in TOURNAMENT_LOCATIONS:
        return TOURNAMENT_LOCATIONS[name_lower]
    
    # Try partial match
    for key, info in TOURNAMENT_LOCATIONS.items():
        if key in name_lower or name_lower in key:
            return info
    return None


def parse_date(date_str):
    """Parse YYYYMMDD date."""
    if not date_str:
        return None
    try:
        return datetime.strptime(str(date_str), '%Y%m%d')
    except:
        return None


# Round day offsets for estimating actual match dates
ROUND_OFFSETS = {
    'R128': 0, 'R64': 2, 'R32': 4, 'R16': 6,
    'QF': 8, 'SF': 10, 'F': 12,
    'Q1': -3, 'Q2': -2, 'Q3': -1, 'RR': 0,
}


def estimate_match_date(tournament_date, round_name):
    """Estimate actual match date from tournament date and round."""
    base = parse_date(tournament_date)
    if base is None:
        return None
    offset = ROUND_OFFSETS.get(round_name.upper() if round_name else '', 0)
    return base + timedelta(days=offset)


def count_sets(score):
    """Count sets in a score string."""
    if not score:
        return 2
    return len([s for s in score.split() if '-' in s])


def get_points_from_match(match):
    """Extract total points played from a match."""
    serve_raw = match.get('serve_raw', {})
    return_raw = match.get('return_raw', {})
    
    # Player's service points + return points
    serve_pts = serve_raw.get('pts', 0) or 0
    return_pts = return_raw.get('opts', 0) or 0
    
    return serve_pts + return_pts


# ============================================
# MAIN FATIGUE CALCULATOR
# ============================================

class FatigueCalculator:
    """
    Calculate fatigue as a state variable that accumulates and decays.
    
    Fatigue_t = decay × Fatigue_{t-1} + Load_{t-1}
    
    Where Load = points_weight × points + 3set_bonus × is_3set + travel + timezone
    """
    
    def __init__(self):
        self.decay = FATIGUE_DECAY
    
    def calculate_match_load(self, match, prev_tournament=None, current_tournament=None):
        """
        Calculate load from a single match.
        
        Load = points + 3-set bonus + travel + timezone shift
        """
        load = 0.0
        details = {}
        
        # 1. Points played
        points = get_points_from_match(match)
        if points > 0:
            load += POINTS_WEIGHT * points
            details['points'] = points
        else:
            # Fallback to minutes if no points data
            mins = match.get('time_mins')
            if mins:
                try:
                    load += 0.003 * int(mins)  # Roughly similar scale
                    details['minutes'] = int(mins)
                except:
                    pass
        
        # 2. Three-set bonus
        if count_sets(match.get('score', '')) >= 3:
            load += THREE_SET_BONUS
            details['three_set'] = True
        
        # 3. Travel (if tournament changed)
        match_tournament = match.get('tournament', '')
        if prev_tournament and match_tournament:
            if prev_tournament.lower() != match_tournament.lower():
                prev_info = get_tournament_info(prev_tournament)
                curr_info = get_tournament_info(match_tournament)
                
                if prev_info and curr_info:
                    # Distance
                    dist = haversine_distance(prev_info[0], prev_info[1], 
                                              curr_info[0], curr_info[1])
                    if dist > 1000:  # Significant travel
                        load += TRAVEL_WEIGHT * dist
                        details['travel_km'] = round(dist)
                    
                    # Timezone shift
                    tz_diff = abs(prev_info[2] - curr_info[2])
                    if tz_diff >= 3:  # Significant jet lag
                        load += TIMEZONE_WEIGHT * tz_diff
                        details['timezone_shift'] = tz_diff
        
        return load, details
    
    def calculate_fatigue_state(self, matches, target_date, current_tournament=None, 
                                 current_round=None, lookback_days=21):
        """
        Calculate cumulative fatigue state for a player.
        
        Iterates through recent matches, applying decay and accumulating load.
        
        Returns:
            fatigue: Float 0-1 representing fatigue level
            details: Dict with breakdown
        """
        if current_round:
            target_dt = estimate_match_date(target_date, current_round)
        else:
            target_dt = parse_date(target_date)
        
        if target_dt is None:
            return 0.0, {}
        
        # Collect matches in lookback window
        recent_matches = []
        for match in matches:
            match_round = match.get('round', '')
            match_dt = estimate_match_date(match.get('date', ''), match_round)
            if match_dt is None:
                continue
            
            days_ago = (target_dt - match_dt).days
            if 0 < days_ago <= lookback_days:
                recent_matches.append((days_ago, match))
        
        # Sort by days ago (most recent first for proper decay)
        recent_matches.sort(key=lambda x: x[0])
        
        # Calculate cumulative fatigue with decay
        fatigue = 0.0
        total_points = 0
        total_load = 0.0
        match_count = 0
        last_match_days = None
        prev_tournament = None
        
        # Process from oldest to newest for proper decay
        for days_ago, match in reversed(recent_matches):
            if last_match_days is None:
                last_match_days = days_ago
            
            # Calculate load for this match
            load, _ = self.calculate_match_load(match, prev_tournament, current_tournament)
            
            # Apply decay for days passed since last update
            if match_count > 0:
                days_between = prev_days - days_ago
                fatigue *= (self.decay ** days_between)
            
            # Add new load
            fatigue += load
            total_load += load
            total_points += get_points_from_match(match)
            match_count += 1
            
            prev_days = days_ago
            prev_tournament = match.get('tournament', '')
        
        # Apply final decay to current date
        if match_count > 0 and last_match_days:
            # Decay from most recent match to target date
            fatigue *= (self.decay ** recent_matches[0][0])
        
        # Clamp fatigue
        fatigue = min(fatigue, MAX_FATIGUE)
        
        # Calculate rust/layoff penalty
        rust_penalty = 0.0
        if last_match_days and last_match_days > RUST_THRESHOLD:
            rust_penalty = RUST_PENALTY_PER_DAY * (last_match_days - RUST_THRESHOLD)
            rust_penalty = min(rust_penalty, 0.3)  # Cap rust penalty
        
        # Check travel to current tournament
        travel_info = {}
        if prev_tournament and current_tournament:
            prev_info = get_tournament_info(prev_tournament)
            curr_info = get_tournament_info(current_tournament)
            if prev_info and curr_info and prev_tournament.lower() != current_tournament.lower():
                dist = haversine_distance(prev_info[0], prev_info[1], curr_info[0], curr_info[1])
                tz_diff = abs(prev_info[2] - curr_info[2])
                if dist > 1000 and last_match_days and last_match_days <= 10:
                    travel_info = {
                        'from': prev_tournament,
                        'distance_km': round(dist),
                        'timezone_shift': tz_diff,
                    }
        
        return fatigue, {
            'fatigue_raw': round(fatigue, 4),
            'rust_penalty': round(rust_penalty, 4),
            'matches_in_window': match_count,
            'total_points': total_points,
            'total_load': round(total_load, 4),
            'days_since_last_match': last_match_days,
            'travel': travel_info,
        }
    
    def get_serve_adjustment(self, fatigue, rust_penalty=0.0):
        """
        Convert fatigue to ServeSkill adjustment.
        
        ServeSkill_adj = ServeSkill - factor × (fatigue + rust)
        
        Returns negative adjustment to apply to ServeSkill.
        """
        total = fatigue + rust_penalty
        return -FATIGUE_TO_SERVE_FACTOR * total


def calculate_advanced_fatigue(matches, match_date, current_tournament=None, 
                                current_round=None):
    """
    Main entry point for advanced fatigue calculation.
    
    Returns dict with:
        - serve_adjustment: Adjustment to apply to ServeSkill
        - fatigue: Raw fatigue value
        - details: Breakdown of components
    """
    calc = FatigueCalculator()
    fatigue, details = calc.calculate_fatigue_state(
        matches, match_date, current_tournament, current_round
    )
    
    rust = details.get('rust_penalty', 0)
    serve_adj = calc.get_serve_adjustment(fatigue, rust)
    
    return {
        'serve_adjustment': round(serve_adj, 4),
        'fatigue': round(fatigue, 4),
        'rust_penalty': round(rust, 4),
        'total_penalty': round(fatigue + rust, 4),
        'details': details,
    }


# ============================================
# TESTING
# ============================================

if __name__ == '__main__':
    import json
    
    print("=== FATIGUE V2 TEST ===\n")
    
    # Test timezone differences
    print("Timezone Tests:")
    tests = [
        ('Australian Open', 'Dubai'),
        ('Miami', 'Indian Wells'),
        ('US Open', 'Beijing'),
        ('Rome', 'Paris'),
    ]
    for t1, t2 in tests:
        i1 = get_tournament_info(t1)
        i2 = get_tournament_info(t2)
        if i1 and i2:
            dist = haversine_distance(i1[0], i1[1], i2[0], i2[1])
            tz = abs(i1[2] - i2[2])
            print(f"  {i1[3]:15} -> {i2[3]:15}: {dist:,.0f}km, {tz}h timezone diff")
    
    # Test with real data
    print("\n\nPlayer Fatigue Test:")
    try:
        data = json.load(open('player_data.json'))
        
        for name in ['Aryna Sabalenka', 'Iga Swiatek', 'Madison Keys']:
            if name not in data['players']:
                continue
            
            player = data['players'][name]
            result = calculate_advanced_fatigue(
                player['matches'],
                match_date='20260120',
                current_tournament='Australian Open',
                current_round='F'
            )
            
            print(f"\n{name} (AO Final):")
            print(f"  Fatigue: {result['fatigue']:.3f}")
            print(f"  Rust: {result['rust_penalty']:.3f}")
            print(f"  ServeSkill adjustment: {result['serve_adjustment']:+.3f}")
            print(f"  Matches in window: {result['details'].get('matches_in_window', 0)}")
            print(f"  Points played: {result['details'].get('total_points', 0)}")
            if result['details'].get('travel'):
                t = result['details']['travel']
                print(f"  Travel: {t.get('distance_km')}km from {t.get('from')}, {t.get('timezone_shift')}h shift")
    except Exception as e:
        print(f"Error: {e}")
