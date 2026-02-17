"""
Fatigue calculation for tennis players.
Factors in recent matches, court time, travel, and rest.
"""

from datetime import datetime, timedelta
from tournament_locations import calculate_travel_distance, get_travel_fatigue_factor

# Fatigue parameters (tunable)
MAX_FATIGUE_PENALTY = 0.08  # Maximum win probability reduction (8%)

# Thresholds
BACK_TO_BACK_DAYS = 10  # Days between tournaments to count as "back-to-back"
RECENT_DAYS = 7         # Days to look back for recent match load
MIN_TRAVEL_KM = 3000    # Minimum travel distance to apply fatigue


def parse_date(date_str: str) -> datetime:
    """Parse date from YYYYMMDD format."""
    if not date_str:
        return None
    try:
        return datetime.strptime(str(date_str), '%Y%m%d')
    except:
        return None


# Estimate days offset from tournament start for each round (Grand Slam)
# This helps when data only has tournament start dates
ROUND_DAY_OFFSET = {
    'R128': 0,   # Day 1-2
    'R64': 2,    # Day 3-4
    'R32': 4,    # Day 5-6
    'R16': 6,    # Day 7-8 (2nd week)
    'QF': 8,     # Day 9-10
    'SF': 10,    # Day 11-12
    'F': 12,     # Day 13-14
    # Non-Grand Slam rounds
    'Q1': -3,
    'Q2': -2,
    'Q3': -1,
    'RR': 0,     # Round robin
}


def estimate_match_date(tournament_date: str, round_name: str) -> datetime:
    """
    Estimate actual match date from tournament date and round.
    Useful when data only has tournament start dates.
    """
    base_date = parse_date(tournament_date)
    if base_date is None:
        return None
    
    # Get day offset for this round
    round_upper = round_name.upper() if round_name else ''
    offset = ROUND_DAY_OFFSET.get(round_upper, 0)
    
    return base_date + timedelta(days=offset)


def count_sets(score: str) -> int:
    """Count number of sets in a match score."""
    if not score:
        return 2
    # Count set scores (separated by spaces)
    sets = score.strip().split()
    return len([s for s in sets if '-' in s])


def calculate_fatigue(matches: list, match_date: str, current_tournament: str = None, 
                      current_round: str = None) -> dict:
    """
    Calculate fatigue factors for a player before a match.
    
    Args:
        matches: List of recent matches (most recent first)
        match_date: Date of the upcoming match (YYYYMMDD)
        current_tournament: Name of current tournament
        current_round: Round of current match (for date estimation)
    
    Returns:
        dict with fatigue metrics and overall penalty
    """
    # Estimate actual match date if round is provided
    if current_round:
        target_date = estimate_match_date(match_date, current_round)
    else:
        target_date = parse_date(match_date)
    
    if target_date is None:
        return {'fatigue_penalty': 0.0, 'details': {}, 'penalties': {}}
    
    # Initialize metrics
    matches_last_7_days = 0
    minutes_last_7_days = 0
    three_setters_last_7_days = 0
    days_since_last_match = None
    travel_distance = None
    prev_tournament = None
    
    for match in matches:
        # Estimate actual match date using round
        match_round = match.get('round', '')
        match_tournament_date = match.get('date')
        match_dt = estimate_match_date(match_tournament_date, match_round)
        
        if match_dt is None:
            continue
        
        days_ago = (target_date - match_dt).days
        
        # Skip future matches or same-day matches
        if days_ago < 0:
            continue
        
        # Track days since last match
        if days_since_last_match is None:
            days_since_last_match = days_ago
            prev_tournament = match.get('tournament')
        
        # Count matches in last 7 days
        if days_ago <= RECENT_DAYS:
            matches_last_7_days += 1
            
            # Add court time
            time_mins = match.get('time_mins')
            if time_mins:
                try:
                    minutes_last_7_days += int(time_mins)
                except:
                    pass
            
            # Count 3-setters
            if count_sets(match.get('score', '')) >= 3:
                three_setters_last_7_days += 1
    
    # Calculate travel fatigue (only if back-to-back tournaments)
    travel_fatigue = 0.0
    if (days_since_last_match is not None and 
        days_since_last_match <= BACK_TO_BACK_DAYS and
        prev_tournament and current_tournament and
        prev_tournament.lower() != current_tournament.lower()):
        
        travel_distance = calculate_travel_distance(prev_tournament, current_tournament)
        if travel_distance:
            travel_fatigue = get_travel_fatigue_factor(travel_distance)
    
    # Calculate component penalties
    # 1. Match load penalty (0-2% for 0-7 matches)
    match_load_penalty = min(matches_last_7_days / 7, 1.0) * 0.02
    
    # 2. Court time penalty (0-2% for 0-10 hours on court)
    court_time_penalty = min(minutes_last_7_days / 600, 1.0) * 0.02
    
    # 3. Three-setter penalty (1% per 3-setter in last 7 days)
    three_set_penalty = min(three_setters_last_7_days * 0.01, 0.03)
    
    # 4. No rest penalty (if played yesterday or day before)
    rest_penalty = 0.0
    if days_since_last_match == 0:
        rest_penalty = 0.02  # Same day (rare)
    elif days_since_last_match == 1:
        rest_penalty = 0.01  # Played yesterday
    
    # 5. Travel penalty (only for back-to-back)
    travel_penalty = travel_fatigue * 0.03  # Up to 2.4% for ultra-long travel
    
    # Total fatigue penalty (capped)
    total_penalty = min(
        match_load_penalty + court_time_penalty + three_set_penalty + rest_penalty + travel_penalty,
        MAX_FATIGUE_PENALTY
    )
    
    return {
        'fatigue_penalty': round(total_penalty, 4),
        'details': {
            'matches_last_7_days': matches_last_7_days,
            'minutes_last_7_days': minutes_last_7_days,
            'three_setters': three_setters_last_7_days,
            'days_since_last_match': days_since_last_match,
            'prev_tournament': prev_tournament,
            'travel_distance_km': round(travel_distance) if travel_distance else None,
            'travel_fatigue': round(travel_fatigue, 2),
        },
        'penalties': {
            'match_load': round(match_load_penalty, 4),
            'court_time': round(court_time_penalty, 4),
            'three_setters': round(three_set_penalty, 4),
            'rest': round(rest_penalty, 4),
            'travel': round(travel_penalty, 4),
        }
    }


def calculate_travel_fatigue_only(matches: list, match_date: str, current_tournament: str,
                                   current_round: str = None) -> dict:
    """
    Calculate only travel fatigue (no match load, court time, etc.).
    This simpler model has better backtested accuracy.
    
    Only applies if:
    1. Previous tournament was different from current
    2. Previous tournament was within BACK_TO_BACK_DAYS
    3. Travel distance > MIN_TRAVEL_KM
    """
    if not current_tournament:
        return {'fatigue_penalty': 0.0, 'details': {}, 'penalties': {}}
    
    # Estimate actual match date if round is provided
    if current_round:
        target_date = estimate_match_date(match_date, current_round)
    else:
        target_date = parse_date(match_date)
    
    if target_date is None:
        return {'fatigue_penalty': 0.0, 'details': {}, 'penalties': {}}
    
    # Find most recent match from DIFFERENT tournament
    prev_tournament = None
    days_since_prev_tournament = None
    travel_distance = None
    
    for match in matches:
        match_tournament = match.get('tournament', '')
        if not match_tournament:
            continue
        
        # Skip matches from current tournament
        if match_tournament.lower() == current_tournament.lower():
            continue
        
        # Found previous tournament
        prev_tournament = match_tournament
        match_round = match.get('round', '')
        match_dt = estimate_match_date(match.get('date', ''), match_round)
        
        if match_dt:
            days_since_prev_tournament = (target_date - match_dt).days
        
        # Calculate travel distance
        travel_distance = calculate_travel_distance(prev_tournament, current_tournament)
        break
    
    # Calculate travel penalty (only if back-to-back and significant distance)
    travel_penalty = 0.0
    travel_fatigue_factor = 0.0
    
    if (days_since_prev_tournament is not None and 
        0 < days_since_prev_tournament <= BACK_TO_BACK_DAYS and
        travel_distance and travel_distance > MIN_TRAVEL_KM):
        
        travel_fatigue_factor = get_travel_fatigue_factor(travel_distance)
        travel_penalty = travel_fatigue_factor * 0.03  # Up to ~2.4% for ultra-long
    
    return {
        'fatigue_penalty': round(travel_penalty, 4),
        'details': {
            'prev_tournament': prev_tournament,
            'days_since_prev_tournament': days_since_prev_tournament,
            'travel_distance_km': round(travel_distance) if travel_distance else None,
            'travel_fatigue': round(travel_fatigue_factor, 2),
        },
        'penalties': {
            'travel': round(travel_penalty, 4),
        }
    }


def get_fatigue_description(fatigue: dict) -> str:
    """Get human-readable fatigue description."""
    penalty = fatigue['fatigue_penalty']
    details = fatigue['details']
    
    if penalty < 0.01:
        return "Fresh (no fatigue)"
    
    factors = []
    
    if details.get('matches_last_7_days', 0) >= 4:
        factors.append(f"{details['matches_last_7_days']} matches in 7 days")
    
    if details.get('minutes_last_7_days', 0) >= 300:
        hours = details['minutes_last_7_days'] / 60
        factors.append(f"{hours:.1f}h on court")
    
    if details.get('three_setters', 0) >= 1:
        factors.append(f"{details['three_setters']} three-setter(s)")
    
    if details.get('days_since_last_match') is not None and details['days_since_last_match'] <= 1:
        factors.append("minimal rest")
    
    if details.get('travel_distance_km') and details['travel_distance_km'] > 3000:
        factors.append(f"{details['travel_distance_km']:,}km travel")
    
    if factors:
        return f"Fatigued ({', '.join(factors)})"
    elif penalty >= 0.03:
        return "Moderately fatigued"
    else:
        return "Slightly fatigued"


if __name__ == '__main__':
    # Test with sample data
    import json
    
    data = json.load(open('player_data.json'))
    
    # Test Sabalenka's fatigue going into a hypothetical match
    sabalenka = data['players']['Aryna Sabalenka']
    
    print("=== SABALENKA FATIGUE TEST ===")
    print("Matches:", len(sabalenka['matches']))
    
    # Simulate fatigue for a match at Australian Open (date: 20260119)
    fatigue = calculate_fatigue(
        sabalenka['matches'],
        match_date='20260120',
        current_tournament='Australian Open'
    )
    
    print(f"\nFatigue for match on 2026-01-20:")
    print(f"  Total penalty: {fatigue['fatigue_penalty']*100:.1f}%")
    print(f"  Details: {fatigue['details']}")
    print(f"  Status: {get_fatigue_description(fatigue)}")
    
    # Test after traveling from Brisbane
    print("\n--- If coming from Brisbane ---")
    fatigue2 = calculate_fatigue(
        sabalenka['matches'],
        match_date='20260110',
        current_tournament='Australian Open'
    )
    print(f"  Travel distance: {fatigue2['details'].get('travel_distance_km')} km")
    print(f"  Travel fatigue: {fatigue2['details'].get('travel_fatigue')}")
    print(f"  Total penalty: {fatigue2['fatigue_penalty']*100:.1f}%")
