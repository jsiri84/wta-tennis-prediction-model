"""Test 'performance at opponent's level' feature"""

import json
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load data
with open('all_players_matches.json', 'r') as f:
    data = json.load(f)

# Build player data with ELO
player_data = {}
for name, info in data['players'].items():
    player_data[name] = {
        'elo': info.get('elo_overall', 1500),
        'elo_hard': info.get('elo_hard', 1500),
        'elo_clay': info.get('elo_clay', 1500),
        'elo_grass': info.get('elo_grass', 1500),
        'matches': sorted(info.get('matches', []), key=lambda x: x.get('date', ''), reverse=True)
    }

# Name lookup
name_lookup = {}
for name in player_data:
    name_lookup[name.lower()] = name
    name_lookup[name.lower().replace(' ', '')] = name
    parts = name.split()
    if len(parts) > 1:
        name_lookup[parts[-1].lower()] = name


def get_opponent_elo(match, surface='Hard'):
    """Get opponent's ELO from match"""
    opp_name = match.get('opponent', '')
    opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
    if not opp_key:
        last = opp_name.split()[-1].lower() if opp_name else ''
        opp_key = name_lookup.get(last)
    
    if opp_key and opp_key in player_data:
        surface_key = f'elo_{surface.lower()}'
        return player_data[opp_key].get(surface_key, player_data[opp_key]['elo'])
    return None


def calc_performance_at_level(matches, target_elo, surface='Hard', elo_range=150, lookback=30):
    """
    Calculate how player performs against opponents near target_elo.
    
    Returns win rate and average spread against opponents within elo_range of target_elo.
    """
    wins = 0
    losses = 0
    spreads = []
    
    for m in matches[:lookback]:
        opp_elo = get_opponent_elo(m, surface)
        if opp_elo is None:
            continue
        
        # Check if opponent is within range of target
        if abs(opp_elo - target_elo) <= elo_range:
            if m['result'] == 'W':
                wins += 1
                # Calculate spread from score
                score = m.get('score', '')
                if score:
                    try:
                        sets = score.replace('/', ' ').split()
                        p_games = 0
                        o_games = 0
                        for s in sets:
                            if '-' in s:
                                parts = s.split('-')
                                p_games += int(parts[0].split('(')[0])
                                o_games += int(parts[1].split('(')[0])
                        spreads.append(p_games - o_games)
                    except:
                        pass
            else:
                losses += 1
                # Negative spread for losses
                score = m.get('score', '')
                if score:
                    try:
                        sets = score.replace('/', ' ').split()
                        p_games = 0
                        o_games = 0
                        for s in sets:
                            if '-' in s:
                                parts = s.split('-')
                                p_games += int(parts[0].split('(')[0])
                                o_games += int(parts[1].split('(')[0])
                        spreads.append(p_games - o_games)  # Will be negative for losses
                    except:
                        pass
    
    total = wins + losses
    if total == 0:
        return None
    
    return {
        'win_rate': wins / total,
        'matches': total,
        'avg_spread': np.mean(spreads) if spreads else 0,
    }


# Build training data
print("Building training data with 'performance at level' feature...")

X_base = []  # Base features (ELO + surface ELO)
X_level = []  # With level performance feature
y = []

for player_name, pdata in player_data.items():
    for i, match in enumerate(pdata['matches']):
        if i < 5:
            continue
        
        result = match.get('result')
        if result not in ('W', 'L'):
            continue
        
        # Find opponent
        opp_name = match.get('opponent', '')
        opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
        if not opp_key:
            last = opp_name.split()[-1].lower() if opp_name else ''
            opp_key = name_lookup.get(last)
        
        if not opp_key or opp_key not in player_data:
            continue
        
        opp_data = player_data[opp_key]
        
        surface = match.get('surface', 'Hard')
        if surface not in ['Hard', 'Clay', 'Grass']:
            surface = 'Hard'
        surface_key = f'elo_{surface.lower()}'
        
        # Get surface ELOs
        p1_elo = pdata.get(surface_key, pdata['elo'])
        p2_elo = opp_data.get(surface_key, opp_data['elo'])
        
        # Base features
        base_features = [
            pdata['elo'] - opp_data['elo'],
            p1_elo - p2_elo,
        ]
        
        # Performance at opponent's level
        p1_matches = pdata['matches'][i+1:]
        p2_matches = opp_data['matches']
        
        # P1's performance against opponents at P2's level
        p1_at_p2_level = calc_performance_at_level(p1_matches, p2_elo, surface)
        # P2's performance against opponents at P1's level
        p2_at_p1_level = calc_performance_at_level(p2_matches, p1_elo, surface)
        
        # Only include if we have data for both
        if p1_at_p2_level and p2_at_p1_level:
            level_features = base_features + [
                p1_at_p2_level['win_rate'] - 0.5,  # P1's win rate at P2's level (centered)
                p2_at_p1_level['win_rate'] - 0.5,  # P2's win rate at P1's level (centered)
                p1_at_p2_level['avg_spread'],       # P1's avg spread at P2's level
                p2_at_p1_level['avg_spread'],       # P2's avg spread at P1's level
            ]
            
            X_base.append(base_features)
            X_level.append(level_features)
            y.append(1 if result == 'W' else 0)

X_base = np.array(X_base)
X_level = np.array(X_level)
y = np.array(y)

print(f"Samples with level data: {len(y)}")
print()

# Scale and test
scaler_base = StandardScaler()
scaler_level = StandardScaler()
X_base_scaled = scaler_base.fit_transform(X_base)
X_level_scaled = scaler_level.fit_transform(X_level)

model = LogisticRegression(max_iter=1000, C=0.5)

# Test base (ELO only)
scores_base = cross_val_score(model, X_base_scaled, y, cv=5, scoring='accuracy')
print(f"ELO only (2 features):     {scores_base.mean()*100:.1f}% (+/- {scores_base.std()*100:.1f}%)")

# Test with level features
scores_level = cross_val_score(model, X_level_scaled, y, cv=5, scoring='accuracy')
print(f"ELO + level (6 features):  {scores_level.mean()*100:.1f}% (+/- {scores_level.std()*100:.1f}%)")

# Show feature importance
model.fit(X_level_scaled, y)
feature_names = [
    'elo_diff', 'surface_elo_diff',
    'p1_winrate_at_p2_level', 'p2_winrate_at_p1_level',
    'p1_spread_at_p2_level', 'p2_spread_at_p1_level',
]
print(f"\nFeature coefficients:")
for name, coef in zip(feature_names, model.coef_[0]):
    print(f"  {name:<25} {coef:+.4f}")

# Check if improvement is significant
improvement = scores_level.mean() - scores_base.mean()
print(f"\nImprovement: {improvement*100:+.2f}%")
