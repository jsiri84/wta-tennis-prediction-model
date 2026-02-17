"""Explore 3 potential features: Momentum, H2H, Surface Specialist"""

import json
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load data
with open('all_players_matches.json', 'r') as f:
    data = json.load(f)

player_data = {}
for name, info in data['players'].items():
    player_data[name] = {
        'elo': info.get('elo_overall', 1500),
        'elo_hard': info.get('elo_hard', 1500),
        'elo_clay': info.get('elo_clay', 1500),
        'elo_grass': info.get('elo_grass', 1500),
        'matches': sorted(info.get('matches', []), key=lambda x: x.get('date', ''), reverse=True)
    }

name_lookup = {}
for name in player_data:
    name_lookup[name.lower()] = name
    name_lookup[name.lower().replace(' ', '')] = name
    parts = name.split()
    if len(parts) > 1:
        name_lookup[parts[-1].lower()] = name


def parse_score(score_str):
    """Parse score to get spread"""
    if not score_str:
        return None
    try:
        sets = score_str.replace('/', ' ').split()
        p_games, o_games = 0, 0
        for s in sets:
            if '-' in s:
                parts = s.split('-')
                p_games += int(parts[0].split('(')[0])
                o_games += int(parts[1].split('(')[0])
        return p_games - o_games
    except:
        return None


def get_opp_elo(match, surface='Hard'):
    opp = match.get('opponent', '')
    key = name_lookup.get(opp.lower()) or name_lookup.get(opp.lower().replace(' ', ''))
    if not key:
        last = opp.split()[-1].lower() if opp else ''
        key = name_lookup.get(last)
    if key and key in player_data:
        return player_data[key].get(f'elo_{surface.lower()}', player_data[key]['elo'])
    return None


# =============================================================================
# FEATURE 1: RECENT MOMENTUM (last 3-5 matches)
# =============================================================================
def calc_momentum(matches, window=3):
    """
    Recent momentum: win rate and performance in last N matches.
    Captures hot/cold streaks.
    """
    recent = matches[:window]
    if len(recent) < window:
        return None
    
    wins = sum(1 for m in recent if m['result'] == 'W')
    
    # Also get average spread
    spreads = []
    for m in recent:
        spread = parse_score(m.get('score', ''))
        if spread is not None:
            if m['result'] == 'L':
                spread = -spread
            spreads.append(spread)
    
    # Streak: consecutive W or L
    streak = 0
    if recent[0]['result'] == 'W':
        for m in recent:
            if m['result'] == 'W':
                streak += 1
            else:
                break
    else:
        for m in recent:
            if m['result'] == 'L':
                streak -= 1
            else:
                break
    
    return {
        'win_rate': wins / window,
        'avg_spread': np.mean(spreads) if spreads else 0,
        'streak': streak,
    }


# =============================================================================
# FEATURE 2: HEAD-TO-HEAD
# =============================================================================
def calc_h2h(matches, opponent_name):
    """
    H2H record against specific opponent.
    """
    opp_lower = opponent_name.lower()
    opp_last = opponent_name.split()[-1].lower() if opponent_name else ''
    
    wins, losses = 0, 0
    spreads = []
    
    for m in matches:
        match_opp = m.get('opponent', '').lower()
        if opp_last in match_opp or opp_lower.replace(' ', '') in match_opp.replace(' ', ''):
            if m['result'] == 'W':
                wins += 1
                spread = parse_score(m.get('score', ''))
                if spread:
                    spreads.append(spread)
            else:
                losses += 1
                spread = parse_score(m.get('score', ''))
                if spread:
                    spreads.append(-spread)
    
    total = wins + losses
    if total == 0:
        return None
    
    return {
        'wins': wins,
        'losses': losses,
        'win_rate': wins / total,
        'dominance': (wins - losses) / total,  # -1 to +1
        'avg_spread': np.mean(spreads) if spreads else 0,
    }


# =============================================================================
# FEATURE 3: SURFACE SPECIALIST
# =============================================================================
def calc_surface_specialist(matches, player_elo, surface_elo, surface, lookback=20):
    """
    Surface specialist: how much better/worse on this surface vs overall.
    Uses actual results, not just ELO.
    """
    surface_matches = [m for m in matches[:lookback] if m.get('surface', '').lower() == surface.lower()]
    other_matches = [m for m in matches[:lookback] if m.get('surface', '').lower() != surface.lower()]
    
    if len(surface_matches) < 3 or len(other_matches) < 3:
        return None
    
    # Win rate on this surface vs other surfaces
    surface_wins = sum(1 for m in surface_matches if m['result'] == 'W')
    other_wins = sum(1 for m in other_matches if m['result'] == 'W')
    
    surface_wr = surface_wins / len(surface_matches)
    other_wr = other_wins / len(other_matches)
    
    # Also look at ELO gap
    elo_gap = surface_elo - player_elo  # Positive = better on this surface
    
    return {
        'surface_wr': surface_wr,
        'other_wr': other_wr,
        'wr_advantage': surface_wr - other_wr,  # Positive = surface specialist
        'elo_surface_boost': elo_gap,
    }


# =============================================================================
# BUILD AND TEST
# =============================================================================
print("="*70)
print("EXPLORING 3 FEATURES: MOMENTUM, H2H, SURFACE SPECIALIST")
print("="*70)

# Build base data first
X_base = []
y = []
match_data = []

for pname, pdata in player_data.items():
    for i, match in enumerate(pdata['matches']):
        if i < 5 or match.get('result') not in ('W', 'L'):
            continue
        
        opp = match.get('opponent', '')
        okey = name_lookup.get(opp.lower()) or name_lookup.get(opp.lower().replace(' ', ''))
        if not okey:
            last = opp.split()[-1].lower() if opp else ''
            okey = name_lookup.get(last)
        if not okey or okey not in player_data:
            continue
        
        opp_data = player_data[okey]
        surface = match.get('surface', 'Hard')
        if surface not in ['Hard', 'Clay', 'Grass']:
            surface = 'Hard'
        
        p1_elo = pdata.get(f'elo_{surface.lower()}', pdata['elo'])
        p2_elo = opp_data.get(f'elo_{surface.lower()}', opp_data['elo'])
        
        p1_matches = pdata['matches'][i+1:]
        p2_matches = opp_data['matches']
        
        X_base.append([p1_elo - p2_elo])
        y.append(1 if match['result'] == 'W' else 0)
        match_data.append({
            'p1': pname, 'p2': okey, 'surface': surface,
            'p1_matches': p1_matches, 'p2_matches': p2_matches,
            'p1_elo': pdata['elo'], 'p2_elo': opp_data['elo'],
            'p1_surf_elo': p1_elo, 'p2_surf_elo': p2_elo,
        })

X_base = np.array(X_base)
y = np.array(y)

print(f"\nTotal samples: {len(y)}")

# Baseline
scaler = StandardScaler()
X_base_scaled = scaler.fit_transform(X_base)
model = LogisticRegression(max_iter=1000, C=0.5)
scores_base = cross_val_score(model, X_base_scaled, y, cv=5)
print(f"\nBaseline (surface ELO diff): {scores_base.mean()*100:.1f}% (+/- {scores_base.std()*100:.1f}%)")


# -----------------------------------------------------------------------------
# TEST MOMENTUM
# -----------------------------------------------------------------------------
print("\n" + "-"*70)
print("1. MOMENTUM (recent form)")
print("-"*70)

for window in [3, 5]:
    X_mom = []
    y_mom = []
    
    for i, md in enumerate(match_data):
        p1_mom = calc_momentum(md['p1_matches'], window)
        p2_mom = calc_momentum(md['p2_matches'], window)
        
        if p1_mom and p2_mom:
            X_mom.append([
                X_base[i][0],  # ELO diff
                p1_mom['win_rate'] - p2_mom['win_rate'],
                p1_mom['streak'] - p2_mom['streak'],
            ])
            y_mom.append(y[i])
    
    X_mom = np.array(X_mom)
    y_mom = np.array(y_mom)
    X_mom_scaled = StandardScaler().fit_transform(X_mom)
    
    scores = cross_val_score(model, X_mom_scaled, y_mom, cv=5)
    
    # Get coefficients
    model.fit(X_mom_scaled, y_mom)
    
    print(f"\nWindow = {window} matches (n={len(y_mom)})")
    print(f"  Accuracy: {scores.mean()*100:.1f}% (+/- {scores.std()*100:.1f}%)")
    print(f"  Coefficients:")
    for name, coef in zip(['elo_diff', 'momentum_wr_diff', 'streak_diff'], model.coef_[0]):
        print(f"    {name:<20} {coef:+.4f}")


# -----------------------------------------------------------------------------
# TEST H2H
# -----------------------------------------------------------------------------
print("\n" + "-"*70)
print("2. HEAD-TO-HEAD")
print("-"*70)

X_h2h = []
y_h2h = []
h2h_found = 0

for i, md in enumerate(match_data):
    p1_h2h = calc_h2h(md['p1_matches'], md['p2'])
    
    if p1_h2h:
        h2h_found += 1
        X_h2h.append([
            X_base[i][0],  # ELO diff
            p1_h2h['dominance'],  # -1 to +1
            min(p1_h2h['wins'] + p1_h2h['losses'], 5) / 5,  # H2H sample size (capped at 5)
        ])
        y_h2h.append(y[i])
    else:
        # No H2H - use baseline
        X_h2h.append([X_base[i][0], 0, 0])
        y_h2h.append(y[i])

X_h2h = np.array(X_h2h)
y_h2h = np.array(y_h2h)
X_h2h_scaled = StandardScaler().fit_transform(X_h2h)

scores = cross_val_score(model, X_h2h_scaled, y_h2h, cv=5)
model.fit(X_h2h_scaled, y_h2h)

print(f"\nMatches with H2H data: {h2h_found} ({h2h_found/len(y)*100:.1f}%)")
print(f"Accuracy: {scores.mean()*100:.1f}% (+/- {scores.std()*100:.1f}%)")
print(f"Coefficients:")
for name, coef in zip(['elo_diff', 'h2h_dominance', 'h2h_sample_size'], model.coef_[0]):
    print(f"  {name:<20} {coef:+.4f}")


# -----------------------------------------------------------------------------
# TEST SURFACE SPECIALIST
# -----------------------------------------------------------------------------
print("\n" + "-"*70)
print("3. SURFACE SPECIALIST")
print("-"*70)

X_surf = []
y_surf = []

for i, md in enumerate(match_data):
    p1_spec = calc_surface_specialist(md['p1_matches'], md['p1_elo'], md['p1_surf_elo'], md['surface'])
    p2_spec = calc_surface_specialist(md['p2_matches'], md['p2_elo'], md['p2_surf_elo'], md['surface'])
    
    if p1_spec and p2_spec:
        X_surf.append([
            X_base[i][0],  # ELO diff
            p1_spec['wr_advantage'] - p2_spec['wr_advantage'],  # Surface specialist diff
            p1_spec['elo_surface_boost'] - p2_spec['elo_surface_boost'],  # ELO boost diff
        ])
        y_surf.append(y[i])

X_surf = np.array(X_surf)
y_surf = np.array(y_surf)
X_surf_scaled = StandardScaler().fit_transform(X_surf)

scores = cross_val_score(model, X_surf_scaled, y_surf, cv=5)
model.fit(X_surf_scaled, y_surf)

print(f"\nSamples with surface data: {len(y_surf)}")
print(f"Accuracy: {scores.mean()*100:.1f}% (+/- {scores.std()*100:.1f}%)")
print(f"Coefficients:")
for name, coef in zip(['elo_diff', 'surface_wr_adv_diff', 'elo_boost_diff'], model.coef_[0]):
    print(f"  {name:<20} {coef:+.4f}")


# -----------------------------------------------------------------------------
# TEST ALL TOGETHER
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("COMBINED: ELO + MOMENTUM + H2H")
print("="*70)

X_all = []
y_all = []

for i, md in enumerate(match_data):
    p1_mom = calc_momentum(md['p1_matches'], 3)
    p2_mom = calc_momentum(md['p2_matches'], 3)
    p1_h2h = calc_h2h(md['p1_matches'], md['p2'])
    
    if p1_mom and p2_mom:
        h2h_dom = p1_h2h['dominance'] if p1_h2h else 0
        
        X_all.append([
            X_base[i][0],  # Surface ELO diff
            p1_mom['win_rate'] - p2_mom['win_rate'],  # Momentum
            p1_mom['streak'] - p2_mom['streak'],  # Streak
            h2h_dom,  # H2H dominance
        ])
        y_all.append(y[i])

X_all = np.array(X_all)
y_all = np.array(y_all)
X_all_scaled = StandardScaler().fit_transform(X_all)

scores = cross_val_score(model, X_all_scaled, y_all, cv=5)
model.fit(X_all_scaled, y_all)

print(f"\nSamples: {len(y_all)}")
print(f"Accuracy: {scores.mean()*100:.1f}% (+/- {scores.std()*100:.1f}%)")
print(f"\nCoefficients:")
for name, coef in zip(['surface_elo_diff', 'momentum_diff', 'streak_diff', 'h2h_dominance'], model.coef_[0]):
    print(f"  {name:<20} {coef:+.4f}")


# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
Feature                      Accuracy    vs Baseline
-----------------------------------------------
Baseline (ELO only)          {scores_base.mean()*100:.1f}%       -
+ Momentum (3-match)         See above
+ H2H                        See above  
+ Surface Specialist         See above
Combined (ELO+Mom+H2H)       {scores.mean()*100:.1f}%
""")
