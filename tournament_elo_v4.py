"""
Tournament Momentum v4: Extreme Cases Only

Only fire the momentum feature when conditions are VERY strong:
- Beat someone 100+ ELO above you
- Now facing someone 150+ ELO below last opponent
- Won last match dominantly (spread 4+)

These rare cases might have predictive power that's being diluted
in the normal approach.
"""

import json
import re
import numpy as np
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error


def load_data():
    with open('all_players_matches.json', 'r') as f:
        return json.load(f)


def calculate_extreme_momentum(matches, player_data, name_lookup, current_opp_elo, base_elo):
    """
    Calculate momentum ONLY in extreme cases.
    
    Returns strong signal only when:
    1. Beat someone well above your level (scalp 100+)
    2. AND now facing someone much easier (step down 150+)
    3. AND won dominantly (spread 4+)
    """
    if not matches:
        return {'extreme_signal': 0, 'raw_cascade': 0}
    
    last_match = matches[0]
    opp_name = last_match.get('opponent', '')
    opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
    
    if not opp_key or opp_key not in player_data:
        return {'extreme_signal': 0, 'raw_cascade': 0}
    
    last_opp_elo = player_data[opp_key].get('elo', 1500)
    won_last = last_match.get('result') == 'W'
    
    if not won_last:
        return {'extreme_signal': 0, 'raw_cascade': 0}
    
    score = last_match.get('score', '')
    sets = re.findall(r'(\d+)-(\d+)', score)
    if not sets:
        return {'extreme_signal': 0, 'raw_cascade': 0}
    
    p1 = sum(int(s[0]) for s in sets)
    p2 = sum(int(s[1]) for s in sets)
    last_spread = p1 - p2
    
    # Key metrics
    scalp = last_opp_elo - base_elo  # Positive = beat someone above level
    step_down = last_opp_elo - current_opp_elo  # Positive = current is easier
    
    # Raw cascade (continuous)
    if scalp > 0 and step_down > 0:
        raw_cascade = (scalp / 100) * (step_down / 100) * (last_spread / 5)
    else:
        raw_cascade = 0
    
    # Extreme signal: only fires in strong cases
    extreme_signal = 0
    
    # Level 1: Big scalp (100+) + step down (150+)
    if scalp >= 100 and step_down >= 150:
        extreme_signal = 1
        
        # Level 2: Dominant win too (spread 4+)
        if last_spread >= 4:
            extreme_signal = 2
            
            # Level 3: Very big scalp (150+) + very big step down (200+)
            if scalp >= 150 and step_down >= 200:
                extreme_signal = 3
    
    return {
        'extreme_signal': extreme_signal,
        'raw_cascade': raw_cascade,
        'scalp': scalp,
        'step_down': step_down,
        'last_spread': last_spread,
    }


def build_data(data, include_extreme=False):
    """Build training data"""
    
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
    
    X = []
    y = []
    y_spread = []
    
    extreme_signals = []  # Track for analysis
    
    for player_name, pdata in player_data.items():
        matches = pdata['matches']
        
        for i, match in enumerate(matches):
            historical = matches[i+1:] if i+1 < len(matches) else []
            
            if len(historical) < 5:
                continue
            
            opp_name = match.get('opponent', '')
            opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
            
            if not opp_key or opp_key not in player_data:
                continue
            
            opp_data = player_data[opp_key]
            match_date = match.get('date', '')
            opp_hist = [m for m in opp_data['matches'] if m.get('date', '') < match_date]
            
            if len(opp_hist) < 5:
                continue
            
            surface = match.get('surface', 'Hard').lower()
            surface_key = f'elo_{surface}' if surface in ['hard', 'clay', 'grass'] else 'elo_hard'
            
            # Base features
            features = [
                pdata['elo'] - opp_data['elo'],
                pdata.get(surface_key, 1500) - opp_data.get(surface_key, 1500),
            ]
            
            if include_extreme:
                p1_ext = calculate_extreme_momentum(historical, player_data, name_lookup, opp_data['elo'], pdata['elo'])
                p2_ext = calculate_extreme_momentum(opp_hist, player_data, name_lookup, pdata['elo'], opp_data['elo'])
                
                features.extend([
                    p1_ext['extreme_signal'] - p2_ext['extreme_signal'],
                    p1_ext['raw_cascade'] - p2_ext['raw_cascade'],
                ])
                
                extreme_signals.append((p1_ext['extreme_signal'], p2_ext['extreme_signal']))
            
            # Labels
            score = match.get('score', '')
            sets = re.findall(r'(\d+)-(\d+)', score)
            if not sets:
                continue
            p1_games = sum(int(s[0]) for s in sets)
            p2_games = sum(int(s[1]) for s in sets)
            total = p1_games + p2_games
            spread = p1_games - p2_games
            
            if total < 12 or total > 50:
                continue
            
            X.append(features)
            y.append(1 if match.get('result') == 'W' else 0)
            y_spread.append(spread)
    
    return np.array(X), np.array(y), np.array(y_spread), extreme_signals


def evaluate(X, y, y_spread):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    winner_model = LogisticRegression(max_iter=1000, C=0.1)
    proba = cross_val_predict(winner_model, X_scaled, y, cv=kf, method='predict_proba')[:, 1]
    winner_acc = accuracy_score(y, (proba > 0.5).astype(int))
    winner_auc = roc_auc_score(y, proba)
    
    spread_model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    spread_pred = cross_val_predict(spread_model, X_scaled, y_spread, cv=kf)
    spread_mae = mean_absolute_error(y_spread, spread_pred)
    
    return winner_acc, winner_auc, spread_mae


def main():
    print("="*70)
    print("EXTREME MOMENTUM EXPERIMENT")
    print("="*70)
    print()
    print("Only fire momentum in EXTREME cases:")
    print("- Beat someone 100+ ELO above you")
    print("- Now facing someone 150+ ELO below last opponent")
    print("- Won last match dominantly (spread 4+)")
    print()
    
    data = load_data()
    print(f"Loaded {data['player_count']} players")
    print()
    
    print("Building BASELINE...")
    X_base, y, y_spread, _ = build_data(data, include_extreme=False)
    print(f"  {X_base.shape[0]} samples, {X_base.shape[1]} features")
    base_acc, base_auc, base_spread = evaluate(X_base, y, y_spread)
    
    print("\nBuilding WITH EXTREME MOMENTUM...")
    X_ext, y, y_spread, extreme_signals = build_data(data, include_extreme=True)
    print(f"  {X_ext.shape[0]} samples, {X_ext.shape[1]} features")
    ext_acc, ext_auc, ext_spread = evaluate(X_ext, y, y_spread)
    
    # Analyze extreme signals
    level1 = sum(1 for p1, p2 in extreme_signals if p1 >= 1 or p2 >= 1)
    level2 = sum(1 for p1, p2 in extreme_signals if p1 >= 2 or p2 >= 2)
    level3 = sum(1 for p1, p2 in extreme_signals if p1 >= 3 or p2 >= 3)
    
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()
    print(f"{'Metric':<20} {'Baseline':>12} {'+ Extreme':>12} {'Change':>12}")
    print("-"*60)
    print(f"{'Winner Accuracy':<20} {base_acc:>11.1%} {ext_acc:>11.1%} {(ext_acc-base_acc)*100:>+11.2f}%")
    print(f"{'Winner AUC':<20} {base_auc:>11.3f} {ext_auc:>11.3f} {ext_auc-base_auc:>+11.4f}")
    print(f"{'Spread MAE':<20} {base_spread:>11.2f} {ext_spread:>11.2f} {ext_spread-base_spread:>+11.3f}")
    
    print()
    print("="*70)
    print("EXTREME SIGNAL FREQUENCY")
    print("="*70)
    print(f"Total matches: {len(extreme_signals)}")
    print(f"Level 1+ (scalp 100+, step 150+): {level1} ({100*level1/len(extreme_signals):.1f}%)")
    print(f"Level 2+ (+ dominant spread 4+): {level2} ({100*level2/len(extreme_signals):.1f}%)")
    print(f"Level 3  (scalp 150+, step 200+, spread 4+): {level3} ({100*level3/len(extreme_signals):.1f}%)")
    
    # Example: Jovic
    print()
    print("="*70)
    print("EXAMPLE: JOVIC vs PUTINTSEVA")
    print("="*70)
    
    player_data = {}
    for name, info in data['players'].items():
        player_data[name] = {
            'elo': info.get('elo_overall', 1500),
            'matches': sorted(info.get('matches', []), key=lambda x: x.get('date', ''), reverse=True)
        }
    
    name_lookup = {}
    for name in player_data:
        name_lookup[name.lower()] = name
        name_lookup[name.lower().replace(' ', '')] = name
    
    j_hist = player_data['Iva Jovic']['matches'][1:]  # Before R4
    j_ext = calculate_extreme_momentum(j_hist, player_data, name_lookup, 1788, 1862)
    
    print()
    print("Jovic (ELO 1862):")
    print(f"  Last match: Beat Paolini (2013) by {j_ext['last_spread']:+d} spread")
    print(f"  Scalp: {j_ext['scalp']:+d} (beat someone {j_ext['scalp']} above level)")
    print(f"  Step down: {j_ext['step_down']:+d} (facing easier opponent)")
    print(f"  Extreme Signal Level: {j_ext['extreme_signal']}")
    print(f"  Raw Cascade: {j_ext['raw_cascade']:.2f}")
    
    if j_ext['extreme_signal'] >= 1:
        print()
        print("  => EXTREME MOMENTUM TRIGGERED!")
        if j_ext['extreme_signal'] >= 2:
            print("  => LEVEL 2: Dominant win too!")
        if j_ext['extreme_signal'] >= 3:
            print("  => LEVEL 3: MAXIMUM CONFIDENCE!")


if __name__ == '__main__':
    main()
