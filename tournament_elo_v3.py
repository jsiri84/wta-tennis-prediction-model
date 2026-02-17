"""
Tournament ELO v3: Focus on IMMEDIATE tournament momentum

Simpler approach:
1. Look at LAST match only (most recent form)
2. Compare: did you just beat someone BETTER than current opponent?
3. How dominant was that win?

Key insight: If you beat Paolini (2013) then face Putintseva (1788),
you have a "step down" of 225 ELO points -> easier match expected
"""

import json
import re
import numpy as np
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error


def load_data():
    with open('all_players_matches.json', 'r') as f:
        return json.load(f)


def calculate_immediate_momentum(matches, player_data, name_lookup, current_opp_elo, base_elo):
    """
    Calculate immediate tournament momentum from LAST match.
    
    Returns metrics comparing last opponent to current opponent.
    """
    if not matches:
        return None
    
    last_match = matches[0]
    
    opp_name = last_match.get('opponent', '')
    opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
    
    if not opp_key or opp_key not in player_data:
        return None
    
    last_opp_elo = player_data[opp_key].get('elo', 1500)
    won_last = last_match.get('result') == 'W'
    
    score = last_match.get('score', '')
    sets = re.findall(r'(\d+)-(\d+)', score)
    if not sets:
        return None
    
    p1_games = sum(int(s[0]) for s in sets)
    p2_games = sum(int(s[1]) for s in sets)
    last_spread = p1_games - p2_games if won_last else p2_games - p1_games
    
    # Count clean sets (6-0, 6-1, 6-2)
    if won_last:
        clean_sets = sum(1 for s in sets if int(s[0]) >= 6 and int(s[1]) <= 2 and int(s[0]) > int(s[1]))
    else:
        clean_sets = -sum(1 for s in sets if int(s[1]) >= 6 and int(s[0]) <= 2 and int(s[1]) > int(s[0]))
    
    # Key metrics
    # 1. Step down: How much easier is current opponent vs last?
    #    Positive = step down (current is easier)
    step_down = last_opp_elo - current_opp_elo
    
    # 2. Scalp factor: Did you beat someone above your level?
    scalp_factor = last_opp_elo - base_elo if won_last else base_elo - last_opp_elo
    
    # 3. Combined: Big scalp + step down = confidence cascade
    confidence_cascade = 0
    if won_last and scalp_factor > 0 and step_down > 50:
        # Beat someone above your level, now facing easier opponent
        confidence_cascade = (scalp_factor / 100) * (step_down / 100)
    
    return {
        'won_last': 1 if won_last else -1,
        'last_opp_elo': last_opp_elo,
        'last_spread': last_spread,
        'step_down': step_down if won_last else -step_down,  # Negative if lost
        'scalp_factor': scalp_factor,
        'confidence_cascade': confidence_cascade,
        'clean_sets': clean_sets,
        
        # Interaction: dominant win + step down
        'dominant_step_down': (last_spread * step_down / 100) if won_last and step_down > 0 else 0,
    }


def build_data(data, include_momentum=False):
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
    y_total = []
    
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
            
            if include_momentum:
                p1_mom = calculate_immediate_momentum(historical, player_data, name_lookup, 
                                                       opp_data['elo'], pdata['elo'])
                p2_mom = calculate_immediate_momentum(opp_hist, player_data, name_lookup,
                                                       pdata['elo'], opp_data['elo'])
                
                if p1_mom and p2_mom:
                    features.extend([
                        # Step down differential
                        p1_mom['step_down'] - p2_mom['step_down'],
                        
                        # Scalp factor (did you beat someone above your level?)
                        p1_mom['scalp_factor'] - p2_mom['scalp_factor'],
                        
                        # Last spread
                        p1_mom['last_spread'] - p2_mom['last_spread'],
                        
                        # Confidence cascade
                        p1_mom['confidence_cascade'] - p2_mom['confidence_cascade'],
                        
                        # Dominant step down interaction
                        p1_mom['dominant_step_down'] - p2_mom['dominant_step_down'],
                    ])
                else:
                    features.extend([0, 0, 0, 0, 0])
            
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
            y_total.append(total)
    
    return np.array(X), np.array(y), np.array(y_spread), np.array(y_total)


def evaluate(X, y, y_spread, y_total):
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
    
    total_model = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    total_pred = cross_val_predict(total_model, X_scaled, y_total, cv=kf)
    total_mae = mean_absolute_error(y_total, total_pred)
    
    return winner_acc, winner_auc, spread_mae, total_mae


def main():
    print("="*70)
    print("IMMEDIATE TOURNAMENT MOMENTUM (v3)")
    print("="*70)
    print()
    print("Focus: Compare LAST opponent to CURRENT opponent")
    print("- Step down: Is current opponent easier than last?")
    print("- Scalp: Did you beat someone above your level?")
    print("- Confidence cascade: Big scalp + step down = blowout potential")
    print()
    
    data = load_data()
    print(f"Loaded {data['player_count']} players")
    print()
    
    print("Building BASELINE...")
    X_base, y, y_spread, y_total = build_data(data, include_momentum=False)
    print(f"  {X_base.shape[0]} samples, {X_base.shape[1]} features")
    base_acc, base_auc, base_spread, base_total = evaluate(X_base, y, y_spread, y_total)
    
    print("\nBuilding WITH IMMEDIATE MOMENTUM...")
    X_new, y, y_spread, y_total = build_data(data, include_momentum=True)
    print(f"  {X_new.shape[0]} samples, {X_new.shape[1]} features")
    new_acc, new_auc, new_spread, new_total = evaluate(X_new, y, y_spread, y_total)
    
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()
    print(f"{'Metric':<20} {'Baseline':>12} {'+ Momentum':>12} {'Change':>12}")
    print("-"*60)
    print(f"{'Winner Accuracy':<20} {base_acc:>11.1%} {new_acc:>11.1%} {(new_acc-base_acc)*100:>+11.2f}%")
    print(f"{'Winner AUC':<20} {base_auc:>11.3f} {new_auc:>11.3f} {new_auc-base_auc:>+11.4f}")
    print(f"{'Spread MAE':<20} {base_spread:>11.2f} {new_spread:>11.2f} {new_spread-base_spread:>+11.3f}")
    print(f"{'Total MAE':<20} {base_total:>11.2f} {new_total:>11.2f} {new_total-base_total:>+11.3f}")
    
    # Example for Jovic vs Putintseva
    print()
    print("="*70)
    print("EXAMPLE: JOVIC vs PUTINTSEVA (before R4)")
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
    
    # Note: In actual prediction, we'd use the state BEFORE the R4 match
    # The matches[0] is now the Putintseva match, so we use matches[1:] to simulate pre-R4
    j_hist = player_data['Iva Jovic']['matches'][1:]  # Skip the Putintseva result
    p_hist = player_data['Yulia Putintseva']['matches'][1:]  # Skip the Jovic result
    
    j_mom = calculate_immediate_momentum(j_hist, player_data, name_lookup, 1788, 1862)  # Facing Putin (1788)
    p_mom = calculate_immediate_momentum(p_hist, player_data, name_lookup, 1862, 1788)  # Facing Jovic (1862)
    
    print()
    print("JOVIC (ELO 1862) - Last match: Beat Paolini (2013) 6-2, 7-6")
    if j_mom:
        print(f"  Step down to Putintseva (1788): +{j_mom['step_down']:.0f} ELO easier")
        print(f"  Scalp factor (beat above level): +{j_mom['scalp_factor']:.0f}")
        print(f"  Confidence cascade: {j_mom['confidence_cascade']:.2f}")
        print(f"  Dominant step down: {j_mom['dominant_step_down']:.1f}")
    
    print()
    print("PUTINTSEVA (ELO 1788) - Last match: Beat Jacquemot (1718) 6-1, 6-2")
    if p_mom:
        print(f"  Step down to Jovic (1862): {p_mom['step_down']:+.0f} (step UP, harder match)")
        print(f"  Scalp factor (beat above level): {p_mom['scalp_factor']:+.0f}")
        print(f"  Confidence cascade: {p_mom['confidence_cascade']:.2f}")
        print(f"  Dominant step down: {p_mom['dominant_step_down']:.1f}")
    
    if j_mom and p_mom:
        print()
        print("DIFFERENTIALS (Jovic - Putintseva):")
        print(f"  Step down diff: {j_mom['step_down'] - p_mom['step_down']:+.0f}")
        print(f"  Scalp factor diff: {j_mom['scalp_factor'] - p_mom['scalp_factor']:+.0f}")
        print(f"  Confidence cascade diff: {j_mom['confidence_cascade'] - p_mom['confidence_cascade']:+.2f}")


if __name__ == '__main__':
    main()
