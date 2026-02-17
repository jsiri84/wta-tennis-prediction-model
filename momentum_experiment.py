"""
Experiment: Tournament Momentum / Confidence Cascade

If a player just beat an opponent with HIGHER ELO than their current opponent,
they may have a confidence boost leading to bigger wins.

Example: Jovic beat Paolini (2013 ELO), then faced Putintseva (1788 ELO)
- Jovic should feel confident: "I just beat a top 5 player, this is easier"
- This could explain blowout wins after big scalps
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


def calculate_momentum_boost(matches, player_data, name_lookup, current_opponent_elo, lookback=1):
    """
    Calculate momentum/confidence boost from recent wins.
    
    If last opponent was tougher than current opponent:
    - positive boost = confidence from beating better player
    - bigger boost = bigger ELO gap
    
    Returns:
        momentum_boost: ELO difference (last_opp - current_opp), positive = confidence
        scalp_factor: How much of an "upset" the last win was
        last_opp_elo: ELO of most recent opponent
        won_last: Did they win their last match?
    """
    if not matches:
        return None
    
    last_match = matches[0]
    
    # Get last opponent's ELO
    opp_name = last_match.get('opponent', '')
    opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
    
    if not opp_key or opp_key not in player_data:
        return None
    
    last_opp_elo = player_data[opp_key].get('elo', 1500)
    won_last = last_match.get('result') == 'W'
    
    # Parse last match spread
    score = last_match.get('score', '')
    sets = re.findall(r'(\d+)-(\d+)', score)
    if sets:
        p1_games = sum(int(s[0]) for s in sets)
        p2_games = sum(int(s[1]) for s in sets)
        last_spread = p1_games - p2_games
    else:
        last_spread = 0
    
    # Momentum boost = how much tougher was last opponent vs current?
    # Positive = last opponent was harder, so current should be "easier"
    momentum_boost = last_opp_elo - current_opponent_elo
    
    # Scalp factor = did they beat someone above their level?
    player_elo = player_data.get(opp_key, {}).get('elo', 1700)  # rough estimate
    
    # Confidence from dominant win
    dominant_win = won_last and last_spread >= 5
    
    return {
        'momentum_boost': momentum_boost if won_last else -momentum_boost,  # Negative if lost
        'last_opp_elo': last_opp_elo,
        'won_last': 1 if won_last else 0,
        'last_spread': last_spread if won_last else -last_spread,
        'facing_easier': 1 if (won_last and momentum_boost > 100) else 0,  # Beat tough, facing easier
        'dominant_win_vs_tough': 1 if (dominant_win and momentum_boost > 0) else 0,
    }


def build_data(data, include_momentum=False):
    """Build training data with optional momentum features"""
    
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
            
            # Momentum features
            if include_momentum:
                p1_mom = calculate_momentum_boost(historical, player_data, name_lookup, opp_data['elo'])
                p2_mom = calculate_momentum_boost(opp_hist, player_data, name_lookup, pdata['elo'])
                
                if p1_mom and p2_mom:
                    features.extend([
                        # Momentum differential
                        p1_mom['momentum_boost'] - p2_mom['momentum_boost'],
                        
                        # Last spread differential
                        p1_mom['last_spread'] - p2_mom['last_spread'],
                        
                        # Facing easier opponent after big win
                        p1_mom['facing_easier'] - p2_mom['facing_easier'],
                        
                        # Dominant win vs tough opponent
                        p1_mom['dominant_win_vs_tough'] - p2_mom['dominant_win_vs_tough'],
                        
                        # Raw values
                        p1_mom['momentum_boost'],
                        p2_mom['momentum_boost'],
                    ])
                else:
                    features.extend([0, 0, 0, 0, 0, 0])
            
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
    
    # Winner
    winner_model = LogisticRegression(max_iter=1000, C=0.1)
    proba = cross_val_predict(winner_model, X_scaled, y, cv=kf, method='predict_proba')[:, 1]
    winner_acc = accuracy_score(y, (proba > 0.5).astype(int))
    winner_auc = roc_auc_score(y, proba)
    
    # Spread
    spread_model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    spread_pred = cross_val_predict(spread_model, X_scaled, y_spread, cv=kf)
    spread_mae = mean_absolute_error(y_spread, spread_pred)
    
    # Total
    total_model = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    total_pred = cross_val_predict(total_model, X_scaled, y_total, cv=kf)
    total_mae = mean_absolute_error(y_total, total_pred)
    
    return winner_acc, winner_auc, spread_mae, total_mae


def main():
    print("="*70)
    print("TOURNAMENT MOMENTUM / CONFIDENCE CASCADE EXPERIMENT")
    print("="*70)
    print()
    print("Concept: If you just beat someone tougher than current opponent,")
    print("         you should have a confidence boost -> bigger spread")
    print()
    
    data = load_data()
    print(f"Loaded {data['player_count']} players")
    print()
    
    # Baseline
    print("Building BASELINE (ELO only)...")
    X_base, y, y_spread, y_total = build_data(data, include_momentum=False)
    print(f"  {X_base.shape[0]} samples, {X_base.shape[1]} features")
    base_acc, base_auc, base_spread, base_total = evaluate(X_base, y, y_spread, y_total)
    
    # With momentum
    print("\nBuilding WITH MOMENTUM FEATURES...")
    X_mom, y, y_spread, y_total = build_data(data, include_momentum=True)
    print(f"  {X_mom.shape[0]} samples, {X_mom.shape[1]} features")
    mom_acc, mom_auc, mom_spread, mom_total = evaluate(X_mom, y, y_spread, y_total)
    
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()
    print(f"{'Metric':<20} {'Baseline':>12} {'+ Momentum':>12} {'Change':>12}")
    print("-"*60)
    print(f"{'Winner Accuracy':<20} {base_acc:>11.1%} {mom_acc:>11.1%} {(mom_acc-base_acc)*100:>+11.2f}%")
    print(f"{'Winner AUC':<20} {base_auc:>11.3f} {mom_auc:>11.3f} {mom_auc-base_auc:>+11.4f}")
    print(f"{'Spread MAE':<20} {base_spread:>11.2f} {mom_spread:>11.2f} {mom_spread-base_spread:>+11.3f}")
    print(f"{'Total MAE':<20} {base_total:>11.2f} {mom_total:>11.2f} {mom_total-base_total:>+11.3f}")
    
    print()
    print("="*70)
    print("MOMENTUM FEATURES")
    print("="*70)
    print()
    print("1. momentum_boost: How much tougher was last opponent vs current?")
    print("   Positive = beat tough, now facing easier (confidence boost)")
    print()
    print("2. last_spread: How dominant was their last win?")
    print("   Big spread + facing easier = potential blowout")
    print()
    print("3. facing_easier: Binary flag for big ELO drop after win")
    print()
    print("4. dominant_win_vs_tough: Beat someone tough by 5+ games")


if __name__ == '__main__':
    main()
