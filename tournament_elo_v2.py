"""
Tournament ELO v2: Focus on quality of path and dominance

Key insight: Jovic beat Paolini (2013 ELO), Putintseva's best was Bouzkova (1841)
That 172 ELO difference in "best scalp" should matter A LOT
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


def calculate_tournament_form(matches, base_elo, player_data, name_lookup, max_matches=4):
    """
    Calculate tournament form metrics focused on:
    1. Best scalp (highest ELO beaten)
    2. Dominance score (spread-weighted)
    3. Tournament path quality
    """
    if not matches:
        return None
    
    recent = matches[:max_matches]
    
    best_scalp_elo = 0
    total_quality = 0
    weighted_dominance = 0
    total_weight = 0
    wins = 0
    bagels_breadsticks = 0  # 6-0 or 6-1 sets
    
    for i, match in enumerate(recent):
        opp_name = match.get('opponent', '')
        opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
        
        if not opp_key or opp_key not in player_data:
            continue
        
        opp_elo = player_data[opp_key].get('elo', 1500)
        won = match.get('result') == 'W'
        
        score = match.get('score', '')
        sets = re.findall(r'(\d+)-(\d+)', score)
        if not sets:
            continue
        
        p1_games = sum(int(s[0]) for s in sets)
        p2_games = sum(int(s[1]) for s in sets)
        spread = p1_games - p2_games
        
        if won:
            wins += 1
            
            # Track best scalp
            if opp_elo > best_scalp_elo:
                best_scalp_elo = opp_elo
            
            # Quality of opponent beaten
            total_quality += opp_elo
            
            # Dominance weighted by recency (recent wins matter more)
            weight = max_matches - i  # 4, 3, 2, 1 for most recent
            weighted_dominance += spread * weight
            total_weight += weight
            
            # Count bagels/breadsticks (6-0, 6-1)
            for s in sets:
                if int(s[0]) >= 6 and int(s[1]) <= 1 and int(s[0]) > int(s[1]):
                    bagels_breadsticks += 1
    
    if wins == 0:
        return None
    
    avg_dominance = weighted_dominance / total_weight if total_weight > 0 else 0
    avg_quality = total_quality / wins
    
    # "Tournament ELO" = base + boost from quality wins
    # Beating someone 100 ELO higher = +10 boost
    # Doing it dominantly (spread 6+) = another +5 boost
    tournament_boost = 0
    for i, match in enumerate(recent):
        opp_name = match.get('opponent', '')
        opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
        if not opp_key or opp_key not in player_data:
            continue
        opp_elo = player_data[opp_key].get('elo', 1500)
        won = match.get('result') == 'W'
        
        if won:
            # Scalp bonus: beating higher ELO
            scalp_diff = opp_elo - base_elo
            if scalp_diff > 0:
                tournament_boost += scalp_diff / 10  # +10 per 100 ELO above
            
            # Dominance bonus
            score = match.get('score', '')
            sets = re.findall(r'(\d+)-(\d+)', score)
            if sets:
                p1 = sum(int(s[0]) for s in sets)
                p2 = sum(int(s[1]) for s in sets)
                spread = p1 - p2
                if spread >= 6:
                    tournament_boost += 5
    
    return {
        'tournament_elo': base_elo + tournament_boost,
        'tournament_boost': tournament_boost,
        'best_scalp': best_scalp_elo,
        'best_scalp_diff': best_scalp_elo - base_elo,  # Positive = beat someone better
        'avg_quality': avg_quality,
        'avg_dominance': avg_dominance,
        'bagels_breadsticks': bagels_breadsticks,
        'wins': wins,
    }


def build_data(data, include_features=False):
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
            
            if include_features:
                p1_form = calculate_tournament_form(historical, pdata['elo'], player_data, name_lookup)
                p2_form = calculate_tournament_form(opp_hist, opp_data['elo'], player_data, name_lookup)
                
                if p1_form and p2_form:
                    features.extend([
                        # Tournament-adjusted ELO difference
                        p1_form['tournament_elo'] - p2_form['tournament_elo'],
                        
                        # Best scalp differential (who beat better opponents?)
                        p1_form['best_scalp'] - p2_form['best_scalp'],
                        
                        # Did they beat someone above their level?
                        p1_form['best_scalp_diff'] - p2_form['best_scalp_diff'],
                        
                        # Average dominance (spread-weighted by recency)
                        p1_form['avg_dominance'] - p2_form['avg_dominance'],
                        
                        # Bagels/breadsticks (pure dominance indicator)
                        p1_form['bagels_breadsticks'] - p2_form['bagels_breadsticks'],
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
    print("TOURNAMENT FORM v2 - QUALITY OF PATH")
    print("="*70)
    print()
    print("Focus on: Best scalp, dominance, bagels/breadsticks")
    print()
    
    data = load_data()
    print(f"Loaded {data['player_count']} players")
    print()
    
    print("Building BASELINE...")
    X_base, y, y_spread, y_total = build_data(data, include_features=False)
    print(f"  {X_base.shape[0]} samples, {X_base.shape[1]} features")
    base_acc, base_auc, base_spread, base_total = evaluate(X_base, y, y_spread, y_total)
    
    print("\nBuilding WITH TOURNAMENT FORM v2...")
    X_new, y, y_spread, y_total = build_data(data, include_features=True)
    print(f"  {X_new.shape[0]} samples, {X_new.shape[1]} features")
    new_acc, new_auc, new_spread, new_total = evaluate(X_new, y, y_spread, y_total)
    
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()
    print(f"{'Metric':<20} {'Baseline':>12} {'+ Form v2':>12} {'Change':>12}")
    print("-"*60)
    print(f"{'Winner Accuracy':<20} {base_acc:>11.1%} {new_acc:>11.1%} {(new_acc-base_acc)*100:>+11.2f}%")
    print(f"{'Winner AUC':<20} {base_auc:>11.3f} {new_auc:>11.3f} {new_auc-base_auc:>+11.4f}")
    print(f"{'Spread MAE':<20} {base_spread:>11.2f} {new_spread:>11.2f} {new_spread-base_spread:>+11.3f}")
    print(f"{'Total MAE':<20} {base_total:>11.2f} {new_total:>11.2f} {new_total-base_total:>+11.3f}")
    
    # Show example for Jovic vs Putintseva
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
    
    j = calculate_tournament_form(player_data['Iva Jovic']['matches'], 1862, player_data, name_lookup)
    p = calculate_tournament_form(player_data['Yulia Putintseva']['matches'], 1788, player_data, name_lookup)
    
    print()
    print(f"{'Feature':<25} {'Jovic':>12} {'Putintseva':>12} {'Diff':>12}")
    print("-"*65)
    print(f"{'Base ELO':<25} {'1862':>12} {'1788':>12} {'+74':>12}")
    print(f"{'Tournament ELO':<25} {j['tournament_elo']:>12.0f} {p['tournament_elo']:>12.0f} {j['tournament_elo']-p['tournament_elo']:>+12.0f}")
    print(f"{'Best Scalp ELO':<25} {j['best_scalp']:>12.0f} {p['best_scalp']:>12.0f} {j['best_scalp']-p['best_scalp']:>+12.0f}")
    print(f"{'Beat Someone Above Level':<25} {j['best_scalp_diff']:>+12.0f} {p['best_scalp_diff']:>+12.0f} {j['best_scalp_diff']-p['best_scalp_diff']:>+12.0f}")
    print(f"{'Avg Dominance':<25} {j['avg_dominance']:>12.1f} {p['avg_dominance']:>12.1f} {j['avg_dominance']-p['avg_dominance']:>+12.1f}")
    print(f"{'Bagels/Breadsticks':<25} {j['bagels_breadsticks']:>12} {p['bagels_breadsticks']:>12} {j['bagels_breadsticks']-p['bagels_breadsticks']:>+12}")


if __name__ == '__main__':
    main()
