"""
Experiment: Tournament-Adjusted ELO

Estimate a player's CURRENT ELO within the tournament based on:
1. Their base ELO entering the tournament
2. Quality of opponents beaten (opponent ELOs)
3. How convincing the wins were (spread, dominance)

This captures "playing into form" during a tournament.
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


def calculate_tournament_elo(matches, base_elo, player_data, name_lookup, max_matches=4):
    """
    Calculate a player's tournament-adjusted ELO.
    
    Looks at recent matches (assumed to be same tournament) and adjusts ELO based on:
    - Quality of opponents beaten
    - Dominance of wins (spread)
    
    Returns:
        tournament_elo: Adjusted ELO based on tournament performance
        elo_gain: How much they've "risen" during tournament
        quality_beaten: Average ELO of opponents beaten
        avg_dominance: Average dominance score of wins
    """
    if not matches:
        return {
            'tournament_elo': base_elo,
            'elo_gain': 0,
            'quality_beaten': 0,
            'avg_dominance': 0,
            'wins_in_tournament': 0,
        }
    
    # Look at recent matches (up to max_matches, assumed same tournament)
    recent = matches[:max_matches]
    
    tournament_elo = base_elo
    quality_beaten = []
    dominance_scores = []
    wins = 0
    
    for match in recent:
        opp_name = match.get('opponent', '')
        opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
        
        if not opp_key or opp_key not in player_data:
            continue
        
        opp_elo = player_data[opp_key].get('elo', 1500)
        won = match.get('result') == 'W'
        
        # Parse score for dominance metrics
        score = match.get('score', '')
        sets = re.findall(r'(\d+)-(\d+)', score)
        
        if not sets:
            continue
        
        p1_games = sum(int(s[0]) for s in sets)
        p2_games = sum(int(s[1]) for s in sets)
        total_games = p1_games + p2_games
        spread = p1_games - p2_games
        
        # Dominance score: how convincing was the win?
        # Factors: spread, sets won cleanly, total games (fewer = more dominant)
        if won:
            wins += 1
            quality_beaten.append(opp_elo)
            
            # Dominance metrics
            sets_won = sum(1 for s in sets if int(s[0]) > int(s[1]))
            sets_lost = len(sets) - sets_won
            
            # Spread component: +1 per game margin
            spread_score = spread
            
            # Clean sets: bonus for 6-0, 6-1, 6-2 sets
            clean_sets = sum(1 for s in sets if int(s[0]) >= 6 and int(s[1]) <= 2 and int(s[0]) > int(s[1]))
            
            # Efficiency: winning in fewer total games
            efficiency = max(0, 24 - total_games) / 10  # ~2.4 bonus for a 6-0 6-0
            
            dominance = spread_score + (clean_sets * 2) + efficiency
            dominance_scores.append(dominance)
            
            # ELO adjustment for this win
            # Expected score based on ELO difference
            expected = 1 / (1 + 10 ** ((opp_elo - base_elo) / 400))
            
            # Actual performance: 1 = win, but scale by dominance
            # A dominant win (spread 10+) counts as more than a close win
            performance_bonus = min(spread / 10, 0.5)  # Up to 0.5 bonus for dominant win
            actual = 1 + performance_bonus
            
            # K-factor: how much to adjust (higher for tournament matches)
            K = 32
            
            # ELO gain for this match
            elo_change = K * (actual - expected)
            tournament_elo += elo_change
        else:
            # Loss: ELO decreases
            expected = 1 / (1 + 10 ** ((opp_elo - base_elo) / 400))
            K = 32
            elo_change = K * (0 - expected)
            tournament_elo += elo_change
    
    return {
        'tournament_elo': tournament_elo,
        'elo_gain': tournament_elo - base_elo,
        'quality_beaten': np.mean(quality_beaten) if quality_beaten else 0,
        'avg_dominance': np.mean(dominance_scores) if dominance_scores else 0,
        'wins_in_tournament': wins,
    }


def build_data(data, include_tournament_elo=False):
    """Build training data with optional tournament ELO features"""
    
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
            
            # Tournament ELO features
            if include_tournament_elo:
                p1_tourn = calculate_tournament_elo(historical, pdata['elo'], player_data, name_lookup)
                p2_tourn = calculate_tournament_elo(opp_hist, opp_data['elo'], player_data, name_lookup)
                
                features.extend([
                    # Tournament ELO difference (instead of base ELO)
                    p1_tourn['tournament_elo'] - p2_tourn['tournament_elo'],
                    
                    # ELO gain differential (who's rising more?)
                    p1_tourn['elo_gain'] - p2_tourn['elo_gain'],
                    
                    # Quality of opponents beaten
                    p1_tourn['quality_beaten'] - p2_tourn['quality_beaten'],
                    
                    # Average dominance in tournament
                    p1_tourn['avg_dominance'] - p2_tourn['avg_dominance'],
                    
                    # Wins in tournament
                    p1_tourn['wins_in_tournament'] - p2_tourn['wins_in_tournament'],
                ])
            
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
    print("TOURNAMENT-ADJUSTED ELO EXPERIMENT")
    print("="*70)
    print()
    print("Concept: Estimate a player's CURRENT ELO based on tournament performance")
    print("- Quality of opponents beaten")
    print("- Dominance of wins (spread, clean sets, efficiency)")
    print()
    
    data = load_data()
    print(f"Loaded {data['player_count']} players")
    print()
    
    # Baseline
    print("Building BASELINE (base ELO only)...")
    X_base, y, y_spread, y_total = build_data(data, include_tournament_elo=False)
    print(f"  {X_base.shape[0]} samples, {X_base.shape[1]} features")
    base_acc, base_auc, base_spread, base_total = evaluate(X_base, y, y_spread, y_total)
    
    # With tournament ELO
    print("\nBuilding WITH TOURNAMENT ELO...")
    X_tourn, y, y_spread, y_total = build_data(data, include_tournament_elo=True)
    print(f"  {X_tourn.shape[0]} samples, {X_tourn.shape[1]} features")
    tourn_acc, tourn_auc, tourn_spread, tourn_total = evaluate(X_tourn, y, y_spread, y_total)
    
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()
    print(f"{'Metric':<20} {'Baseline':>12} {'+ Tourn ELO':>12} {'Change':>12}")
    print("-"*60)
    print(f"{'Winner Accuracy':<20} {base_acc:>11.1%} {tourn_acc:>11.1%} {(tourn_acc-base_acc)*100:>+11.2f}%")
    print(f"{'Winner AUC':<20} {base_auc:>11.3f} {tourn_auc:>11.3f} {tourn_auc-base_auc:>+11.4f}")
    print(f"{'Spread MAE':<20} {base_spread:>11.2f} {tourn_spread:>11.2f} {tourn_spread-base_spread:>+11.3f}")
    print(f"{'Total MAE':<20} {base_total:>11.2f} {tourn_total:>11.2f} {tourn_total-base_total:>+11.3f}")
    
    print()
    print("="*70)
    print("TOURNAMENT ELO FEATURES")
    print("="*70)
    print()
    print("1. tournament_elo: Base ELO + adjustments from tournament wins")
    print("   - Adjusted by opponent ELO (beating better = bigger boost)")
    print("   - Adjusted by win dominance (bigger spread = bigger boost)")
    print()
    print("2. elo_gain: How much has ELO risen during tournament?")
    print("   - Positive = playing above base level")
    print("   - Captures 'form' during tournament")
    print()
    print("3. quality_beaten: Average ELO of opponents beaten")
    print("   - Higher = facing tougher competition path")
    print()
    print("4. avg_dominance: How dominant have wins been?")
    print("   - Spread + clean sets + efficiency")
    print()
    print("5. wins_in_tournament: Number of wins (depth in tournament)")


if __name__ == '__main__':
    main()
