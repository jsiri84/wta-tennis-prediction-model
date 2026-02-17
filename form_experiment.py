"""
Experiment: Current Form vs Expected Performance

Measures if a player is currently playing ABOVE or BELOW their expected level.
- Look at last 3-5 matches
- Calculate expected win probability based on ELO difference for each
- Compare actual wins to expected wins
- Hot player = winning more than ELO suggests
- Cold player = losing more than ELO suggests
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


def elo_expected_score(player_elo, opponent_elo):
    """Calculate expected score (0-1) based on ELO difference"""
    return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))


def calculate_form_vs_expected(matches, player_elo, player_surface_elo, player_data, name_lookup, surface, window=5):
    """
    Calculate how player is performing vs their expected level.
    
    Returns:
        form_diff: actual_wins - expected_wins (positive = overperforming)
        form_pct_diff: (actual_win_pct - expected_win_pct) 
        surface_form_diff: same but only on current surface
        streak: current win/loss streak (positive = wins)
    """
    recent = matches[:window]
    if len(recent) < 3:
        return None
    
    # Overall form
    actual_wins = 0
    expected_wins = 0
    
    # Surface-specific form
    surface_actual = 0
    surface_expected = 0
    surface_matches = 0
    
    # Spread analysis
    actual_spreads = []
    expected_spreads = []  # Based on ELO diff
    
    for match in recent:
        opp_name = match.get('opponent', '')
        opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
        
        if not opp_key or opp_key not in player_data:
            continue
        
        opp_elo = player_data[opp_key].get('elo', 1500)
        match_surface = match.get('surface', '').lower()
        
        # Expected win probability
        exp_win = elo_expected_score(player_elo, opp_elo)
        expected_wins += exp_win
        
        # Actual result
        won = match.get('result') == 'W'
        if won:
            actual_wins += 1
        
        # Surface-specific
        if match_surface == surface.lower():
            surface_matches += 1
            opp_surface_elo = player_data[opp_key].get(f'elo_{surface.lower()}', opp_elo)
            surface_exp = elo_expected_score(player_surface_elo, opp_surface_elo)
            surface_expected += surface_exp
            if won:
                surface_actual += 1
        
        # Spread analysis
        score = match.get('score', '')
        sets = re.findall(r'(\d+)-(\d+)', score)
        if sets:
            p1_games = sum(int(s[0]) for s in sets)
            p2_games = sum(int(s[1]) for s in sets)
            actual_spreads.append(p1_games - p2_games)
            
            # Expected spread based on ELO (rough approximation)
            elo_diff = player_elo - opp_elo
            expected_spread = elo_diff / 50  # ~1 game per 50 ELO
            expected_spreads.append(expected_spread)
    
    n = len(recent)
    
    # Win streak
    streak = 0
    for m in recent:
        if m.get('result') == 'W':
            if streak >= 0:
                streak += 1
            else:
                break
        else:
            if streak <= 0:
                streak -= 1
            else:
                break
    
    # Form metrics
    form_diff = actual_wins - expected_wins  # Positive = overperforming
    form_pct_diff = (actual_wins / n) - (expected_wins / n) if n > 0 else 0
    
    # Surface form
    if surface_matches >= 2:
        surface_form_diff = surface_actual - surface_expected
        surface_form_pct = (surface_actual / surface_matches) - (surface_expected / surface_matches)
    else:
        surface_form_diff = form_diff * 0.5  # Fallback to overall form
        surface_form_pct = form_pct_diff * 0.5
    
    # Spread form (are they winning by more/less than expected?)
    if actual_spreads and expected_spreads:
        avg_actual_spread = np.mean(actual_spreads)
        avg_expected_spread = np.mean(expected_spreads)
        spread_form_diff = avg_actual_spread - avg_expected_spread
    else:
        spread_form_diff = 0
    
    return {
        'form_diff': form_diff,  # +ve = hot, -ve = cold
        'form_pct_diff': form_pct_diff,
        'surface_form_diff': surface_form_diff,
        'surface_form_pct': surface_form_pct,
        'spread_form_diff': spread_form_diff,  # Winning by more/less than expected
        'streak': streak,
        'recent_win_pct': actual_wins / n if n > 0 else 0.5,
        'expected_win_pct': expected_wins / n if n > 0 else 0.5,
    }


def build_data(data, include_form=False, window=5):
    """Build training data with optional form features"""
    
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
            
            # Form features
            if include_form:
                p1_form = calculate_form_vs_expected(
                    historical, pdata['elo'], pdata.get(surface_key, 1500),
                    player_data, name_lookup, surface, window
                )
                p2_form = calculate_form_vs_expected(
                    opp_hist, opp_data['elo'], opp_data.get(surface_key, 1500),
                    player_data, name_lookup, surface, window
                )
                
                if p1_form and p2_form:
                    features.extend([
                        # Form vs expected (differences)
                        p1_form['form_diff'] - p2_form['form_diff'],
                        p1_form['form_pct_diff'] - p2_form['form_pct_diff'],
                        p1_form['surface_form_diff'] - p2_form['surface_form_diff'],
                        p1_form['spread_form_diff'] - p2_form['spread_form_diff'],
                        
                        # Individual form states (for spread/total)
                        p1_form['form_diff'],
                        p2_form['form_diff'],
                        p1_form['streak'] - p2_form['streak'],
                    ])
                else:
                    features.extend([0] * 7)
            
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
    ou_acc = ((y_total > 21.5) == (total_pred > 21.5)).mean()
    
    return winner_acc, winner_auc, spread_mae, total_mae, ou_acc


def main():
    print("="*70)
    print("FORM VS EXPECTED PERFORMANCE EXPERIMENT")
    print("="*70)
    print()
    print("Concept: Is player playing ABOVE or BELOW their ELO level recently?")
    print("Hot = winning more than ELO suggests, Cold = losing more")
    print()
    
    data = load_data()
    print(f"Loaded {data['player_count']} players")
    print()
    
    results = []
    
    # Test different window sizes
    configs = [
        ('Baseline (ELO only)', False, 5),
        ('+ Form (3-match window)', True, 3),
        ('+ Form (5-match window)', True, 5),
        ('+ Form (7-match window)', True, 7),
    ]
    
    for name, include_form, window in configs:
        print(f"Testing: {name}...")
        X, y, y_spread, y_total = build_data(data, include_form, window)
        print(f"  {X.shape[0]} samples, {X.shape[1]} features")
        
        acc, auc, spread, total, ou = evaluate(X, y, y_spread, y_total)
        results.append((name, X.shape[1], acc, auc, spread, total, ou))
    
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()
    print(f"{'Config':<30} {'Feat':>5} {'Win%':>7} {'AUC':>7} {'Spread':>7} {'Total':>7} {'O/U':>7}")
    print("-"*75)
    
    baseline = results[0]
    for name, feat, acc, auc, spread, total, ou in results:
        print(f"{name:<30} {feat:>5} {acc:>6.1%} {auc:>7.3f} {spread:>7.2f} {total:>7.2f} {ou:>6.1%}")
    
    print()
    print("IMPROVEMENT FROM BASELINE:")
    print("-"*75)
    for name, feat, acc, auc, spread, total, ou in results[1:]:
        acc_d = (acc - baseline[2]) * 100
        auc_d = auc - baseline[3]
        spread_d = spread - baseline[4]
        total_d = total - baseline[5]
        ou_d = (ou - baseline[6]) * 100
        
        print(f"{name:<30}       {acc_d:>+6.2f}% {auc_d:>+7.4f} {spread_d:>+7.3f} {total_d:>+7.3f} {ou_d:>+6.2f}%")
    
    print()
    print("="*70)
    print("FORM FEATURES TESTED")
    print("="*70)
    print()
    print("1. form_diff: actual_wins - expected_wins (based on ELO)")
    print("   +ve = overperforming (hot), -ve = underperforming (cold)")
    print()
    print("2. surface_form_diff: same but surface-specific")
    print()
    print("3. spread_form_diff: winning by more/less margin than ELO suggests")
    print()
    print("4. streak: current win/loss streak")


if __name__ == '__main__':
    main()
