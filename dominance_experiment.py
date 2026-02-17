"""
Experiment: Player Dominance vs Quality Opponents

Measures HOW players win against strong opponents, not just IF they win.
- Average spread vs strong opponents
- Serve dominance vs strong opponents
- Return dominance vs strong opponents
- Break point efficiency vs strong opponents
"""

import json
import re
import numpy as np
from collections import defaultdict
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error


def load_data():
    with open('all_players_matches.json', 'r') as f:
        return json.load(f)


def parse_score(score_str):
    """Parse score string into games"""
    if not score_str:
        return None, None, None
    try:
        sets = re.findall(r'(\d+)-(\d+)', score_str)
        if not sets:
            return None, None, None
        p1_games = sum(int(s[0]) for s in sets)
        p2_games = sum(int(s[1]) for s in sets)
        return p1_games, p2_games, p1_games + p2_games
    except:
        return None, None, None


def calculate_dominance_vs_strong(matches, player_data, name_lookup, strong_threshold=1900, lookback=20):
    """
    Calculate how DOMINANT a player is against strong opponents.
    Not just win/loss, but margin and stats when playing top players.
    
    Returns:
        - avg_spread_vs_strong: Average game spread vs strong opponents (positive = winning by more)
        - serve_dominance_vs_strong: Serve performance vs strong opponents
        - return_dominance_vs_strong: Return performance vs strong opponents
        - bp_efficiency_vs_strong: Break point conversion vs strong opponents
        - matches_vs_strong: Sample size
    """
    recent = matches[:lookback]
    
    spreads = []
    serve_pcts = []
    return_pcts = []
    bp_converted = []
    bp_saved = []
    dominance_ratios = []
    
    matches_vs_strong = 0
    
    for match in recent:
        opp_name = match.get('opponent', '')
        opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
        
        if not opp_key or opp_key not in player_data:
            continue
            
        opp_elo = player_data[opp_key].get('elo_overall', player_data[opp_key].get('elo', 1500))
        
        if opp_elo >= strong_threshold:
            matches_vs_strong += 1
            
            # Parse spread
            p1_games, p2_games, total = parse_score(match.get('score', ''))
            if p1_games is not None:
                spread = p1_games - p2_games
                spreads.append(spread)
            
            # Serve stats
            serve = match.get('serve', {})
            if serve:
                first_won = serve.get('first_won_pct') or 0
                second_won = serve.get('second_won_pct') or 0
                if first_won or second_won:
                    serve_pcts.append(first_won * 0.6 + second_won * 0.4)
                if serve.get('bp_saved_pct'):
                    bp_saved.append(serve['bp_saved_pct'])
            
            # Return stats
            ret = match.get('return', {})
            if ret:
                if ret.get('rpw_pct'):
                    return_pcts.append(ret['rpw_pct'])
                if ret.get('bp_conv_pct'):
                    bp_converted.append(ret['bp_conv_pct'])
            
            # Dominance ratio (games won / games lost)
            if p2_games and p2_games > 0:
                dominance_ratios.append(p1_games / p2_games)
    
    return {
        'avg_spread_vs_strong': np.mean(spreads) if spreads else 0,
        'spread_std_vs_strong': np.std(spreads) if len(spreads) > 1 else 5,
        'serve_pct_vs_strong': np.mean(serve_pcts) if serve_pcts else 55,
        'return_pct_vs_strong': np.mean(return_pcts) if return_pcts else 35,
        'bp_conv_vs_strong': np.mean(bp_converted) if bp_converted else 40,
        'bp_saved_vs_strong': np.mean(bp_saved) if bp_saved else 60,
        'dominance_ratio_vs_strong': np.mean(dominance_ratios) if dominance_ratios else 1.0,
        'matches_vs_strong': matches_vs_strong
    }


def calculate_dominance_vs_similar(matches, player_elo, player_data, name_lookup, elo_range=150, lookback=20):
    """
    Calculate dominance vs opponents of SIMILAR level (within elo_range).
    Shows how a player handles peers.
    """
    recent = matches[:lookback]
    
    spreads = []
    serve_pcts = []
    return_pcts = []
    dominance_ratios = []
    matches_vs_similar = 0
    
    for match in recent:
        opp_name = match.get('opponent', '')
        opp_key = name_lookup.get(opp_name.lower()) or name_lookup.get(opp_name.lower().replace(' ', ''))
        
        if not opp_key or opp_key not in player_data:
            continue
            
        opp_elo = player_data[opp_key].get('elo_overall', player_data[opp_key].get('elo', 1500))
        
        # Check if opponent is within similar ELO range
        if abs(opp_elo - player_elo) <= elo_range:
            matches_vs_similar += 1
            
            p1_games, p2_games, total = parse_score(match.get('score', ''))
            if p1_games is not None:
                spreads.append(p1_games - p2_games)
                if p2_games > 0:
                    dominance_ratios.append(p1_games / p2_games)
            
            serve = match.get('serve', {})
            if serve:
                first_won = serve.get('first_won_pct') or 0
                second_won = serve.get('second_won_pct') or 0
                if first_won or second_won:
                    serve_pcts.append(first_won * 0.6 + second_won * 0.4)
            
            ret = match.get('return', {})
            if ret and ret.get('rpw_pct'):
                return_pcts.append(ret['rpw_pct'])
    
    return {
        'avg_spread_vs_similar': np.mean(spreads) if spreads else 0,
        'serve_pct_vs_similar': np.mean(serve_pcts) if serve_pcts else 55,
        'return_pct_vs_similar': np.mean(return_pcts) if return_pcts else 35,
        'dominance_ratio_vs_similar': np.mean(dominance_ratios) if dominance_ratios else 1.0,
        'matches_vs_similar': matches_vs_similar
    }


def calculate_close_match_tendency(matches, lookback=15):
    """
    Calculate how often a player's matches go to 3 sets or tiebreaks.
    High value = wins/losses on fine margins.
    """
    recent = matches[:lookback]
    
    three_setters = 0
    tiebreaks = 0
    close_matches = 0  # Spread <= 3 games
    
    for match in recent:
        score = match.get('score', '')
        p1_games, p2_games, total = parse_score(score)
        
        if p1_games is not None:
            spread = abs(p1_games - p2_games)
            
            # Count sets
            sets = len(re.findall(r'\d+-\d+', score))
            if sets == 3:
                three_setters += 1
            
            # Count tiebreaks
            tiebreaks += score.count('(')
            
            # Close match (spread <= 3)
            if spread <= 3:
                close_matches += 1
    
    n = len(recent) if recent else 1
    
    return {
        'three_set_rate': three_setters / n,
        'tiebreak_rate': tiebreaks / n,
        'close_match_rate': close_matches / n
    }


# ============================================================
# BUILD TRAINING DATA WITH DOMINANCE FEATURES
# ============================================================

def build_training_data(data, include_dominance=False):
    """Build training data with optional dominance features"""
    
    player_data = {}
    for name, info in data['players'].items():
        player_data[name] = {
            'elo_overall': info.get('elo_overall', 1500),
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
                pdata['elo_overall'] - opp_data['elo_overall'],
                pdata.get(surface_key, 1500) - opp_data.get(surface_key, 1500),
            ]
            
            # Dominance features
            if include_dominance:
                p1_dom_strong = calculate_dominance_vs_strong(historical, player_data, name_lookup)
                p2_dom_strong = calculate_dominance_vs_strong(opp_hist, player_data, name_lookup)
                
                p1_dom_similar = calculate_dominance_vs_similar(historical, pdata['elo_overall'], player_data, name_lookup)
                p2_dom_similar = calculate_dominance_vs_similar(opp_hist, opp_data['elo_overall'], player_data, name_lookup)
                
                p1_close = calculate_close_match_tendency(historical)
                p2_close = calculate_close_match_tendency(opp_hist)
                
                features.extend([
                    # Dominance vs strong opponents (7 features)
                    p1_dom_strong['avg_spread_vs_strong'] - p2_dom_strong['avg_spread_vs_strong'],
                    p1_dom_strong['serve_pct_vs_strong'] - p2_dom_strong['serve_pct_vs_strong'],
                    p1_dom_strong['return_pct_vs_strong'] - p2_dom_strong['return_pct_vs_strong'],
                    p1_dom_strong['bp_conv_vs_strong'] - p2_dom_strong['bp_conv_vs_strong'],
                    p1_dom_strong['bp_saved_vs_strong'] - p2_dom_strong['bp_saved_vs_strong'],
                    p1_dom_strong['dominance_ratio_vs_strong'] - p2_dom_strong['dominance_ratio_vs_strong'],
                    p1_dom_strong['matches_vs_strong'] - p2_dom_strong['matches_vs_strong'],
                    
                    # Dominance vs similar opponents (4 features)
                    p1_dom_similar['avg_spread_vs_similar'] - p2_dom_similar['avg_spread_vs_similar'],
                    p1_dom_similar['serve_pct_vs_similar'] - p2_dom_similar['serve_pct_vs_similar'],
                    p1_dom_similar['dominance_ratio_vs_similar'] - p2_dom_similar['dominance_ratio_vs_similar'],
                    p1_dom_similar['matches_vs_similar'] - p2_dom_similar['matches_vs_similar'],
                    
                    # Close match tendency (3 features) - negative = relies on fine margins
                    p2_close['three_set_rate'] - p1_close['three_set_rate'],  # Higher for P2 = P1 more dominant
                    p2_close['tiebreak_rate'] - p1_close['tiebreak_rate'],
                    p2_close['close_match_rate'] - p1_close['close_match_rate'],
                ])
            
            # Parse score for labels
            p1_games, p2_games, total = parse_score(match.get('score', ''))
            if p1_games is None or total < 12 or total > 50:
                continue
            
            won = match.get('result') == 'W'
            spread = p1_games - p2_games
            
            X.append(features)
            y.append(1 if won else 0)
            y_spread.append(spread)
            y_total.append(total)
    
    return np.array(X), np.array(y), np.array(y_spread), np.array(y_total)


def evaluate(X, y, y_spread, y_total):
    """Evaluate model performance"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Winner
    winner_model = LogisticRegression(max_iter=1000, C=0.1)
    winner_proba = cross_val_predict(winner_model, X_scaled, y, cv=kf, method='predict_proba')[:, 1]
    winner_acc = accuracy_score(y, (winner_proba > 0.5).astype(int))
    winner_auc = roc_auc_score(y, winner_proba)
    
    # Spread
    spread_model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    spread_pred = cross_val_predict(spread_model, X_scaled, y_spread, cv=kf)
    spread_mae = mean_absolute_error(y_spread, spread_pred)
    
    # Total
    total_model = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    total_pred = cross_val_predict(total_model, X_scaled, y_total, cv=kf)
    total_mae = mean_absolute_error(y_total, total_pred)
    ou_acc = ((y_total > 21.5) == (total_pred > 21.5)).mean()
    
    return {
        'winner_acc': winner_acc,
        'winner_auc': winner_auc,
        'spread_mae': spread_mae,
        'total_mae': total_mae,
        'ou_acc': ou_acc
    }


def run_experiment():
    print("="*70)
    print("DOMINANCE VS QUALITY OPPONENTS - EXPERIMENT")
    print("="*70)
    print()
    
    data = load_data()
    print(f"Loaded {data['player_count']} players")
    print()
    
    # Baseline (just ELO)
    print("Building BASELINE (ELO only)...")
    X_base, y, y_spread, y_total = build_training_data(data, include_dominance=False)
    print(f"  {X_base.shape[0]} samples, {X_base.shape[1]} features")
    
    print("Evaluating baseline...")
    baseline = evaluate(X_base, y, y_spread, y_total)
    
    # With dominance features
    print()
    print("Building WITH DOMINANCE FEATURES...")
    X_dom, y, y_spread, y_total = build_training_data(data, include_dominance=True)
    print(f"  {X_dom.shape[0]} samples, {X_dom.shape[1]} features")
    
    print("Evaluating with dominance...")
    with_dom = evaluate(X_dom, y, y_spread, y_total)
    
    # Results
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()
    print(f"{'Metric':<20} {'Baseline':>12} {'+ Dominance':>12} {'Change':>12}")
    print("-"*60)
    print(f"{'Winner Accuracy':<20} {baseline['winner_acc']:>11.1%} {with_dom['winner_acc']:>11.1%} {(with_dom['winner_acc']-baseline['winner_acc'])*100:>+11.2f}%")
    print(f"{'Winner AUC':<20} {baseline['winner_auc']:>11.3f} {with_dom['winner_auc']:>11.3f} {with_dom['winner_auc']-baseline['winner_auc']:>+11.4f}")
    print(f"{'Spread MAE':<20} {baseline['spread_mae']:>11.2f} {with_dom['spread_mae']:>11.2f} {with_dom['spread_mae']-baseline['spread_mae']:>+11.3f}")
    print(f"{'Total MAE':<20} {baseline['total_mae']:>11.2f} {with_dom['total_mae']:>11.2f} {with_dom['total_mae']-baseline['total_mae']:>+11.3f}")
    print(f"{'O/U 21.5 Acc':<20} {baseline['ou_acc']:>11.1%} {with_dom['ou_acc']:>11.1%} {(with_dom['ou_acc']-baseline['ou_acc'])*100:>+11.2f}%")
    
    print()
    print("="*70)
    print("NEW FEATURES ADDED (14 total)")
    print("="*70)
    print()
    print("Dominance vs Strong Opponents (ELO > 1900):")
    print("  - avg_spread_vs_strong: How much they win by vs top players")
    print("  - serve_pct_vs_strong: Serve points won vs top players")
    print("  - return_pct_vs_strong: Return points won vs top players")
    print("  - bp_conv_vs_strong: Break point conversion vs top players")
    print("  - bp_saved_vs_strong: Break point saving vs top players")
    print("  - dominance_ratio_vs_strong: Games won / games lost vs top players")
    print("  - matches_vs_strong: Experience vs top players")
    print()
    print("Dominance vs Similar Opponents (within 150 ELO):")
    print("  - avg_spread_vs_similar: How much they win by vs peers")
    print("  - serve_pct_vs_similar: Serve performance vs peers")
    print("  - dominance_ratio_vs_similar: Games ratio vs peers")
    print("  - matches_vs_similar: Experience vs peers")
    print()
    print("Close Match Tendency:")
    print("  - three_set_rate: How often matches go 3 sets (fine margins)")
    print("  - tiebreak_rate: How often matches have tiebreaks")
    print("  - close_match_rate: How often spread <= 3 games")
    
    return baseline, with_dom


if __name__ == '__main__':
    run_experiment()
