"""
Experiment: Optimal lookback window for form calculation
Also test: recent form vs overall form comparison
"""

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def load_data():
    with open('all_players_matches.json', 'r') as f:
        return json.load(f)


def calculate_stats(matches, lookback):
    """Calculate stats for given lookback window"""
    if not matches or len(matches) < lookback:
        return None
    
    recent = matches[:lookback]
    
    wins = sum(1 for m in recent if m['result'] == 'W')
    win_pct = wins / len(recent)
    
    # Serve/return stats
    stats = {
        'first_won_pct': [], 'ace_pct': [], 'bp_saved_pct': [],
        'rpw_pct': [], 'bp_conv_pct': []
    }
    
    dr_values = []
    
    for m in recent:
        serve = m.get('serve', {})
        ret = m.get('return', {})
        
        for key in ['first_won_pct', 'ace_pct', 'bp_saved_pct']:
            if serve.get(key) is not None:
                stats[key].append(serve[key])
        
        for key in ['rpw_pct', 'bp_conv_pct']:
            if ret.get(key) is not None:
                stats[key].append(ret[key])
        
        spw = (serve.get('first_won_pct') or 65) * 0.6 + (serve.get('second_won_pct') or 50) * 0.4
        rpw = ret.get('rpw_pct') or 35
        if (100 - rpw) > 0:
            dr_values.append(spw / (100 - rpw))
    
    return {
        'win_pct': win_pct,
        'avg_dr': np.mean(dr_values) if dr_values else 1.0,
        **{k: np.mean(v) if v else 50 for k, v in stats.items()}
    }


def build_data_with_lookback(data, recent_lookback, overall_lookback=None):
    """
    Build training data with specified lookback windows.
    If overall_lookback is provided, also include overall stats as separate features.
    """
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
    
    X, y = [], []
    
    for player_name, pdata in player_data.items():
        matches = pdata['matches']
        
        for i, match in enumerate(matches):
            historical = matches[i+1:] if i+1 < len(matches) else []
            
            if len(historical) < max(recent_lookback, overall_lookback or 0):
                continue
            
            opp_name = match.get('opponent', '')
            opp_key = opp_name.lower().replace(' ', '')
            if opp_key not in name_lookup:
                continue
            
            opp_data = player_data.get(name_lookup[opp_key])
            if not opp_data:
                continue
            
            match_date = match.get('date', '')
            opp_historical = [m for m in opp_data['matches'] if m.get('date', '') < match_date]
            
            if len(opp_historical) < max(recent_lookback, overall_lookback or 0):
                continue
            
            surface = match.get('surface', 'Hard').lower()
            surface_elo_key = f'elo_{surface}' if surface in ['hard', 'clay', 'grass'] else 'elo_hard'
            
            # Recent stats
            p1_recent = calculate_stats(historical, recent_lookback)
            p2_recent = calculate_stats(opp_historical, recent_lookback)
            
            if not p1_recent or not p2_recent:
                continue
            
            # Base features (ELO + recent form)
            features = [
                pdata['elo_overall'] - opp_data['elo_overall'],
                pdata.get(surface_elo_key, 1500) - opp_data.get(surface_elo_key, 1500),
                p1_recent['win_pct'] - p2_recent['win_pct'],
                p1_recent['avg_dr'] - p2_recent['avg_dr'],
                p1_recent['first_won_pct'] - p2_recent['first_won_pct'],
                p1_recent['ace_pct'] - p2_recent['ace_pct'],
                p1_recent['bp_saved_pct'] - p2_recent['bp_saved_pct'],
                p1_recent['rpw_pct'] - p2_recent['rpw_pct'],
                p1_recent['bp_conv_pct'] - p2_recent['bp_conv_pct'],
            ]
            
            # Add overall stats if specified
            if overall_lookback:
                p1_overall = calculate_stats(historical, overall_lookback)
                p2_overall = calculate_stats(opp_historical, overall_lookback)
                
                if p1_overall and p2_overall:
                    features.extend([
                        p1_overall['win_pct'] - p2_overall['win_pct'],
                        p1_overall['avg_dr'] - p2_overall['avg_dr'],
                        p1_overall['first_won_pct'] - p2_overall['first_won_pct'],
                        p1_overall['rpw_pct'] - p2_overall['rpw_pct'],
                        # Recent vs Overall difference (momentum indicator)
                        (p1_recent['win_pct'] - p1_overall['win_pct']) - (p2_recent['win_pct'] - p2_overall['win_pct']),
                        (p1_recent['avg_dr'] - p1_overall['avg_dr']) - (p2_recent['avg_dr'] - p2_overall['avg_dr']),
                    ])
                else:
                    continue
            
            X.append(features)
            y.append(1 if match.get('result') == 'W' else 0)
    
    return np.array(X), np.array(y)


def evaluate(X, y):
    """Evaluate model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, C=0.1)
    model.fit(X_train_s, y_train)
    
    acc = model.score(X_test_s, y_test)
    auc = roc_auc_score(y_test, model.predict_proba(X_test_s)[:, 1])
    cv = cross_val_score(model, X_train_s, y_train, cv=5).mean()
    
    return acc, auc, cv, len(X)


def main():
    data = load_data()
    print(f"Loaded {data['player_count']} players with {data['total_matches']} matches\n")
    
    # Test 1: Different lookback windows (recent form only)
    print("="*70)
    print("TEST 1: OPTIMAL LOOKBACK WINDOW (Recent Form Only)")
    print("="*70)
    print(f"\n{'Lookback':<12} {'Accuracy':>10} {'AUC-ROC':>10} {'CV':>10} {'Samples':>10}")
    print("-"*55)
    
    results = []
    for lookback in [5, 7, 10, 15, 20, 25, 30, 40, 50]:
        X, y = build_data_with_lookback(data, lookback, overall_lookback=None)
        if len(X) > 100:
            acc, auc, cv, n = evaluate(X, y)
            results.append((lookback, acc, auc, cv, n))
            print(f"{lookback:<12} {acc:>9.1%} {auc:>10.3f} {cv:>9.1%} {n:>10}")
    
    best = max(results, key=lambda x: x[2])
    print(f"\n*** Best lookback: {best[0]} matches (AUC: {best[2]:.3f}) ***")
    
    # Test 2: Recent + Overall combined
    print("\n" + "="*70)
    print("TEST 2: RECENT vs OVERALL FORM COMPARISON")
    print("="*70)
    print("\nTesting: Recent form (X matches) + Overall form (50 matches) + Momentum")
    print(f"\n{'Recent':<10} {'Overall':<10} {'Accuracy':>10} {'AUC-ROC':>10} {'CV':>10} {'Samples':>10}")
    print("-"*65)
    
    results2 = []
    for recent in [5, 7, 10, 15, 20]:
        X, y = build_data_with_lookback(data, recent, overall_lookback=50)
        if len(X) > 100:
            acc, auc, cv, n = evaluate(X, y)
            results2.append((recent, 50, acc, auc, cv, n))
            print(f"{recent:<10} {50:<10} {acc:>9.1%} {auc:>10.3f} {cv:>9.1%} {n:>10}")
    
    # Test 3: Different overall windows
    print("\n" + "="*70)
    print("TEST 3: VARYING OVERALL WINDOW (with recent=10)")
    print("="*70)
    print(f"\n{'Recent':<10} {'Overall':<10} {'Accuracy':>10} {'AUC-ROC':>10} {'CV':>10}")
    print("-"*55)
    
    results3 = []
    for overall in [20, 30, 40, 50]:
        X, y = build_data_with_lookback(data, 10, overall_lookback=overall)
        if len(X) > 100:
            acc, auc, cv, n = evaluate(X, y)
            results3.append((10, overall, acc, auc, cv, n))
            print(f"{10:<10} {overall:<10} {acc:>9.1%} {auc:>10.3f} {cv:>9.1%}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Best from test 1
    best1 = max(results, key=lambda x: x[2])
    print(f"\nBest single lookback: {best1[0]} matches -> AUC {best1[2]:.3f}")
    
    if results2:
        best2 = max(results2, key=lambda x: x[3])
        print(f"Best recent+overall:  Recent {best2[0]} + Overall {best2[1]} → AUC {best2[3]:.3f}")
    
    if results3:
        best3 = max(results3, key=lambda x: x[3])
        print(f"Best overall window:  Recent 10 + Overall {best3[1]} → AUC {best3[3]:.3f}")
    
    # Compare current (10) vs best
    X_10, y_10 = build_data_with_lookback(data, 10, overall_lookback=None)
    acc_10, auc_10, _, _ = evaluate(X_10, y_10)
    
    X_best, y_best = build_data_with_lookback(data, best1[0], overall_lookback=None)
    acc_best, auc_best, _, _ = evaluate(X_best, y_best)
    
    print(f"\nCurrent model (10):   AUC {auc_10:.3f}, Acc {acc_10:.1%}")
    print(f"Best lookback ({best1[0]}):  AUC {auc_best:.3f}, Acc {acc_best:.1%}")
    print(f"Improvement:          AUC {auc_best - auc_10:+.3f}")


if __name__ == '__main__':
    main()
