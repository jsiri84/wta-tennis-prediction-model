"""
Final tuning for total games model - testing regularization
"""

import json
import numpy as np
import re
from collections import defaultdict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def parse_score(score_str):
    if not score_str:
        return None
    score_upper = score_str.upper()
    if any(x in score_upper for x in ['RET', 'W/O', 'WO', 'DEF', 'ABD']):
        return None
    p_games, o_games = 0, 0
    tiebreaks = 0
    sets = score_str.strip().split()
    
    for s in sets:
        tiebreaks += s.count('(')
        s_clean = re.sub(r'\(\d+\)', '', s)
        match = re.match(r'(\d+)-(\d+)', s_clean)
        if match:
            p_games += int(match.group(1))
            o_games += int(match.group(2))
    
    if p_games == 0 and o_games == 0:
        return None
    
    return {'total': p_games + o_games, 'tiebreaks': tiebreaks}


def load_data():
    with open('all_players_matches.json', 'r') as f:
        return json.load(f)

def get_tournament_level(tourn):
    tourn_lower = tourn.lower()
    if any(gs in tourn_lower for gs in ['australian open', 'french open', 'roland garros', 'wimbledon', 'us open']):
        return 4
    if '1000' in tourn_lower:
        return 3
    if '500' in tourn_lower:
        return 2
    return 1

def get_round_level(round_str):
    return {'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4, 'QF': 5, 'SF': 6, 'F': 7, 'RR': 2}.get(round_str, 3)


def calc_stats(matches, lookback=15):
    if not matches or len(matches) < 3:
        return None
    recent = matches[:lookback]
    wins = sum(1 for m in recent if m['result'] == 'W')
    
    totals, tiebreaks = [], 0
    serve_stats = defaultdict(list)
    return_stats = defaultdict(list)
    
    for m in recent:
        parsed = parse_score(m.get('score', ''))
        if parsed:
            totals.append(parsed['total'])
            tiebreaks += parsed['tiebreaks']
        
        serve = m.get('serve', {})
        for key in ['first_won_pct', 'second_won_pct', 'ace_pct', 'df_pct', 'bp_saved_pct']:
            if serve.get(key) is not None:
                serve_stats[key].append(serve[key])
        
        ret = m.get('return', {})
        for key in ['rpw_pct', 'bp_conv_pct']:
            if ret.get(key) is not None:
                return_stats[key].append(ret[key])
    
    n = len(recent)
    
    return {
        'win_pct': wins / n,
        'avg_total': np.mean(totals) if totals else 20,
        'std_total': np.std(totals) if len(totals) > 1 else 3,
        'tb_rate': tiebreaks / n,
        'first_won': np.mean(serve_stats['first_won_pct']) if serve_stats['first_won_pct'] else 65,
        'second_won': np.mean(serve_stats['second_won_pct']) if serve_stats['second_won_pct'] else 50,
        'ace': np.mean(serve_stats['ace_pct']) if serve_stats['ace_pct'] else 5,
        'df': np.mean(serve_stats['df_pct']) if serve_stats['df_pct'] else 3,
        'bp_saved': np.mean(serve_stats['bp_saved_pct']) if serve_stats['bp_saved_pct'] else 60,
        'rpw': np.mean(return_stats['rpw_pct']) if return_stats['rpw_pct'] else 35,
        'bp_conv': np.mean(return_stats['bp_conv_pct']) if return_stats['bp_conv_pct'] else 40,
    }


def build_data(data):
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
        if ' ' in name:
            name_lookup[name.split()[-1].lower()] = name
    
    X, y = [], []
    
    for player_name, pdata in player_data.items():
        for i, match in enumerate(pdata['matches']):
            if i < 5:
                continue
            
            parsed = parse_score(match.get('score', ''))
            if not parsed:
                continue
            
            opp_name = match.get('opponent', '')
            opp_key = opp_name.lower().replace(' ', '')
            if opp_key not in name_lookup:
                last = opp_name.split()[-1].lower() if opp_name else ''
                if last in name_lookup:
                    opp_key = last
                else:
                    continue
            
            opp_full = name_lookup.get(opp_key)
            if not opp_full or opp_full not in player_data:
                continue
            
            opp_data = player_data[opp_full]
            match_date = match.get('date', '')
            hist = [m for m in pdata['matches'][i+1:] if m.get('date', '') < match_date]
            opp_hist = [m for m in opp_data['matches'] if m.get('date', '') < match_date]
            
            if len(hist) < 5 or len(opp_hist) < 5:
                continue
            
            surface = match.get('surface', 'Hard').lower()
            surface_key = f'elo_{surface}' if surface in ['hard', 'clay', 'grass'] else 'elo_hard'
            
            p1 = calc_stats(hist)
            p2 = calc_stats(opp_hist)
            if not p1 or not p2:
                continue
            
            tourn_level = get_tournament_level(match.get('tournament', ''))
            round_level = get_round_level(match.get('round', 'R32'))
            
            features = [
                abs(pdata['elo'] - opp_data['elo']),
                abs(pdata.get(surface_key, 1500) - opp_data.get(surface_key, 1500)),
                (pdata['elo'] + opp_data['elo']) / 2,
                (p1['avg_total'] + p2['avg_total']) / 2,
                p1['std_total'] + p2['std_total'],
                p1['tb_rate'] + p2['tb_rate'],
                abs(p1['win_pct'] - p2['win_pct']),
                (p1['first_won'] + p2['first_won']) / 2,
                (p1['second_won'] + p2['second_won']) / 2,
                (p1['ace'] + p2['ace']) / 2,
                (p1['df'] + p2['df']) / 2,
                (p1['bp_saved'] + p2['bp_saved']) / 2,
                (p1['rpw'] + p2['rpw']) / 2,
                (p1['bp_conv'] + p2['bp_conv']) / 2,
                tourn_level,
                round_level,
                1 if surface == 'hard' else 0,
                1 if surface == 'clay' else 0,
                1 if surface == 'grass' else 0,
            ]
            
            X.append(features)
            y.append(parsed['total'])
    
    return np.array(X), np.array(y)


print("="*70)
print("FINAL TUNING - REGULARIZATION TEST")
print("="*70)

data = load_data()
X, y = build_data(data)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nDataset: {len(X)} samples, {len(X[0])} features")

# Test regularization parameters
configs = [
    # (name, n_estimators, max_depth, learning_rate, min_samples_split, min_samples_leaf, subsample)
    ("Current (depth=6)", 200, 6, 0.05, 2, 1, 1.0),
    ("Depth=8 no reg", 200, 8, 0.05, 2, 1, 1.0),
    ("Depth=8 + subsample", 200, 8, 0.05, 2, 1, 0.8),
    ("Depth=8 + min_samples", 200, 8, 0.05, 10, 4, 1.0),
    ("Depth=8 + both", 200, 8, 0.05, 10, 4, 0.8),
    ("Depth=7 + subsample", 250, 7, 0.05, 5, 2, 0.8),
    ("Depth=8 + heavy reg", 200, 8, 0.03, 20, 8, 0.7),
    ("Depth=6 + subsample", 250, 6, 0.05, 5, 2, 0.8),
]

print("\n" + "-"*85)
print(f"{'Configuration':<25} {'Train':<8} {'Test':<8} {'CV MAE':<12} {'Gap':<8} {'Status'}")
print("-"*85)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

best_cv = 999
best_config = None

for name, n_est, depth, lr, min_split, min_leaf, subsample in configs:
    model = GradientBoostingRegressor(
        n_estimators=n_est, 
        max_depth=depth, 
        learning_rate=lr,
        min_samples_split=min_split,
        min_samples_leaf=min_leaf,
        subsample=subsample,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    train_mae = mean_absolute_error(y_train, model.predict(X_train))
    test_mae = mean_absolute_error(y_test, model.predict(X_test))
    
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    gap = test_mae - train_mae
    status = "BEST" if cv_mae < best_cv else ""
    
    if cv_mae < best_cv:
        best_cv = cv_mae
        best_config = (name, n_est, depth, lr, min_split, min_leaf, subsample)
    
    print(f"{name:<25} {train_mae:<8.3f} {test_mae:<8.3f} {cv_mae:.3f}+/-{cv_std:.2f}  {gap:<8.2f} {status}")

print("\n" + "="*70)
print("BEST CONFIGURATION")
print("="*70)
print(f"Name: {best_config[0]}")
print(f"n_estimators: {best_config[1]}")
print(f"max_depth: {best_config[2]}")
print(f"learning_rate: {best_config[3]}")
print(f"min_samples_split: {best_config[4]}")
print(f"min_samples_leaf: {best_config[5]}")
print(f"subsample: {best_config[6]}")
print(f"CV MAE: {best_cv:.3f}")

# Calculate O/U accuracy with best model
print("\n--- O/U Accuracy with Best Model ---")
name, n_est, depth, lr, min_split, min_leaf, subsample = best_config
best_model = GradientBoostingRegressor(
    n_estimators=n_est, max_depth=depth, learning_rate=lr,
    min_samples_split=min_split, min_samples_leaf=min_leaf,
    subsample=subsample, random_state=42
)
best_model.fit(X_train, y_train)
pred = best_model.predict(X_test)

for line in [20.5, 21.5, 22.5]:
    pred_over = pred > line
    actual_over = y_test > line
    acc = np.mean(pred_over == actual_over)
    print(f"O/U {line}: {acc*100:.1f}%")
