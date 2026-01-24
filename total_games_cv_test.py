"""
Cross-validation test for total games model improvements
"""

import json
import numpy as np
import re
from collections import defaultdict
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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
    decisive = 0
    close = 0
    
    for s in sets:
        tiebreaks += s.count('(')
        s_clean = re.sub(r'\(\d+\)', '', s)
        match = re.match(r'(\d+)-(\d+)', s_clean)
        if match:
            pg, og = int(match.group(1)), int(match.group(2))
            p_games += pg
            o_games += og
            if abs(pg - og) >= 4:
                decisive += 1
            if abs(pg - og) <= 2:
                close += 1
    
    if p_games == 0 and o_games == 0:
        return None
    
    return {
        'total': p_games + o_games,
        'sets': len(sets),
        'tiebreaks': tiebreaks,
        'decisive': decisive,
        'close': close,
        'avg_set_games': (p_games + o_games) / len(sets) if sets else 0
    }


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
    
    totals, tiebreaks, decisive, close, set_games = [], 0, 0, 0, []
    serve_stats = defaultdict(list)
    return_stats = defaultdict(list)
    
    for m in recent:
        parsed = parse_score(m.get('score', ''))
        if parsed:
            totals.append(parsed['total'])
            tiebreaks += parsed['tiebreaks']
            decisive += parsed['decisive']
            close += parsed['close']
            set_games.append(parsed['avg_set_games'])
        
        serve = m.get('serve', {})
        for key in ['first_won_pct', 'second_won_pct', 'ace_pct', 'df_pct', 'bp_saved_pct', 'dr']:
            if serve.get(key) is not None:
                serve_stats[key].append(serve[key])
        
        ret = m.get('return', {})
        for key in ['rpw_pct', 'bp_conv_pct']:
            if ret.get(key) is not None:
                return_stats[key].append(ret[key])
    
    n = len(recent)
    n_parsed = len(totals) if totals else 1
    
    return {
        'win_pct': wins / n,
        'avg_total': np.mean(totals) if totals else 20,
        'std_total': np.std(totals) if len(totals) > 1 else 3,
        'median_total': np.median(totals) if totals else 20,
        'three_set_rate': sum(1 for t in totals if t > 21) / n_parsed if totals else 0.3,
        'tb_rate': tiebreaks / n,
        'decisive_rate': decisive / (n_parsed * 2),
        'close_rate': close / (n_parsed * 2),
        'avg_set_games': np.mean(set_games) if set_games else 10,
        'first_won': np.mean(serve_stats['first_won_pct']) if serve_stats['first_won_pct'] else 65,
        'second_won': np.mean(serve_stats['second_won_pct']) if serve_stats['second_won_pct'] else 50,
        'ace': np.mean(serve_stats['ace_pct']) if serve_stats['ace_pct'] else 5,
        'df': np.mean(serve_stats['df_pct']) if serve_stats['df_pct'] else 3,
        'bp_saved': np.mean(serve_stats['bp_saved_pct']) if serve_stats['bp_saved_pct'] else 60,
        'dr': np.mean(serve_stats['dr']) if serve_stats['dr'] else 1.0,
        'rpw': np.mean(return_stats['rpw_pct']) if return_stats['rpw_pct'] else 35,
        'bp_conv': np.mean(return_stats['bp_conv_pct']) if return_stats['bp_conv_pct'] else 40,
    }


def get_h2h_avg(p1_matches, p2_name, match_date):
    h2h_totals = []
    for m in p1_matches:
        if m.get('date', '') >= match_date:
            continue
        opp = m.get('opponent', '').lower()
        if p2_name.lower() in opp or opp in p2_name.lower():
            parsed = parse_score(m.get('score', ''))
            if parsed:
                h2h_totals.append(parsed['total'])
    return np.mean(h2h_totals) if h2h_totals else None, len(h2h_totals)


def build_data(data, include_new=False):
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
                p1['three_set_rate'] + p2['three_set_rate'],
                p1['tb_rate'] + p2['tb_rate'],
                abs(p1['win_pct'] - p2['win_pct']),
                (p1['first_won'] + p2['first_won']) / 2,
                (p1['ace'] + p2['ace']) / 2,
                (p1['bp_saved'] + p2['bp_saved']) / 2,
                (p1['rpw'] + p2['rpw']) / 2,
                (p1['bp_conv'] + p2['bp_conv']) / 2,
                tourn_level,
                round_level,
                1 if surface == 'hard' else 0,
                1 if surface == 'clay' else 0,
                1 if surface == 'grass' else 0,
            ]
            
            if include_new:
                h2h_avg, h2h_count = get_h2h_avg(hist, opp_full, match_date)
                features.extend([
                    (p1['median_total'] + p2['median_total']) / 2,
                    p1['decisive_rate'] + p2['decisive_rate'],
                    p1['close_rate'] + p2['close_rate'],
                    (p1['avg_set_games'] + p2['avg_set_games']) / 2,
                    (p1['second_won'] + p2['second_won']) / 2,
                    (p1['df'] + p2['df']) / 2,
                    (p1['dr'] + p2['dr']) / 2,
                    h2h_avg if h2h_avg else (p1['avg_total'] + p2['avg_total']) / 2,
                    h2h_count,
                    1 if round_level >= 5 else 0,
                    min(pdata['elo'], opp_data['elo']),
                ])
            
            X.append(features)
            y.append(parsed['total'])
    
    return np.array(X), np.array(y)


print("="*70)
print("CROSS-VALIDATION TEST FOR OVERFITTING CHECK")
print("="*70)

data = load_data()
X, y = build_data(data, include_new=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nDataset: {len(X)} samples, {len(X[0])} features")
print(f"Target stats: mean={y.mean():.1f}, std={y.std():.1f}")

# Test different model configurations with CV
configs = [
    ("GB depth=5", GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)),
    ("GB depth=6", GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)),
    ("GB depth=7", GradientBoostingRegressor(n_estimators=200, max_depth=7, learning_rate=0.05, random_state=42)),
    ("GB depth=8", GradientBoostingRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42)),
    ("GB depth=8 lr=0.03", GradientBoostingRegressor(n_estimators=300, max_depth=8, learning_rate=0.03, random_state=42)),
    ("GB depth=6 n=300", GradientBoostingRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)),
]

print("\n" + "-"*70)
print(f"{'Model':<25} {'Train MAE':<12} {'Test MAE':<12} {'CV MAE':<15} {'Overfit?'}")
print("-"*70)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

for name, model in configs:
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    gap = test_mae - train_mae
    overfit = "YES" if gap > 0.5 else ("SOME" if gap > 0.3 else "NO")
    
    print(f"{name:<25} {train_mae:<12.3f} {test_mae:<12.3f} {cv_mae:.3f} +/- {cv_std:.2f}   {overfit}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("Lower CV MAE is better. High train-test gap indicates overfitting.")
print("The model with lowest CV MAE and smallest gap is most reliable.")
