"""
Tennis Match Prediction Model

Factors:
1. Player current ELO (overall)
2. Player surface ELO
3. ELO difference
4. Current form (recent record, DR)
5. Serve stats (1st%, 1stIn%, Ace%, DF%)
6. Return stats (RPW%, BP conversion)
7. Cumulative match time in event
8. Head-to-head record
"""

import json
import numpy as np
import os
import joblib
from collections import defaultdict
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Model save paths
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
WINNER_MODEL_PATH = os.path.join(MODEL_DIR, 'winner_model.pkl')
DATA_PATH = os.path.join(MODEL_DIR, 'all_players_matches.json')


def load_data():
    """Load all player match data"""
    with open('all_players_matches.json', 'r') as f:
        return json.load(f)


def calculate_rolling_stats(matches, lookback=15):
    """Calculate rolling stats for a player's recent matches"""
    if not matches:
        return None
    
    recent = matches[:lookback]
    
    # Record
    wins = sum(1 for m in recent if m['result'] == 'W')
    losses = len(recent) - wins
    win_pct = wins / len(recent) if recent else 0.5
    
    # Dominance Ratio (serve points won / return points won ratio approximation)
    dr_values = []
    for m in recent:
        serve = m.get('serve', {})
        ret = m.get('return', {})
        
        # DR = (1st won% + 2nd won%) / (opponent 1st won% + opponent 2nd won%)
        # Simplified: serve efficiency vs return efficiency
        spw = (serve.get('first_won_pct') or 0) * 0.6 + (serve.get('second_won_pct') or 0) * 0.4
        rpw = ret.get('rpw_pct') or 0
        
        if rpw > 0:
            dr = spw / (100 - rpw) if (100 - rpw) > 0 else 1.0
            dr_values.append(dr)
    
    avg_dr = np.mean(dr_values) if dr_values else 1.0
    
    # Serve stats
    serve_stats = {
        'first_in_pct': [],
        'first_won_pct': [],
        'ace_pct': [],
        'df_pct': [],
        'bp_saved_pct': []
    }
    
    for m in recent:
        serve = m.get('serve', {})
        for key in serve_stats:
            val = serve.get(key)
            if val is not None:
                serve_stats[key].append(val)
    
    # Return stats
    return_stats = {
        'rpw_pct': [],
        'bp_conv_pct': []
    }
    
    for m in recent:
        ret = m.get('return', {})
        for key in return_stats:
            val = ret.get(key)
            if val is not None:
                return_stats[key].append(val)
    
    return {
        'win_pct': win_pct,
        'wins': wins,
        'losses': losses,
        'match_count': len(recent),
        'avg_dr': avg_dr,
        'first_in_pct': np.mean(serve_stats['first_in_pct']) if serve_stats['first_in_pct'] else 60,
        'first_won_pct': np.mean(serve_stats['first_won_pct']) if serve_stats['first_won_pct'] else 65,
        'ace_pct': np.mean(serve_stats['ace_pct']) if serve_stats['ace_pct'] else 5,
        'df_pct': np.mean(serve_stats['df_pct']) if serve_stats['df_pct'] else 3,
        'bp_saved_pct': np.mean(serve_stats['bp_saved_pct']) if serve_stats['bp_saved_pct'] else 60,
        'rpw_pct': np.mean(return_stats['rpw_pct']) if return_stats['rpw_pct'] else 35,
        'bp_conv_pct': np.mean(return_stats['bp_conv_pct']) if return_stats['bp_conv_pct'] else 40
    }


def calculate_surface_form(matches, surface, lookback=15):
    """Calculate form on specific surface"""
    surface_matches = [m for m in matches if m.get('surface', '').lower() == surface.lower()]
    return calculate_rolling_stats(surface_matches[:lookback], lookback)


def calculate_blended_stats(matches, surface, lookback=15, surface_weight=0.6):
    """
    Calculate blended stats weighting surface-specific matches more heavily.
    
    Surface matches get surface_weight (default 60%)
    Other surface matches get (1 - surface_weight) (default 40%)
    
    Returns stats with proper weighting based on available data.
    """
    if not matches:
        return None
    
    # Split matches by surface
    surface_matches = [m for m in matches if m.get('surface', '').lower() == surface.lower()]
    other_matches = [m for m in matches if m.get('surface', '').lower() != surface.lower()]
    
    # Take recent matches from each
    recent_surface = surface_matches[:lookback]
    recent_other = other_matches[:lookback]
    
    # Calculate stats for each group
    surf_stats = calculate_rolling_stats(recent_surface, lookback) if recent_surface else None
    other_stats = calculate_rolling_stats(recent_other, lookback) if recent_other else None
    
    # If only one type available, use that
    if surf_stats and not other_stats:
        surf_stats['surface_match_count'] = len(recent_surface)
        surf_stats['blend_weight'] = 1.0
        return surf_stats
    if other_stats and not surf_stats:
        other_stats['surface_match_count'] = 0
        other_stats['blend_weight'] = 0.0
        return other_stats
    if not surf_stats and not other_stats:
        return None
    
    # Calculate adaptive weight based on sample sizes
    # More surface matches = trust surface stats more
    surf_count = len(recent_surface)
    other_count = len(recent_other)
    
    # Base weight adjusted by sample size confidence
    # If we have 10+ surface matches, use full surface_weight
    # If fewer, reduce proportionally
    surface_confidence = min(surf_count / 10, 1.0)
    adaptive_weight = surface_weight * surface_confidence + (1 - surface_weight) * (1 - surface_confidence)
    
    # Blend the stats
    blended = {}
    stat_keys = ['win_pct', 'avg_dr', 'first_in_pct', 'first_won_pct', 'ace_pct', 
                 'df_pct', 'bp_saved_pct', 'rpw_pct', 'bp_conv_pct']
    
    for key in stat_keys:
        surf_val = surf_stats.get(key, 0)
        other_val = other_stats.get(key, 0)
        blended[key] = adaptive_weight * surf_val + (1 - adaptive_weight) * other_val
    
    blended['wins'] = surf_stats['wins'] + other_stats['wins']
    blended['losses'] = surf_stats['losses'] + other_stats['losses']
    blended['match_count'] = surf_count + other_count
    blended['surface_match_count'] = surf_count
    blended['blend_weight'] = adaptive_weight
    
    return blended


def calculate_event_fatigue(matches, tournament, match_date):
    """Calculate cumulative match time in current event"""
    total_time = 0
    matches_played = 0
    
    for m in matches:
        if m.get('tournament') == tournament and m.get('date', '') < match_date:
            time_str = m.get('time_mins', '0')
            try:
                total_time += int(time_str) if time_str else 0
                matches_played += 1
            except:
                pass
    
    return total_time, matches_played


def calculate_head_to_head(matches, opponent_name):
    """Calculate head-to-head record against a specific opponent"""
    h2h_wins = 0
    h2h_losses = 0
    
    opp_lower = opponent_name.lower()
    
    for m in matches:
        match_opp = m.get('opponent', '').lower()
        # Match by full name or last name
        if match_opp == opp_lower or match_opp.split()[-1] == opp_lower.split()[-1]:
            if m['result'] == 'W':
                h2h_wins += 1
            else:
                h2h_losses += 1
    
    total = h2h_wins + h2h_losses
    win_pct = h2h_wins / total if total > 0 else 0.5
    
    return {
        'wins': h2h_wins,
        'losses': h2h_losses,
        'total': total,
        'win_pct': win_pct,
        'diff': h2h_wins - h2h_losses
    }


def build_training_data(data):
    """Build training dataset from historical matches"""
    
    # Create player lookup with matches sorted by date (newest first)
    player_data = {}
    for name, info in data['players'].items():
        player_data[name] = {
            'elo_overall': info.get('elo_overall', 1500),
            'elo_hard': info.get('elo_hard', 1500),
            'elo_clay': info.get('elo_clay', 1500),
            'elo_grass': info.get('elo_grass', 1500),
            'matches': sorted(info.get('matches', []), key=lambda x: x.get('date', ''), reverse=True)
        }
    
    # Also create lookup by various name formats
    name_lookup = {}
    for name in player_data:
        name_lookup[name.lower()] = name
        name_lookup[name.lower().replace(' ', '')] = name
        # Handle "Firstname Lastname" -> "Lastname"
        parts = name.split()
        if len(parts) > 1:
            name_lookup[parts[-1].lower()] = name
    
    X = []  # Features
    y = []  # Labels (1 = win, 0 = loss)
    match_info = []  # For analysis
    
    print("Building training data...")
    
    for player_name, pdata in player_data.items():
        matches = pdata['matches']
        
        for i, match in enumerate(matches):
            # Use matches AFTER this one for form calculation (simulating prediction time)
            historical_matches = matches[i+1:] if i+1 < len(matches) else []
            
            if len(historical_matches) < 5:
                continue  # Need some history
            
            # Find opponent in our database
            opp_name = match.get('opponent', '')
            opp_key = None
            for key in [opp_name.lower(), opp_name.lower().replace(' ', '')]:
                if key in name_lookup:
                    opp_key = name_lookup[key]
                    break
            
            if not opp_key or opp_key not in player_data:
                continue  # Opponent not in database
            
            opp_data = player_data[opp_key]
            
            # Find opponent's matches before this date
            match_date = match.get('date', '')
            opp_historical = [m for m in opp_data['matches'] if m.get('date', '') < match_date]
            
            if len(opp_historical) < 5:
                continue
            
            # Surface
            surface = match.get('surface', 'Hard').lower()
            surface_elo_key = f'elo_{surface}' if surface in ['hard', 'clay', 'grass'] else 'elo_hard'
            
            # Calculate form for both players (all surfaces)
            p1_form = calculate_rolling_stats(historical_matches, 15)
            p2_form = calculate_rolling_stats(opp_historical, 15)
            
            if not p1_form or not p2_form:
                continue
            
            # Surface-specific form
            p1_surf_form = calculate_surface_form(historical_matches, surface, 15)
            p2_surf_form = calculate_surface_form(opp_historical, surface, 15)
            
            # Blended stats (surface-weighted)
            p1_blended = calculate_blended_stats(historical_matches, surface, 15, surface_weight=0.6)
            p2_blended = calculate_blended_stats(opp_historical, surface, 15, surface_weight=0.6)
            
            if not p1_blended or not p2_blended:
                continue
            
            # Event fatigue
            tournament = match.get('tournament', '')
            p1_time, p1_matches = calculate_event_fatigue(historical_matches, tournament, match_date)
            p2_time, p2_matches = calculate_event_fatigue(opp_historical, tournament, match_date)
            
            # Head-to-head record
            p1_h2h = calculate_head_to_head(historical_matches, opp_name)
            
            # Build feature vector
            features = [
                # ELO features
                pdata['elo_overall'] - opp_data['elo_overall'],  # Overall ELO diff
                pdata.get(surface_elo_key, 1500) - opp_data.get(surface_elo_key, 1500),  # Surface ELO diff
                pdata['elo_overall'],  # P1 ELO
                opp_data['elo_overall'],  # P2 ELO
                
                # Form features (all surfaces)
                p1_form['win_pct'] - p2_form['win_pct'],  # Win% diff
                p1_form['avg_dr'] - p2_form['avg_dr'],  # DR diff
                p1_form['win_pct'],
                p2_form['win_pct'],
                
                # Serve features (all surfaces)
                p1_form['first_in_pct'] - p2_form['first_in_pct'],
                p1_form['first_won_pct'] - p2_form['first_won_pct'],
                p1_form['ace_pct'] - p2_form['ace_pct'],
                p1_form['df_pct'] - p2_form['df_pct'],  # Lower is better
                p1_form['bp_saved_pct'] - p2_form['bp_saved_pct'],
                
                # Return features (all surfaces)
                p1_form['rpw_pct'] - p2_form['rpw_pct'],
                p1_form['bp_conv_pct'] - p2_form['bp_conv_pct'],
                
                # Surface form (if available)
                (p1_surf_form['win_pct'] if p1_surf_form else 0.5) - (p2_surf_form['win_pct'] if p2_surf_form else 0.5),
                
                # Fatigue features
                p2_time - p1_time,  # Opponent more tired = good for P1
                p2_matches - p1_matches,
                
                # === NEW: Surface-weighted blended stats ===
                # Blended serve (surface-weighted)
                p1_blended['first_won_pct'] - p2_blended['first_won_pct'],
                p1_blended['ace_pct'] - p2_blended['ace_pct'],
                p1_blended['bp_saved_pct'] - p2_blended['bp_saved_pct'],
                
                # Blended return (surface-weighted)
                p1_blended['rpw_pct'] - p2_blended['rpw_pct'],
                p1_blended['bp_conv_pct'] - p2_blended['bp_conv_pct'],
                
                # Blended form (surface-weighted)
                p1_blended['win_pct'] - p2_blended['win_pct'],
                p1_blended['avg_dr'] - p2_blended['avg_dr'],
                
                # Surface experience features
                p1_blended.get('surface_match_count', 0) - p2_blended.get('surface_match_count', 0),
                p1_blended.get('surface_match_count', 0),  # P1 surface experience
                p2_blended.get('surface_match_count', 0),  # P2 surface experience
                
                # Head-to-head features
                p1_h2h['diff'],      # H2H record difference (wins - losses)
                p1_h2h['win_pct'],   # H2H win percentage
                p1_h2h['total'],     # Number of previous meetings
            ]
            
            # Label
            label = 1 if match.get('result') == 'W' else 0
            
            X.append(features)
            y.append(label)
            match_info.append({
                'player': player_name,
                'opponent': opp_name,
                'date': match_date,
                'surface': surface,
                'tournament': tournament,
                'result': match.get('result')
            })
    
    return np.array(X), np.array(y), match_info


def train_and_evaluate():
    """Train model and evaluate performance"""
    
    data = load_data()
    print(f"Loaded {data['player_count']} players with {data['total_matches']} matches")
    
    X, y, match_info = build_training_data(data)
    print(f"\nBuilt {len(X)} training samples")
    print(f"Win rate in data: {y.mean():.1%}")
    
    # Feature names for analysis
    feature_names = [
        # ELO (4)
        'elo_diff', 'surface_elo_diff', 'p1_elo', 'p2_elo',
        # Form - all surfaces (4)
        'win_pct_diff', 'dr_diff', 'p1_win_pct', 'p2_win_pct',
        # Serve - all surfaces (5)
        'first_in_diff', 'first_won_diff', 'ace_diff', 'df_diff', 'bp_saved_diff',
        # Return - all surfaces (2)
        'rpw_diff', 'bp_conv_diff',
        # Surface form (1)
        'surface_form_diff',
        # Fatigue (2)
        'fatigue_time_diff', 'fatigue_matches_diff',
        # Blended serve - surface weighted (3)
        'blended_first_won_diff', 'blended_ace_diff', 'blended_bp_saved_diff',
        # Blended return - surface weighted (2)
        'blended_rpw_diff', 'blended_bp_conv_diff',
        # Blended form - surface weighted (2)
        'blended_win_pct_diff', 'blended_dr_diff',
        # Surface experience (3)
        'surface_exp_diff', 'p1_surface_exp', 'p2_surface_exp',
        # Head-to-head (3)
        'h2h_diff', 'h2h_win_pct', 'h2h_meetings'
    ]
    
    # Split data
    X_train, X_test, y_train, y_test, info_train, info_test = train_test_split(
        X, y, match_info, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=0.1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    }
    
    best_model = None
    best_acc = 0
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        
        train_acc = model.score(X_train_scaled, y_train)
        test_acc = model.score(X_test_scaled, y_test)
        
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        print(f"\n{name}:")
        print(f"  Train Accuracy: {train_acc:.1%}")
        print(f"  Test Accuracy:  {test_acc:.1%}")
        print(f"  AUC-ROC:        {auc:.3f}")
        print(f"  CV Accuracy:    {cv_scores.mean():.1%} (+/- {cv_scores.std()*2:.1%})")
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = (name, model)
    
    # Analyze best model
    print("\n" + "="*60)
    print(f"FEATURE IMPORTANCE ({best_model[0]})")
    print("="*60)
    
    model = best_model[1]
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # For logistic regression, use absolute coefficients
        importances = np.abs(model.coef_[0])
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    print("\nTop Features (most predictive of winning):")
    for i, idx in enumerate(indices[:10]):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Correlation analysis
    print("\n" + "="*60)
    print("FEATURE CORRELATIONS WITH WINNING")
    print("="*60)
    
    correlations = []
    for i, name in enumerate(feature_names):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        correlations.append((name, corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\nStrongest correlations:")
    for name, corr in correlations:
        direction = "+" if corr > 0 else "-"
        print(f"  {name}: {direction}{abs(corr):.3f}")
    
    # Calibration analysis
    print("\n" + "="*60)
    print("PREDICTION CALIBRATION")
    print("="*60)
    
    y_pred_proba = best_model[1].predict_proba(X_test_scaled)[:, 1]
    
    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    print("\nPredicted probability vs actual win rate:")
    for low, high in bins:
        mask = (y_pred_proba >= low) & (y_pred_proba < high)
        if mask.sum() > 0:
            actual = y_test[mask].mean()
            count = mask.sum()
            print(f"  {low:.0%}-{high:.0%} predicted: {actual:.1%} actual ({count} matches)")
    
    return best_model[1], scaler, feature_names


def analyze_trends(data):
    """Analyze specific trends in the data"""
    
    print("\n" + "="*60)
    print("TREND ANALYSIS")
    print("="*60)
    
    # Collect stats by surface
    surface_stats = defaultdict(lambda: {'wins': 0, 'total': 0, 'elo_diff_wins': [], 'elo_diff_losses': []})
    
    for name, pdata in data['players'].items():
        elo = pdata.get('elo_overall', 1500)
        for match in pdata.get('matches', []):
            surface = match.get('surface', 'Hard')
            result = match.get('result')
            
            surface_stats[surface]['total'] += 1
            if result == 'W':
                surface_stats[surface]['wins'] += 1
    
    print("\nWin rates by surface:")
    for surface, stats in surface_stats.items():
        if stats['total'] > 100:
            win_rate = stats['wins'] / stats['total']
            print(f"  {surface}: {win_rate:.1%} ({stats['total']} matches)")
    
    # Serve stats correlation with winning
    print("\nServe stat averages (winners vs losers):")
    winner_serve = {'ace': [], 'first_in': [], 'first_won': [], 'df': [], 'bp_saved': []}
    loser_serve = {'ace': [], 'first_in': [], 'first_won': [], 'df': [], 'bp_saved': []}
    
    for name, pdata in data['players'].items():
        for match in pdata.get('matches', []):
            serve = match.get('serve', {})
            target = winner_serve if match.get('result') == 'W' else loser_serve
            
            if serve.get('ace_pct') is not None:
                target['ace'].append(serve['ace_pct'])
            if serve.get('first_in_pct') is not None:
                target['first_in'].append(serve['first_in_pct'])
            if serve.get('first_won_pct') is not None:
                target['first_won'].append(serve['first_won_pct'])
            if serve.get('df_pct') is not None:
                target['df'].append(serve['df_pct'])
            if serve.get('bp_saved_pct') is not None:
                target['bp_saved'].append(serve['bp_saved_pct'])
    
    for stat in ['ace', 'first_in', 'first_won', 'df', 'bp_saved']:
        w_avg = np.mean(winner_serve[stat]) if winner_serve[stat] else 0
        l_avg = np.mean(loser_serve[stat]) if loser_serve[stat] else 0
        diff = w_avg - l_avg
        print(f"  {stat}: Winners {w_avg:.1f}% vs Losers {l_avg:.1f}% (diff: {diff:+.1f}%)")
    
    # Return stats
    print("\nReturn stat averages (winners vs losers):")
    winner_ret = {'rpw': [], 'bp_conv': []}
    loser_ret = {'rpw': [], 'bp_conv': []}
    
    for name, pdata in data['players'].items():
        for match in pdata.get('matches', []):
            ret = match.get('return', {})
            target = winner_ret if match.get('result') == 'W' else loser_ret
            
            if ret.get('rpw_pct') is not None:
                target['rpw'].append(ret['rpw_pct'])
            if ret.get('bp_conv_pct') is not None:
                target['bp_conv'].append(ret['bp_conv_pct'])
    
    for stat in ['rpw', 'bp_conv']:
        w_avg = np.mean(winner_ret[stat]) if winner_ret[stat] else 0
        l_avg = np.mean(loser_ret[stat]) if loser_ret[stat] else 0
        diff = w_avg - l_avg
        print(f"  {stat}: Winners {w_avg:.1f}% vs Losers {l_avg:.1f}% (diff: {diff:+.1f}%)")


def prob_to_odds(prob):
    """Convert probability to American odds"""
    if prob >= 0.5:
        return int(-100 * prob / (1 - prob))
    else:
        return int(100 * (1 - prob) / prob)


def prob_to_decimal(prob):
    """Convert probability to decimal odds"""
    return round(1 / prob, 2) if prob > 0 else float('inf')


class TennisPredictor:
    """Tennis match prediction model"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.data = None
        self.player_data = None
        self.name_lookup = None
    
    def _needs_retrain(self):
        """Check if model needs retraining (data newer than saved model)"""
        if not os.path.exists(WINNER_MODEL_PATH):
            return True
        if not os.path.exists(DATA_PATH):
            return True
        
        model_time = os.path.getmtime(WINNER_MODEL_PATH)
        data_time = os.path.getmtime(DATA_PATH)
        
        return data_time > model_time
    
    def _save_model(self):
        """Save trained model to disk"""
        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'player_data': self.player_data,
            'name_lookup': self.name_lookup
        }
        joblib.dump(save_data, WINNER_MODEL_PATH)
        print(f"Model saved to {WINNER_MODEL_PATH}")
    
    def _load_model(self):
        """Load trained model from disk"""
        save_data = joblib.load(WINNER_MODEL_PATH)
        self.model = save_data['model']
        self.scaler = save_data['scaler']
        self.player_data = save_data['player_data']
        self.name_lookup = save_data['name_lookup']
        print("Winner model loaded from cache")
        return True
    
    def train(self, force_retrain=False):
        """Train the model (or load from cache if available)"""
        
        # Try to load cached model
        if not force_retrain and not self._needs_retrain():
            try:
                self._load_model()
                return
            except Exception as e:
                print(f"Could not load cached model: {e}")
        
        # Train fresh
        self.data = load_data()
        
        # Build player lookup
        self.player_data = {}
        for name, info in self.data['players'].items():
            self.player_data[name] = {
                'elo_overall': info.get('elo_overall', 1500),
                'elo_hard': info.get('elo_hard', 1500),
                'elo_clay': info.get('elo_clay', 1500),
                'elo_grass': info.get('elo_grass', 1500),
                'matches': sorted(info.get('matches', []), key=lambda x: x.get('date', ''), reverse=True)
            }
        
        self.name_lookup = {}
        for name in self.player_data:
            self.name_lookup[name.lower()] = name
            self.name_lookup[name.lower().replace(' ', '')] = name
            parts = name.split()
            if len(parts) > 1:
                self.name_lookup[parts[-1].lower()] = name
        
        # Train model
        X, y, _ = build_training_data(self.data)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = LogisticRegression(max_iter=1000, C=0.1)
        self.model.fit(X_scaled, y)
        
        print(f"Model trained on {len(X)} matches")
        
        # Save for next time
        self._save_model()
    
    def find_player(self, name):
        """Find player by name"""
        key = name.lower().replace(' ', '')
        if key in self.name_lookup:
            return self.name_lookup[key]
        # Try partial match
        for k, v in self.name_lookup.items():
            if name.lower() in k:
                return v
        return None
    
    def predict(self, player1: str, player2: str, surface: str = 'Hard'):
        """
        Predict match outcome between two players.
        
        Returns dict with probabilities and odds for both players.
        """
        if not self.model:
            raise ValueError("Model not trained. Call train() first.")
        
        # Find players
        p1_name = self.find_player(player1)
        p2_name = self.find_player(player2)
        
        if not p1_name:
            raise ValueError(f"Player not found: {player1}")
        if not p2_name:
            raise ValueError(f"Player not found: {player2}")
        
        p1_data = self.player_data[p1_name]
        p2_data = self.player_data[p2_name]
        
        # Get recent matches
        p1_matches = p1_data['matches']
        p2_matches = p2_data['matches']
        
        surface = surface.lower()
        surface_elo_key = f'elo_{surface}' if surface in ['hard', 'clay', 'grass'] else 'elo_hard'
        
        # Calculate stats
        p1_form = calculate_rolling_stats(p1_matches, 15)
        p2_form = calculate_rolling_stats(p2_matches, 15)
        p1_surf = calculate_surface_form(p1_matches, surface, 15)
        p2_surf = calculate_surface_form(p2_matches, surface, 15)
        p1_blend = calculate_blended_stats(p1_matches, surface, 15, 0.6)
        p2_blend = calculate_blended_stats(p2_matches, surface, 15, 0.6)
        p1_h2h = calculate_head_to_head(p1_matches, p2_name)
        
        if not all([p1_form, p2_form, p1_blend, p2_blend]):
            raise ValueError("Not enough match data for prediction")
        
        # Build feature vector (same as training)
        features = [
            p1_data['elo_overall'] - p2_data['elo_overall'],
            p1_data.get(surface_elo_key, 1500) - p2_data.get(surface_elo_key, 1500),
            p1_data['elo_overall'],
            p2_data['elo_overall'],
            p1_form['win_pct'] - p2_form['win_pct'],
            p1_form['avg_dr'] - p2_form['avg_dr'],
            p1_form['win_pct'],
            p2_form['win_pct'],
            p1_form['first_in_pct'] - p2_form['first_in_pct'],
            p1_form['first_won_pct'] - p2_form['first_won_pct'],
            p1_form['ace_pct'] - p2_form['ace_pct'],
            p1_form['df_pct'] - p2_form['df_pct'],
            p1_form['bp_saved_pct'] - p2_form['bp_saved_pct'],
            p1_form['rpw_pct'] - p2_form['rpw_pct'],
            p1_form['bp_conv_pct'] - p2_form['bp_conv_pct'],
            (p1_surf['win_pct'] if p1_surf else 0.5) - (p2_surf['win_pct'] if p2_surf else 0.5),
            0,  # fatigue time diff (unknown for future match)
            0,  # fatigue matches diff
            p1_blend['first_won_pct'] - p2_blend['first_won_pct'],
            p1_blend['ace_pct'] - p2_blend['ace_pct'],
            p1_blend['bp_saved_pct'] - p2_blend['bp_saved_pct'],
            p1_blend['rpw_pct'] - p2_blend['rpw_pct'],
            p1_blend['bp_conv_pct'] - p2_blend['bp_conv_pct'],
            p1_blend['win_pct'] - p2_blend['win_pct'],
            p1_blend['avg_dr'] - p2_blend['avg_dr'],
            p1_blend.get('surface_match_count', 0) - p2_blend.get('surface_match_count', 0),
            p1_blend.get('surface_match_count', 0),
            p2_blend.get('surface_match_count', 0),
            p1_h2h['diff'],
            p1_h2h['win_pct'],
            p1_h2h['total'],
        ]
        
        # Scale and predict
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        
        p1_prob = self.model.predict_proba(X_scaled)[0][1]
        p2_prob = 1 - p1_prob
        
        return {
            'player1': {
                'name': p1_name,
                'win_prob': round(p1_prob * 100, 1),
                'american_odds': prob_to_odds(p1_prob),
                'decimal_odds': prob_to_decimal(p1_prob),
                'elo': p1_data['elo_overall'],
                'surface_elo': p1_data.get(surface_elo_key, 1500),
            },
            'player2': {
                'name': p2_name,
                'win_prob': round(p2_prob * 100, 1),
                'american_odds': prob_to_odds(p2_prob),
                'decimal_odds': prob_to_decimal(p2_prob),
                'elo': p2_data['elo_overall'],
                'surface_elo': p2_data.get(surface_elo_key, 1500),
            },
            'surface': surface.capitalize(),
            'h2h': f"{p1_h2h['wins']}-{p1_h2h['losses']}" if p1_h2h['total'] > 0 else "No previous meetings"
        }
    
    def print_prediction(self, player1: str, player2: str, surface: str = 'Hard'):
        """Print formatted prediction"""
        result = self.predict(player1, player2, surface)
        
        p1 = result['player1']
        p2 = result['player2']
        
        print("\n" + "="*60)
        print(f"MATCH PREDICTION - {result['surface']} Court")
        print("="*60)
        print(f"\n{p1['name']} vs {p2['name']}")
        print(f"Head-to-Head: {result['h2h']}")
        print()
        print(f"{'Player':<25} {'Win %':>8} {'American':>10} {'Decimal':>10}")
        print("-"*55)
        print(f"{p1['name']:<25} {p1['win_prob']:>7.1f}% {p1['american_odds']:>+10} {p1['decimal_odds']:>10.2f}")
        print(f"{p2['name']:<25} {p2['win_prob']:>7.1f}% {p2['american_odds']:>+10} {p2['decimal_odds']:>10.2f}")
        print()
        print(f"ELO: {p1['name']} {p1['elo']} | {p2['name']} {p2['elo']}")
        print(f"Surface ELO: {p1['name']} {p1['surface_elo']} | {p2['name']} {p2['surface_elo']}")
        print("="*60)
        
        return result


if __name__ == '__main__':
    # Train the model
    predictor = TennisPredictor()
    predictor.train()
    
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    
    # Example predictions
    examples = [
        ("Aryna Sabalenka", "Iga Swiatek", "Hard"),
        ("Coco Gauff", "Naomi Osaka", "Hard"),
        ("Jasmine Paolini", "Elena Rybakina", "Clay"),
    ]
    
    for p1, p2, surface in examples:
        try:
            predictor.print_prediction(p1, p2, surface)
        except Exception as e:
            print(f"Error predicting {p1} vs {p2}: {e}")
