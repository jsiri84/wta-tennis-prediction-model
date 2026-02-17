"""
Analyze 2025 WTA results and compare model predictions to Pinnacle odds.
Calculates CLV (Closing Line Value) and prediction accuracy.
"""
import pandas as pd
import numpy as np
import sys
import os
import re
sys.path.insert(0, '.')
from skill_model import SkillPredictor

def odds_to_prob(odds):
    """Convert decimal odds to implied probability."""
    if pd.isna(odds) or odds <= 1:
        return None
    return 1 / odds

def prob_to_odds(prob):
    """Convert probability to decimal odds."""
    if prob <= 0 or prob >= 1:
        return None
    return 1 / prob

def american_odds(prob):
    """Convert probability to American odds."""
    if prob >= 0.5:
        return int(-100 * prob / (1 - prob))
    else:
        return int(100 * (1 - prob) / prob)

class NameMatcher:
    """Match player names between different formats."""
    
    def __init__(self, model_players):
        self.model_players = model_players
        self.cache = {}
        self._build_lookup()
    
    def _build_lookup(self):
        """Build reverse lookup from last name + initial to full name."""
        self.lastname_lookup = {}
        self.lastname_first_lookup = {}
        
        for full_name in self.model_players:
            parts = full_name.split()
            if len(parts) >= 2:
                first = parts[0]
                last = ' '.join(parts[1:])  # Handle multi-part last names
                initial = first[0].upper()
                
                # Create key like "Keys M" or "Swiatek I"
                key = f"{last.lower()} {initial.lower()}"
                self.lastname_lookup[key] = full_name
                
                # Also just last name
                self.lastname_first_lookup[last.lower()] = full_name
    
    def match(self, short_name):
        """Convert short name like 'Keys M.' to full name 'Madison Keys'."""
        if short_name in self.cache:
            return self.cache[short_name]
        
        # Parse short name (e.g., "Keys M." or "Van de Zande R.")
        # Format: LASTNAME INITIAL.
        match = re.match(r'^(.+?)\s+([A-Z])\.$', short_name.strip())
        if not match:
            self.cache[short_name] = None
            return None
        
        lastname = match.group(1)
        initial = match.group(2)
        
        # Try exact match
        key = f"{lastname.lower()} {initial.lower()}"
        if key in self.lastname_lookup:
            result = self.lastname_lookup[key]
            self.cache[short_name] = result
            return result
        
        # Try fuzzy matching on last name only
        for model_name in self.model_players:
            parts = model_name.split()
            if len(parts) >= 2:
                model_last = ' '.join(parts[1:])
                model_first = parts[0]
                
                # Check if last names match and first initial matches
                if (model_last.lower() == lastname.lower() and 
                    model_first[0].upper() == initial):
                    self.cache[short_name] = model_name
                    return model_name
        
        self.cache[short_name] = None
        return None


print("="*70)
print("2025 WTA RESULTS ANALYSIS - MODEL vs PINNACLE")
print("="*70)

# Load model
print("\nLoading model...")
model = SkillPredictor(elo_weight=0.5)
model.train()

# Create name matcher
matcher = NameMatcher(list(model.player_skills.keys()))

# Load 2025 data from project data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
print("Loading 2025 WTA data...")
df = pd.read_excel(os.path.join(DATA_DIR, 'wta_2025_results.xlsx'))
print(f"Total matches in file: {len(df)}")

# Filter to completed matches with Pinnacle odds
df = df[df['Comment'] == 'Completed'].copy()
df = df.dropna(subset=['PSW', 'PSL'])
print(f"Completed matches with Pinnacle odds: {len(df)}")

# Convert date
df['Date'] = pd.to_datetime(df['Date'])
df['DateStr'] = df['Date'].dt.strftime('%Y%m%d')

# Test name matching
print("\nTesting name matching:")
test_names = ['Keys M.', 'Sabalenka A.', 'Swiatek I.', 'Gauff C.', 'Osaka N.']
for name in test_names:
    matched = matcher.match(name)
    print(f"  '{name}' -> '{matched}'")

# Results storage
results = []
errors = []
matched_count = 0
unmatched_players = set()

print("\nGenerating predictions for each match...")
for idx, row in df.iterrows():
    winner_short = row['Winner']
    loser_short = row['Loser']
    surface = row['Surface']
    tournament = row['Tournament']
    date_str = row['DateStr']
    round_name = row['Round']
    
    # Match names
    winner = matcher.match(winner_short)
    loser = matcher.match(loser_short)
    
    if not winner:
        unmatched_players.add(winner_short)
        continue
    if not loser:
        unmatched_players.add(loser_short)
        continue
    
    matched_count += 1
    
    # Pinnacle odds
    pin_winner_odds = row['PSW']
    pin_loser_odds = row['PSL']
    pin_winner_prob = odds_to_prob(pin_winner_odds)
    pin_loser_prob = odds_to_prob(pin_loser_odds)
    
    if pin_winner_prob is None:
        continue
    
    # Get model prediction (predicting winner vs loser)
    try:
        pred = model.predict(winner, loser, surface, 
                            tournament=tournament, match_date=date_str,
                            match_round=round_name, n_sims=500)
        
        model_winner_prob = pred['p1_win_prob'] / 100
        model_loser_prob = pred['p2_win_prob'] / 100
        
        # Model odds
        model_winner_odds = prob_to_odds(model_winner_prob) if model_winner_prob > 0 else None
        model_loser_odds = prob_to_odds(model_loser_prob) if model_loser_prob > 0 else None
        
        # CLV = Model implied prob - Pinnacle implied prob (for winner)
        # Positive CLV means model had winner at better odds than Pinnacle close
        clv_winner = (model_winner_prob - pin_winner_prob) * 100 if pin_winner_prob else None
        
        # Did model predict correctly?
        # Model predicts winner correctly if model_winner_prob > 0.5
        model_correct = model_winner_prob > 0.5
        
        # Model's pick
        model_pick = winner if model_winner_prob > 0.5 else loser
        model_pick_short = winner_short if model_winner_prob > 0.5 else loser_short
        model_pick_prob = model_winner_prob if model_winner_prob > 0.5 else model_loser_prob
        model_pick_odds = model_winner_odds if model_winner_prob > 0.5 else model_loser_odds
        pin_pick_odds = pin_winner_odds if model_winner_prob > 0.5 else pin_loser_odds
        
        # CLV on model's pick
        pin_pick_prob = odds_to_prob(pin_pick_odds)
        clv_pick = (model_pick_prob - pin_pick_prob) * 100 if pin_pick_prob else None
        
        # Did model's pick win?
        pick_won = (model_pick == winner)
        
        results.append({
            'date': row['Date'],
            'tournament': tournament,
            'surface': surface,
            'round': round_name,
            'winner': winner_short,
            'loser': loser_short,
            'pin_winner_odds': pin_winner_odds,
            'pin_loser_odds': pin_loser_odds,
            'model_winner_prob': model_winner_prob,
            'model_winner_odds': model_winner_odds,
            'model_loser_odds': model_loser_odds,
            'model_pick': model_pick_short,
            'model_pick_prob': model_pick_prob,
            'model_pick_odds': model_pick_odds,
            'pin_pick_odds': pin_pick_odds,
            'clv_pick': clv_pick,
            'model_correct': model_correct,
            'pick_won': pick_won,
        })
        
    except Exception as e:
        errors.append({'winner': winner, 'loser': loser, 'error': str(e)})

print(f"\nMatched both players: {matched_count} matches")
print(f"Successfully analyzed: {len(results)} matches")
print(f"Errors: {len(errors)}")
print(f"Unmatched players: {len(unmatched_players)}")

if len(results) == 0:
    print("\nNo results to analyze!")
    print("Sample unmatched players:")
    for p in list(unmatched_players)[:20]:
        print(f"  {p}")
    sys.exit(1)

# Convert to DataFrame
results_df = pd.DataFrame(results)

# ============================================
# ACCURACY ANALYSIS
# ============================================
print("\n" + "="*70)
print("PREDICTION ACCURACY")
print("="*70)

total = len(results_df)
correct = results_df['model_correct'].sum()
accuracy = correct / total * 100

print(f"\nOverall Accuracy: {accuracy:.1f}% ({correct}/{total})")

# Accuracy by confidence
print("\nAccuracy by Model Confidence:")
for thresh in [50, 55, 60, 65, 70, 75, 80]:
    mask = results_df['model_pick_prob'] >= thresh/100
    subset = results_df[mask]
    if len(subset) > 0:
        acc = subset['pick_won'].sum() / len(subset) * 100
        print(f"  {thresh}%+ confidence: {acc:.1f}% ({subset['pick_won'].sum()}/{len(subset)})")

# Accuracy by surface
print("\nAccuracy by Surface:")
for surface in results_df['surface'].unique():
    mask = results_df['surface'] == surface
    subset = results_df[mask]
    if len(subset) > 10:
        acc = subset['model_correct'].sum() / len(subset) * 100
        print(f"  {surface}: {acc:.1f}% ({subset['model_correct'].sum()}/{len(subset)})")

# ============================================
# CLV ANALYSIS
# ============================================
print("\n" + "="*70)
print("CLOSING LINE VALUE (CLV) ANALYSIS")
print("="*70)

# CLV on model's picks
clv_data = results_df.dropna(subset=['clv_pick'])
avg_clv = clv_data['clv_pick'].mean()
median_clv = clv_data['clv_pick'].median()
positive_clv = (clv_data['clv_pick'] > 0).sum()
negative_clv = (clv_data['clv_pick'] < 0).sum()

print(f"\nCLV on Model's Picks:")
print(f"  Average CLV: {avg_clv:+.2f}%")
print(f"  Median CLV: {median_clv:+.2f}%")
print(f"  Positive CLV: {positive_clv} matches ({positive_clv/len(clv_data)*100:.1f}%)")
print(f"  Negative CLV: {negative_clv} matches ({negative_clv/len(clv_data)*100:.1f}%)")

# CLV by confidence level
print("\nCLV by Confidence Level:")
for thresh in [50, 55, 60, 65, 70, 75]:
    mask = clv_data['model_pick_prob'] >= thresh/100
    subset = clv_data[mask]
    if len(subset) > 0:
        print(f"  {thresh}%+ conf: avg CLV {subset['clv_pick'].mean():+.2f}%, n={len(subset)}")

# High CLV picks performance
print("\nHigh CLV Picks Performance (CLV > 5%):")
high_clv = results_df[results_df['clv_pick'] > 5]
if len(high_clv) > 0:
    acc = high_clv['pick_won'].sum() / len(high_clv) * 100
    print(f"  Accuracy: {acc:.1f}% ({high_clv['pick_won'].sum()}/{len(high_clv)})")
    print(f"  Avg CLV: {high_clv['clv_pick'].mean():.1f}%")

# Negative CLV picks performance (model disagrees with market)
print("\nNegative CLV Picks (Model < Market, CLV < -5%):")
neg_clv = results_df[results_df['clv_pick'] < -5]
if len(neg_clv) > 0:
    acc = neg_clv['pick_won'].sum() / len(neg_clv) * 100
    print(f"  Accuracy: {acc:.1f}% ({neg_clv['pick_won'].sum()}/{len(neg_clv)})")
    print(f"  Avg CLV: {neg_clv['clv_pick'].mean():.1f}%")

# ============================================
# UPSET ANALYSIS
# ============================================
print("\n" + "="*70)
print("UPSET DETECTION")
print("="*70)

# Upsets where underdog (Pinnacle odds > 2.0) won
upsets = results_df[results_df['pin_winner_odds'] > 2.0]
print(f"\nTotal upsets (winner was underdog at 2.0+ odds): {len(upsets)}")

# Did model catch any upsets?
model_caught_upsets = upsets[upsets['model_correct'] == True]
print(f"Model correctly predicted upset: {len(model_caught_upsets)} ({len(model_caught_upsets)/len(upsets)*100:.1f}%)")

# ============================================
# SAMPLE PREDICTIONS
# ============================================
print("\n" + "="*70)
print("SAMPLE PREDICTIONS (Recent matches)")
print("="*70)

# Show some recent predictions
sample = results_df.tail(20)[['date', 'winner', 'loser', 'model_pick', 
                              'model_pick_prob', 'pin_pick_odds', 'clv_pick', 'pick_won']]
sample = sample.copy()
sample['model_pick_prob'] = sample['model_pick_prob'].apply(lambda x: f"{x*100:.0f}%")
sample['clv_pick'] = sample['clv_pick'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
sample['pick_won'] = sample['pick_won'].apply(lambda x: "W" if x else "L")
sample['date'] = sample['date'].dt.strftime('%m/%d')
print(sample.to_string(index=False))

# ============================================
# BETTING SIMULATION
# ============================================
print("\n" + "="*70)
print("FLAT BETTING SIMULATION (if betting model's pick)")
print("="*70)

# Simulate flat $100 bets on model's pick at Pinnacle odds
bankroll = 0
bets = 0
wins = 0

for _, row in results_df.iterrows():
    if pd.isna(row['pin_pick_odds']):
        continue
    
    stake = 100
    bets += 1
    
    if row['pick_won']:
        profit = stake * (row['pin_pick_odds'] - 1)
        bankroll += profit
        wins += 1
    else:
        bankroll -= stake

print(f"\nAll Picks:")
print(f"  Total bets: {bets}")
print(f"  Wins: {wins} ({wins/bets*100:.1f}%)")
print(f"  Net P/L: ${bankroll:,.2f}")
print(f"  ROI: {bankroll/(bets*100)*100:.2f}%")

# Filter to high confidence picks only
print("\nHigh Confidence Only (65%+ model probability):")
high_conf = results_df[results_df['model_pick_prob'] >= 0.65]
bankroll_hc = 0
bets_hc = 0
wins_hc = 0

for _, row in high_conf.iterrows():
    if pd.isna(row['pin_pick_odds']):
        continue
    
    stake = 100
    bets_hc += 1
    
    if row['pick_won']:
        profit = stake * (row['pin_pick_odds'] - 1)
        bankroll_hc += profit
        wins_hc += 1
    else:
        bankroll_hc -= stake

if bets_hc > 0:
    print(f"  Total bets: {bets_hc}")
    print(f"  Wins: {wins_hc} ({wins_hc/bets_hc*100:.1f}%)")
    print(f"  Net P/L: ${bankroll_hc:,.2f}")
    print(f"  ROI: {bankroll_hc/(bets_hc*100)*100:.2f}%")

# Filter to positive CLV picks only
print("\nPositive CLV Only (model more confident than Pinnacle):")
pos_clv = results_df[results_df['clv_pick'] > 0]
bankroll_clv = 0
bets_clv = 0
wins_clv = 0

for _, row in pos_clv.iterrows():
    if pd.isna(row['pin_pick_odds']):
        continue
    
    stake = 100
    bets_clv += 1
    
    if row['pick_won']:
        profit = stake * (row['pin_pick_odds'] - 1)
        bankroll_clv += profit
        wins_clv += 1
    else:
        bankroll_clv -= stake

if bets_clv > 0:
    print(f"  Total bets: {bets_clv}")
    print(f"  Wins: {wins_clv} ({wins_clv/bets_clv*100:.1f}%)")
    print(f"  Net P/L: ${bankroll_clv:,.2f}")
    print(f"  ROI: {bankroll_clv/(bets_clv*100)*100:.2f}%")

# Save detailed results
output_file = os.path.join(DATA_DIR, 'analysis_2025_results.csv')
results_df.to_csv(output_file, index=False)
print(f"\n\nDetailed results saved to: {output_file}")
