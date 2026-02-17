"""Check spread distribution and model predictions."""
from unified_model import UnifiedPredictor, load_data, build_training_data
import numpy as np

# Load data
data = load_data()
X_spread, X_total, y_spread, y_total, match_info = build_training_data(data)

print("TRAINING DATA SPREAD DISTRIBUTION:")
print(f"  Min: {np.min(y_spread):.1f}")
print(f"  Max: {np.max(y_spread):.1f}")
print(f"  Mean: {np.mean(y_spread):.1f}")
print(f"  Std: {np.std(y_spread):.1f}")
print(f"  Median: {np.median(y_spread):.1f}")

# Check distribution
abs_spread = np.abs(y_spread)
print(f"\n  Spreads < 3: {np.sum(abs_spread < 3)} ({100*np.mean(abs_spread < 3):.1f}%)")
print(f"  Spreads 3-5: {np.sum((abs_spread >= 3) & (abs_spread < 5))} ({100*np.mean((abs_spread >= 3) & (abs_spread < 5)):.1f}%)")
print(f"  Spreads 5-7: {np.sum((abs_spread >= 5) & (abs_spread < 7))} ({100*np.mean((abs_spread >= 5) & (abs_spread < 7)):.1f}%)")
print(f"  Spreads 7-9: {np.sum((abs_spread >= 7) & (abs_spread < 9))} ({100*np.mean((abs_spread >= 7) & (abs_spread < 9)):.1f}%)")
print(f"  Spreads >= 9: {np.sum(abs_spread >= 9)} ({100*np.mean(abs_spread >= 9):.1f}%)")

# Load model and test predictions
p = UnifiedPredictor()
p.train()

print("\n\nTEST PREDICTIONS (checking spread variance):")
test_matches = [
    ("Diana Shnaider", "Alycia Parks", "Hard", "+277 ELO"),
    ("Laura Siegemund", "Varvara Gracheva", "Hard", "+11 ELO"),
    ("Aryna Sabalenka", "Emma Raducanu", "Hard", "+377 ELO"),
    ("Karolina Muchova", "Jaqueline Cristian", "Hard", "+164 ELO"),
    ("Daria Kasatkina", "Moyuka Uchijima", "Hard", "+160 ELO"),
]

for p1, p2, surface, note in test_matches:
    try:
        r = p.predict(p1, p2, surface)
        spread = r["spread"]
        win_prob = r["p1_win_prob"]
        print(f"  {p1} vs {p2} ({note})")
        print(f"    Spread: {spread:.1f}, Win prob: {win_prob:.1f}%")
    except Exception as e:
        print(f"  {p1} vs {p2}: {e}")

# Check model feature importances
print("\n\nSPREAD MODEL FEATURE IMPORTANCES (top 10):")
feature_names = [
    'elo_diff', 'surface_elo_diff', 'hist_spread_diff', 'p1_avg_spread', 'p2_avg_spread',
    'spread_volatility', 'combined_dom_margins', 'win_pct_diff', 'dr_diff', 'surface_win_diff',
    'first_won_diff', 'ace_diff', 'second_won_diff', 'bp_saved_diff', 'rpw_diff', 'bp_conv_diff',
    'straight_set_diff', 'win_pct_gap_abs', 'h2h_diff', 'h2h_win_pct', 'h2h_total',
    'vs_strong_diff', 'exp_vs_strong_diff', 'adj_win_pct_diff', 'avg_opp_elo_diff',
    'level_jump_diff', 'level_jump_pct_diff', 'win_at_level_diff', 'spread_at_level_diff',
    'serve_at_level_diff', 'return_at_level_diff', 'form_diff', 'surface_form_diff',
    'spread_form_diff', 'streak_diff',
]

importances = p.spread_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

for i in sorted_idx[:10]:
    if i < len(feature_names):
        print(f"  {feature_names[i]}: {importances[i]:.3f}")
    else:
        print(f"  Feature {i}: {importances[i]:.3f}")
