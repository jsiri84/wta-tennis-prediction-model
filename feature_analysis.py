"""Analyze features and test different feature subsets for winner model"""

from prediction_model import build_training_data, load_data
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, SelectKBest, f_classif
import numpy as np

# Load data
print("Loading data...")
data = load_data()
X, y, info = build_training_data(data)
X = np.array(X)
y = np.array(y)

print(f"Total samples: {len(y)}")
print(f"Total features: {X.shape[1]}")
print()

# Feature names (from prediction_model.py build_training_data)
feature_names = [
    'elo_diff',                    # 0
    'surface_elo_diff',            # 1
    'win_pct_diff',                # 2
    'avg_spread_diff',             # 3
    'spread_std_diff',             # 4
    'straight_set_diff',           # 5
    'three_set_diff',              # 6
    'first_in_diff',               # 7
    'first_won_diff',              # 8
    'second_won_diff',             # 9
    'ace_diff',                    # 10
    'df_diff',                     # 11
    'bp_saved_diff',               # 12
    'rpw_diff',                    # 13
    'v_first_won_diff',            # 14
    'v_second_won_diff',           # 15
    'bp_conv_diff',                # 16
    'dr_diff',                     # 17
    'h2h_diff',                    # 18
    'h2h_win_pct',                 # 19
    'big_tourn_win_diff',          # 20
    'upset_rate_diff',             # 21
    'vs_strong_win_pct_diff',      # 22
    'vs_strong_matches_diff',      # 23
    'adj_win_pct_diff',            # 24
    'avg_opp_elo_diff',            # 25
    'level_jump_diff',             # 26
    'level_jump_pct_diff',         # 27
    'avg_recent_opp_elo_diff',     # 28
    'perf_at_level_win_pct_diff',  # 29
    'perf_at_level_spread_diff',   # 30
    'perf_at_level_serve_diff',    # 31
    'perf_at_level_return_diff',   # 32
    'form_diff',                   # 33
    'surface_form_diff',           # 34
    'spread_form_diff',            # 35
    'streak_diff',                 # 36
]

print("="*60)
print("1. FEATURE COEFFICIENTS (Logistic Regression)")
print("="*60)

model = LogisticRegression(max_iter=1000, C=0.1)
model.fit(X, y)

# Sort by absolute coefficient
coef_idx = np.argsort(np.abs(model.coef_[0]))[::-1]
print(f"\n{'Feature':<35} {'Coefficient':>12}")
print("-"*50)
for i in coef_idx:
    name = feature_names[i] if i < len(feature_names) else f'feature_{i}'
    print(f"{name:<35} {model.coef_[0][i]:>12.4f}")

print()
print("="*60)
print("2. FEATURE ABLATION TEST")
print("="*60)

# Test with different feature counts
print("\nTesting accuracy with top N features (by coefficient magnitude)...")

results = []
for n_features in [5, 10, 15, 20, 25, 30, 35, 37]:
    if n_features > X.shape[1]:
        continue
    
    top_idx = coef_idx[:n_features]
    X_subset = X[:, top_idx]
    
    scores = cross_val_score(model, X_subset, y, cv=5, scoring='accuracy')
    results.append((n_features, scores.mean(), scores.std()))
    print(f"  Top {n_features:2d} features: {scores.mean()*100:.1f}% (+/- {scores.std()*100:.1f}%)")

print()
print("="*60)
print("3. RECURSIVE FEATURE ELIMINATION")
print("="*60)

# Use RFE to find optimal feature subset
print("\nFinding optimal features with RFE...")
rfe = RFE(LogisticRegression(max_iter=1000, C=0.1), n_features_to_select=15)
rfe.fit(X, y)

selected_features = [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]
print(f"\nRFE selected {len(selected_features)} features:")
for f in selected_features:
    print(f"  - {f}")

X_rfe = X[:, rfe.support_]
scores_rfe = cross_val_score(LogisticRegression(max_iter=1000, C=0.1), X_rfe, y, cv=5, scoring='accuracy')
print(f"\nRFE accuracy: {scores_rfe.mean()*100:.1f}% (+/- {scores_rfe.std()*100:.1f}%)")

print()
print("="*60)
print("4. CORE FEATURES ONLY")
print("="*60)

# Test with just the core predictive features
core_features = [0, 1, 2, 3, 8, 9, 12, 13, 16, 17, 18, 19]  # ELO, form, serve/return, H2H
core_names = [feature_names[i] for i in core_features]
print("\nCore features only:")
for f in core_names:
    print(f"  - {f}")

X_core = X[:, core_features]
scores_core = cross_val_score(LogisticRegression(max_iter=1000, C=0.1), X_core, y, cv=5, scoring='accuracy')
print(f"\nCore features accuracy: {scores_core.mean()*100:.1f}% (+/- {scores_core.std()*100:.1f}%)")

print()
print("="*60)
print("5. MINIMAL ELO-ONLY BASELINE")
print("="*60)

X_elo = X[:, [0, 1]]  # Just ELO diff and surface ELO diff
scores_elo = cross_val_score(LogisticRegression(max_iter=1000, C=0.1), X_elo, y, cv=5, scoring='accuracy')
print(f"ELO-only accuracy: {scores_elo.mean()*100:.1f}% (+/- {scores_elo.std()*100:.1f}%)")

print()
print("="*60)
print("SUMMARY")
print("="*60)
print(f"\n{'Model':<30} {'Accuracy':>10}")
print("-"*42)
print(f"{'ELO only (2 features)':<30} {scores_elo.mean()*100:>9.1f}%")
print(f"{'Core (12 features)':<30} {scores_core.mean()*100:>9.1f}%")
print(f"{'RFE selected (15 features)':<30} {scores_rfe.mean()*100:>9.1f}%")
print(f"{'All features (37)':<30} {cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()*100:>9.1f}%")

# Find best from ablation
best = max(results, key=lambda x: x[1])
print(f"\nBest from ablation: Top {best[0]} features at {best[1]*100:.1f}%")
