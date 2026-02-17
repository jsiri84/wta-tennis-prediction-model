"""Evaluate unified model with 5-fold cross-validation"""

from unified_model import build_training_data, load_data
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
import numpy as np


def main():
    # Load data
    data = load_data()
    print(f"Loaded {data['player_count']} players, {data['total_matches']} total matches")
    print()

    # Build features (returns 5 values)
    X_spread, X_total, y_spread, y_total, match_info = build_training_data(data)
    print(f"Training samples: {len(X_spread)}")
    print(f"Features per sample: {X_spread.shape[1]}")
    print()

    # Create winner labels from spread (positive spread = player 1 wins)
    y_winner = (y_spread > 0).astype(int)

    # Winner model (5-fold CV)
    print("=" * 50)
    print("WINNER PREDICTION (Logistic Regression)")
    print("=" * 50)
    winner_model = LogisticRegression(max_iter=1000, C=0.1)
    winner_scores = cross_val_score(winner_model, X_spread, y_winner, cv=5, scoring='accuracy')
    winner_auc = cross_val_score(winner_model, X_spread, y_winner, cv=5, scoring='roc_auc')
    print(f"Accuracy: {winner_scores.mean()*100:.1f}% (+/- {winner_scores.std()*100:.1f}%)")
    print(f"AUC-ROC:  {winner_auc.mean():.3f} (+/- {winner_auc.std():.3f})")
    print()

    # Spread model (5-fold CV)
    print("=" * 50)
    print("SPREAD PREDICTION (Gradient Boosting)")
    print("=" * 50)
    spread_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    spread_preds = cross_val_predict(spread_model, X_spread, y_spread, cv=5)
    spread_mae = np.mean(np.abs(spread_preds - y_spread))
    spread_std = np.std(np.abs(spread_preds - y_spread))
    print(f"MAE: {spread_mae:.2f} games (+/- {spread_std:.2f})")

    # Direction accuracy
    correct_dir = np.sum((spread_preds > 0) == (y_spread > 0))
    print(f"Direction Accuracy: {correct_dir/len(y_spread)*100:.1f}%")
    print()

    # Total games model (5-fold CV)
    print("=" * 50)
    print("TOTAL GAMES PREDICTION (Gradient Boosting)")
    print("=" * 50)
    total_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    total_preds = cross_val_predict(total_model, X_total, y_total, cv=5)
    total_mae = np.mean(np.abs(total_preds - y_total))
    total_std = np.std(np.abs(total_preds - y_total))
    print(f"MAE: {total_mae:.2f} games (+/- {total_std:.2f})")

    # Over/under accuracy at common lines
    print()
    print("Over/Under Accuracy:")
    for line in [20.5, 21.5, 22.5]:
        actual_over = y_total > line
        pred_over = total_preds > line
        correct = np.sum(actual_over == pred_over)
        print(f"  Line {line}: {correct/len(y_total)*100:.1f}%")

    # Summary stats
    print()
    print("=" * 50)
    print("DATA SUMMARY")
    print("=" * 50)
    print(f"Spread range: {y_spread.min():.0f} to {y_spread.max():.0f} games")
    print(f"Spread mean: {y_spread.mean():.1f} games")
    print(f"Total games range: {y_total.min():.0f} to {y_total.max():.0f} games")
    print(f"Total games mean: {y_total.mean():.1f} games")


def test_predictions():
    """Test the trained model with sample predictions"""
    from unified_model import UnifiedPredictor
    
    # Retrain with fresh data
    predictor = UnifiedPredictor()
    predictor.train(force_retrain=True)
    
    print()
    print("=" * 50)
    print("SAMPLE PREDICTIONS")
    print("=" * 50)
    
    # Test a few matchups
    matchups = [
        ('Aryna Sabalenka', 'Iga Swiatek', 'Hard'),
        ('Coco Gauff', 'Jessica Pegula', 'Hard'),
        ('Emma Navarro', 'Madison Keys', 'Hard'),
        ('Elena Rybakina', 'Paula Badosa', 'Hard'),
    ]
    
    for p1, p2, surface in matchups:
        result = predictor.predict(p1, p2, surface)
        winner = p1 if result['p1_win_prob'] > 50 else p2
        win_prob = max(result['p1_win_prob'], result['p2_win_prob'])
        print(f"{p1} vs {p2} ({surface}):")
        print(f"  Winner: {winner} ({win_prob:.0f}%)")
        print(f"  Spread: {result['spread']:.1f} games (favors {result['spread_favors']})")
        print(f"  Total:  {result['total']:.1f} games")
        print()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--predict':
        test_predictions()
    else:
        main()
        test_predictions()
