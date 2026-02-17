"""Test the skill model against actual match outcomes."""

import json
import numpy as np
from skill_model import SkillPredictor, load_data


def test_accuracy(elo_weight=0.5, use_fatigue=False):
    """
    Test model accuracy on recent matches.
    
    Args:
        elo_weight: Weight for ELO blending (0-1)
        use_fatigue: Whether to include fatigue in predictions
    """
    model = SkillPredictor(elo_weight=elo_weight)
    model.train()
    
    # Load data to get matches
    data = load_data()
    
    correct = 0
    total = 0
    results_by_prob = {50: [0, 0], 55: [0, 0], 60: [0, 0], 65: [0, 0], 70: [0, 0], 75: [0, 0], 80: [0, 0]}
    
    errors = []
    fatigue_impacts = []  # Track how often fatigue changed the prediction
    
    for player_name, player_info in data['players'].items():
        for match in player_info.get('matches', [])[:10]:  # Recent 10 matches per player
            opponent = match.get('opponent', '')
            surface = match.get('surface', 'Hard')
            result = match.get('result', '')
            match_date = match.get('date', '')
            tournament = match.get('tournament', '')
            
            if not opponent or result not in ('W', 'L'):
                continue
            
            try:
                # Get prediction with or without fatigue
                match_round = match.get('round', '')
                if use_fatigue and tournament and match_date:
                    pred = model.predict(player_name, opponent, surface, 
                                        tournament=tournament, match_date=match_date,
                                        match_round=match_round, n_sims=1000)
                    # Track fatigue impact (v2 uses serve_adj instead of fatigue_penalty)
                    p1_adj = pred.get('p1_serve_adj', 0) or 0
                    p2_adj = pred.get('p2_serve_adj', 0) or 0
                    if abs(p1_adj) > 0.001 or abs(p2_adj) > 0.001:
                        fatigue_impacts.append({
                            'p1_adj': p1_adj,
                            'p2_adj': p2_adj,
                            'net': pred.get('fatigue_adjustment', 0)
                        })
                else:
                    pred = model.predict(player_name, opponent, surface, n_sims=1000)
                
                # Did player win?
                actual_win = 1 if result == 'W' else 0
                pred_win_prob = pred['p1_win_prob'] / 100
                pred_win = 1 if pred_win_prob > 0.5 else 0
                
                # Track accuracy
                correct_pred = 1 if pred_win == actual_win else 0
                correct += correct_pred
                total += 1
                
                # Bucket by confidence
                fav_prob = max(pred_win_prob, 1 - pred_win_prob)
                for threshold in sorted(results_by_prob.keys(), reverse=True):
                    if fav_prob * 100 >= threshold:
                        results_by_prob[threshold][0] += correct_pred
                        results_by_prob[threshold][1] += 1
                        break
                
            except ValueError as e:
                errors.append(str(e))
                continue
    
    print(f"\n=== MODEL ACCURACY TEST {'(with fatigue)' if use_fatigue else '(no fatigue)'} ===")
    print(f"Total matches tested: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {100*correct/total:.1f}%")
    print(f"Errors (opponent not found): {len(errors)}")
    
    if use_fatigue and fatigue_impacts:
        print(f"\nFatigue applied: {len(fatigue_impacts)} matches")
        p1_adjs = [abs(f['p1_adj']) for f in fatigue_impacts]
        p2_adjs = [abs(f['p2_adj']) for f in fatigue_impacts]
        net_adjs = [abs(f['net']) for f in fatigue_impacts]
        print(f"  Avg serve adjustment: {np.mean(p1_adjs + p2_adjs):.4f}")
        print(f"  Max serve adjustment: {max(max(p1_adjs), max(p2_adjs)):.4f}")
        print(f"  Avg net impact: {np.mean(net_adjs):.2f}%")
    
    print(f"\n=== ACCURACY BY CONFIDENCE ===")
    for threshold in sorted(results_by_prob.keys()):
        c, t = results_by_prob[threshold]
        if t > 0:
            print(f"  {threshold}%+ confidence: {100*c/t:.1f}% ({c}/{t})")
    
    return correct / total if total > 0 else 0


def check_skill_distribution():
    """Check skill distributions are centered."""
    model = SkillPredictor()
    model.train()
    
    serve_hard = [p['serve']['hard'] for p in model.player_skills.values()]
    serve_clay = [p['serve']['clay'] for p in model.player_skills.values()]
    return_hard = [p['return']['hard'] for p in model.player_skills.values()]
    return_clay = [p['return']['clay'] for p in model.player_skills.values()]
    
    print("\n=== SKILL DISTRIBUTIONS ===")
    print(f"Serve (Hard):   mean={np.mean(serve_hard):.3f}, std={np.std(serve_hard):.3f}")
    print(f"Serve (Clay):   mean={np.mean(serve_clay):.3f}, std={np.std(serve_clay):.3f}")
    print(f"Return (Hard):  mean={np.mean(return_hard):.3f}, std={np.std(return_hard):.3f}")
    print(f"Return (Clay):  mean={np.mean(return_clay):.3f}, std={np.std(return_clay):.3f}")


def compare_fatigue_impact():
    """Compare accuracy with and without fatigue."""
    print("\n" + "="*60)
    print("COMPARING: WITH vs WITHOUT FATIGUE")
    print("="*60)
    
    print("\n--- WITHOUT FATIGUE ---")
    acc_no_fatigue = test_accuracy(elo_weight=0.5, use_fatigue=False)
    
    print("\n--- WITH FATIGUE ---")
    acc_with_fatigue = test_accuracy(elo_weight=0.5, use_fatigue=True)
    
    diff = (acc_with_fatigue - acc_no_fatigue) * 100
    print(f"\n{'='*60}")
    print(f"SUMMARY: Fatigue impact on accuracy: {diff:+.2f}%")
    if diff > 0:
        print("Fatigue model IMPROVED accuracy")
    elif diff < 0:
        print("Fatigue model DECREASED accuracy")
    else:
        print("No change in accuracy")


if __name__ == '__main__':
    compare_fatigue_impact()
