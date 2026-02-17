"""Test with travel-only fatigue (no within-tournament fatigue)."""

import json
import numpy as np
from skill_model import SkillPredictor, load_data
from fatigue import calculate_travel_distance, get_travel_fatigue_factor, parse_date, estimate_match_date
from datetime import timedelta

def test_travel_only_fatigue():
    """Test accuracy with only travel fatigue between tournaments."""
    
    model = SkillPredictor(elo_weight=0.5)
    model.train()
    
    data = load_data()
    
    correct = 0
    total = 0
    travel_applied = 0
    results_by_prob = {50: [0, 0], 55: [0, 0], 60: [0, 0], 65: [0, 0], 70: [0, 0], 75: [0, 0], 80: [0, 0]}
    
    for player_name, player_info in data['players'].items():
        matches = player_info.get('matches', [])
        
        for i, match in enumerate(matches[:10]):
            opponent = match.get('opponent', '')
            surface = match.get('surface', 'Hard')
            result = match.get('result', '')
            tournament = match.get('tournament', '')
            match_date = match.get('date', '')
            match_round = match.get('round', '')
            
            if not opponent or result not in ('W', 'L'):
                continue
            
            try:
                # Get base prediction (no fatigue)
                pred = model.predict(player_name, opponent, surface, n_sims=1000)
                p1_win_prob = pred['p1_win_prob'] / 100
                
                # Check for travel fatigue from previous tournament
                # Find most recent match from DIFFERENT tournament
                travel_penalty_p1 = 0.0
                for prev_match in matches[i+1:i+10]:  # Look at previous matches
                    prev_tournament = prev_match.get('tournament', '')
                    prev_date = prev_match.get('date', '')
                    
                    if prev_tournament and prev_tournament != tournament:
                        # Different tournament - check if back-to-back
                        current_dt = estimate_match_date(match_date, match_round)
                        prev_dt = estimate_match_date(prev_date, prev_match.get('round', ''))
                        
                        if current_dt and prev_dt:
                            days_between = (current_dt - prev_dt).days
                            
                            if 0 < days_between <= 10:  # Back-to-back
                                travel_km = calculate_travel_distance(prev_tournament, tournament)
                                if travel_km and travel_km > 3000:  # Significant travel
                                    travel_penalty_p1 = get_travel_fatigue_factor(travel_km) * 0.03
                                    travel_applied += 1
                        break  # Found previous tournament
                
                # Apply travel penalty (assuming opponent might have similar travel)
                # In reality we'd check opponent's history too, but simplified for now
                adjusted_prob = p1_win_prob - travel_penalty_p1 / 2  # Half penalty as rough estimate
                adjusted_prob = max(0.05, min(0.95, adjusted_prob))
                
                actual_win = 1 if result == 'W' else 0
                pred_win = 1 if adjusted_prob > 0.5 else 0
                
                correct_pred = 1 if pred_win == actual_win else 0
                correct += correct_pred
                total += 1
                
                # Bucket by confidence
                fav_prob = max(adjusted_prob, 1 - adjusted_prob)
                for threshold in sorted(results_by_prob.keys(), reverse=True):
                    if fav_prob * 100 >= threshold:
                        results_by_prob[threshold][0] += correct_pred
                        results_by_prob[threshold][1] += 1
                        break
                        
            except ValueError:
                continue
    
    print(f"\n=== TRAVEL-ONLY FATIGUE TEST ===")
    print(f"Total matches: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {100*correct/total:.1f}%")
    print(f"Travel penalty applied: {travel_applied} times")
    
    print(f"\n=== BY CONFIDENCE ===")
    for threshold in sorted(results_by_prob.keys()):
        c, t = results_by_prob[threshold]
        if t > 0:
            print(f"  {threshold}%+: {100*c/t:.1f}% ({c}/{t})")


if __name__ == '__main__':
    # First show baseline
    print("="*60)
    print("BASELINE (no fatigue)")
    print("="*60)
    
    model = SkillPredictor(elo_weight=0.5)
    model.train()
    data = load_data()
    
    correct = 0
    total = 0
    
    for player_name, player_info in data['players'].items():
        for match in player_info.get('matches', [])[:10]:
            opponent = match.get('opponent', '')
            surface = match.get('surface', 'Hard')
            result = match.get('result', '')
            
            if not opponent or result not in ('W', 'L'):
                continue
            
            try:
                pred = model.predict(player_name, opponent, surface, n_sims=1000)
                actual_win = 1 if result == 'W' else 0
                pred_win = 1 if pred['p1_win_prob'] > 50 else 0
                if pred_win == actual_win:
                    correct += 1
                total += 1
            except:
                pass
    
    print(f"Baseline accuracy: {100*correct/total:.1f}% ({correct}/{total})")
    
    # Then travel-only
    print("\n" + "="*60)
    print("TRAVEL-ONLY FATIGUE")
    print("="*60)
    test_travel_only_fatigue()
