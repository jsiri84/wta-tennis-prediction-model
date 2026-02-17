"""Tune fatigue v2 parameters."""

import json
import numpy as np
from skill_model import SkillPredictor, load_data
import fatigue_v2

def test_with_params(points_weight, three_set_bonus, travel_weight, 
                     timezone_weight, decay, serve_factor):
    """Test model with specific fatigue parameters."""
    # Set params
    fatigue_v2.POINTS_WEIGHT = points_weight
    fatigue_v2.THREE_SET_BONUS = three_set_bonus
    fatigue_v2.TRAVEL_WEIGHT = travel_weight
    fatigue_v2.TIMEZONE_WEIGHT = timezone_weight
    fatigue_v2.FATIGUE_DECAY = decay
    fatigue_v2.FATIGUE_TO_SERVE_FACTOR = serve_factor
    
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
            match_date = match.get('date', '')
            tournament = match.get('tournament', '')
            
            if not opponent or result not in ('W', 'L'):
                continue
            
            try:
                match_round = match.get('round', '')
                pred = model.predict(player_name, opponent, surface, 
                                    tournament=tournament, match_date=match_date,
                                    match_round=match_round, n_sims=500)
                
                actual_win = 1 if result == 'W' else 0
                pred_win = 1 if pred['p1_win_prob'] > 50 else 0
                
                if pred_win == actual_win:
                    correct += 1
                total += 1
            except:
                continue
    
    return correct / total if total > 0 else 0


if __name__ == '__main__':
    print("="*60)
    print("TUNING FATIGUE V2 PARAMETERS")
    print("="*60)
    
    # Baseline
    print("\nBaseline (current settings)...")
    baseline = test_with_params(
        points_weight=0.0005,
        three_set_bonus=0.1,
        travel_weight=0.00002,
        timezone_weight=0.02,
        decay=0.75,
        serve_factor=0.3
    )
    print(f"  Accuracy: {baseline*100:.2f}%")
    
    # Test different decay rates
    print("\nTesting decay rates...")
    for decay in [0.6, 0.7, 0.8, 0.85]:
        acc = test_with_params(0.0005, 0.1, 0.00002, 0.02, decay, 0.3)
        print(f"  Decay={decay}: {acc*100:.2f}%")
    
    # Test different serve factors
    print("\nTesting serve factors...")
    for sf in [0.2, 0.4, 0.5]:
        acc = test_with_params(0.0005, 0.1, 0.00002, 0.02, 0.75, sf)
        print(f"  ServeFactor={sf}: {acc*100:.2f}%")
    
    # Test different points weights
    print("\nTesting points weights...")
    for pw in [0.0003, 0.0008, 0.001]:
        acc = test_with_params(pw, 0.1, 0.00002, 0.02, 0.75, 0.3)
        print(f"  PointsWeight={pw}: {acc*100:.2f}%")
    
    # Test more timezone emphasis
    print("\nTesting timezone weights...")
    for tz in [0.01, 0.03, 0.05]:
        acc = test_with_params(0.0005, 0.1, 0.00002, tz, 0.75, 0.3)
        print(f"  TimezoneWeight={tz}: {acc*100:.2f}%")
    
    print("\n" + "="*60)
    print("Done tuning!")
