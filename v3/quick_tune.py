"""Quick tuning comparison."""
import sys
import json
import numpy as np
from skill_model import SkillPredictor, load_data

# Force output flush
def print_flush(msg):
    print(msg)
    sys.stdout.flush()

def test_accuracy_quick(model, data, n_sims=200):
    """Quick accuracy test with fewer sims."""
    correct = 0
    total = 0
    
    for player_name, player_info in data['players'].items():
        for match in player_info.get('matches', [])[:5]:  # Only 5 matches per player
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
                                    match_round=match_round, n_sims=n_sims)
                
                actual_win = 1 if result == 'W' else 0
                pred_win = 1 if pred['p1_win_prob'] > 50 else 0
                
                if pred_win == actual_win:
                    correct += 1
                total += 1
            except:
                continue
    
    return correct / total if total > 0 else 0, total

if __name__ == '__main__':
    import fatigue_v2
    
    print_flush("="*60)
    print_flush("QUICK FATIGUE PARAMETER COMPARISON")
    print_flush("="*60)
    
    data = load_data()
    
    configs = [
        # (name, points_weight, 3set_bonus, travel_weight, tz_weight, decay, serve_factor)
        ("Current baseline", 0.0005, 0.1, 0.00002, 0.02, 0.75, 0.3),
        ("Higher decay (0.85)", 0.0005, 0.1, 0.00002, 0.02, 0.85, 0.3),
        ("Lower decay (0.6)", 0.0005, 0.1, 0.00002, 0.02, 0.6, 0.3),
        ("Higher serve factor (0.5)", 0.0005, 0.1, 0.00002, 0.02, 0.75, 0.5),
        ("More timezone impact (0.05)", 0.0005, 0.1, 0.00002, 0.05, 0.75, 0.3),
        ("Higher points weight (0.001)", 0.001, 0.1, 0.00002, 0.02, 0.75, 0.3),
    ]
    
    results = []
    for name, pw, tsb, tw, tzw, decay, sf in configs:
        print_flush(f"\nTesting: {name}...")
        
        fatigue_v2.POINTS_WEIGHT = pw
        fatigue_v2.THREE_SET_BONUS = tsb
        fatigue_v2.TRAVEL_WEIGHT = tw
        fatigue_v2.TIMEZONE_WEIGHT = tzw
        fatigue_v2.FATIGUE_DECAY = decay
        fatigue_v2.FATIGUE_TO_SERVE_FACTOR = sf
        
        model = SkillPredictor(elo_weight=0.5)
        model.train()
        
        acc, n = test_accuracy_quick(model, data)
        results.append((name, acc, n))
        print_flush(f"  Accuracy: {acc*100:.2f}% ({n} matches)")
    
    print_flush("\n" + "="*60)
    print_flush("SUMMARY")
    print_flush("="*60)
    results.sort(key=lambda x: x[1], reverse=True)
    for name, acc, n in results:
        print_flush(f"  {name:<35} {acc*100:.2f}%")
