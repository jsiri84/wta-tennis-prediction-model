"""Check surface-specific serve/return statistics."""
import json
import numpy as np
from collections import defaultdict

data = json.load(open('player_data.json'))

# Collect SPW and RPW by surface
spw_by_surface = defaultdict(list)
rpw_by_surface = defaultdict(list)

for player_name, player_info in data['players'].items():
    for match in player_info.get('matches', []):
        surface = match.get('surface', 'Hard').lower()
        if surface not in ['hard', 'clay', 'grass']:
            surface = 'hard'
        
        serve = match.get('serve_raw', {})
        ret = match.get('return_raw', {})
        
        # SPW (service points won) = (fwon + swon) / pts
        sp = serve.get('pts')
        fwon = serve.get('fwon', 0) or 0
        swon = serve.get('swon', 0) or 0
        sw = fwon + swon
        if sp and sp > 0:
            spw_by_surface[surface].append(sw / sp)
        
        # RPW (return points won) = points we won on opponent's serve
        # opts = opponent's total serve points
        # ofwon = opponent's first serve points won (against us)
        # oswon = opponent's second serve points won (against us)
        # So RPW = 1 - (ofwon + oswon) / opts
        opts = ret.get('opts')
        ofwon = ret.get('ofwon', 0) or 0
        oswon = ret.get('oswon', 0) or 0
        opp_won = ofwon + oswon
        if opts and opts > 0:
            rpw_by_surface[surface].append(1 - (opp_won / opts))

print("="*60)
print("SURFACE-SPECIFIC STATISTICS")
print("="*60)

print("\nService Points Won (SPW) by Surface:")
for surface in ['hard', 'clay', 'grass']:
    vals = spw_by_surface[surface]
    if vals:
        print(f"  {surface.capitalize():8} avg={np.mean(vals)*100:.1f}%  std={np.std(vals)*100:.1f}%  n={len(vals)}")

print("\nReturn Points Won (RPW) by Surface:")
for surface in ['hard', 'clay', 'grass']:
    vals = rpw_by_surface[surface]
    if vals:
        print(f"  {surface.capitalize():8} avg={np.mean(vals)*100:.1f}%  std={np.std(vals)*100:.1f}%  n={len(vals)}")

print("\n\nCurrent model constants:")
print("  SPW_AVG = 0.62 (62%)")
print("  RPW_AVG = 0.45 (45%)")

print("\n\nRecommended surface-specific constants:")
for surface in ['hard', 'clay', 'grass']:
    spw_vals = spw_by_surface[surface]
    rpw_vals = rpw_by_surface[surface]
    if spw_vals and rpw_vals:
        spw_avg = np.mean(spw_vals)
        rpw_avg = np.mean(rpw_vals)
        print(f"  {surface.upper()}: SPW_AVG={spw_avg:.3f}, RPW_AVG={rpw_avg:.3f}")
