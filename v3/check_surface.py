from skill_model import SkillPredictor

model = SkillPredictor()
model.train()

# Check surface-specific skills for a few players
for name in ['Iga Swiatek', 'Aryna Sabalenka', 'Elena Rybakina']:
    p = model.player_skills.get(name)
    if p:
        print(f'\n{name}:')
        print(f"  Serve:  Hard {p['serve']['hard']:+.3f} | Clay {p['serve']['clay']:+.3f} | Grass {p['serve']['grass']:+.3f}")
        print(f"  Return: Hard {p['return']['hard']:+.3f} | Clay {p['return']['clay']:+.3f} | Grass {p['return']['grass']:+.3f}")

# Test predictions on different surfaces
print("\n\nSABALENKA vs SWIATEK by surface:")
for surface in ['Hard', 'Clay', 'Grass']:
    r = model.predict('Aryna Sabalenka', 'Iga Swiatek', surface)
    print(f"  {surface}: Sabalenka {r['p1_win_prob']:.1f}% (skill {r['p1_skill_prob']:.1f}%, elo {r['p1_elo_prob']:.1f}%)")
