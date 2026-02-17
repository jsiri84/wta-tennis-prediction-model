"""Check name formats between 2025 data and model."""
import pandas as pd
import json

# Load 2025 data
df = pd.read_excel(r'c:\Users\jason\Downloads\2025.xlsx')

print("Sample player names from 2025 data:")
winners = df['Winner'].unique()[:20]
for w in winners:
    print(f"  '{w}'")

print("\n\nSample player names from model:")
data = json.load(open('player_data.json'))
players = list(data['players'].keys())[:20]
for p in players:
    print(f"  '{p}'")
