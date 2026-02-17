# WTA Tennis Prediction Model v3 - Summary

## Overview
A WTA tennis match prediction model combining:
- **ServeSkill / ReturnSkill** - Surface-specific player abilities
- **ELO Ratings** - Overall and surface-specific rankings
- **Fatigue Model** - Travel, schedule, and workload effects
- **Court Speed** - Tournament-specific court pace adjustments
- **Head-to-Head** - Historical matchup adjustments
- **Clutch/Overperformance** - Win rate vs expected from point stats
- **Monte Carlo Simulation** - Match outcome and total games prediction

---

## Current Performance (vs 2025 WTA Data)

*Last updated: February 17, 2026*

| Metric | Value |
|--------|-------|
| Matches Analyzed | 1,859 |
| Overall Accuracy | 66.6% |
| 55%+ Confidence | 71.3% |
| 60%+ Confidence | 76.1% |
| 65%+ Confidence | 80.9% |
| 70%+ Confidence | 82.8% |
| 75%+ Confidence | 87.4% |
| 80%+ Confidence | 93.0% |
| ROI (All Picks) | +7.3% |
| ROI (High Confidence 65%+) | +12.6% |
| ROI (Positive CLV) | +21.3% |

**Surface Accuracy:**
- Hard: 65.0%
- Clay: 69.9%
- Grass: 69.2%

**CLV Analysis:**
- Average CLV: -1.27%
- Positive CLV Rate: 43.6%
- Upset Detection: 33.0%

---

## 1. Core Skill Model

### Parameters
```
K_DF = 0.5              # Double fault penalty strength
C_OPP = 0.5             # Opponent adjustment strength
EWMA_HALF_LIFE = 8      # Matches for skill decay
SPW_AVG = 0.58          # Tour avg service points won
RPW_AVG = 0.45          # Tour avg return points won
```

### ServeSkill Calculation
1. Calculate raw SPW (Service Points Won):
   ```
   SPW = FSIn × 1stServeWon% + (1-FSIn) × 2ndServeWon%
   ```

2. Apply double fault penalty:
   ```
   SPW* = SPW - K_DF × (DFrate - 0.05)
   ```

3. Opponent-adjust using ReturnSkill:
   ```
   SPW_adj = SPW* + C_OPP × (ReturnSkill_opp - avg)
   ```

4. Convert to logit scale and apply EWMA smoothing
5. Normalize to center at 0 for each surface

### ReturnSkill Calculation
1. Calculate RPW (Return Points Won):
   ```
   RPW = 1 - opponent's SPW
   ```

2. Convert to logit scale and apply EWMA smoothing
3. Normalize to center at 0 for each surface

### Surface-Specific Skills
Separate ServeSkill and ReturnSkill calculated for:
- **Hard** courts
- **Clay** courts
- **Grass** courts

---

## 2. ELO Ratings

### Parameters
```
elo_weight = 0.5        # Blend weight (50% skill, 50% ELO)
```

### ELO Types
- Overall ELO
- Surface-specific ELO (Hard, Clay, Grass)

### Win Probability from ELO
```
P(win) = 1 / (1 + 10^((ELO2 - ELO1) / 400))
```

---

## 3. Fatigue Model (v2)

### Parameters
```
FATIGUE_DECAY = 0.6           # 60% fatigue remains per day
POINTS_WEIGHT = 0.0005        # Per point played
THREE_SET_BONUS = 0.1         # Extra load for 3-setters
TRAVEL_WEIGHT = 0.00002       # Per km traveled
TIMEZONE_WEIGHT = 0.05        # Per timezone crossed
OPTIMAL_REST_DAYS = 3         # Ideal rest between matches
RUST_THRESHOLD = 30           # Days before rust penalty
RUST_PENALTY_PER_DAY = 0.01   # Penalty per day over threshold
MAX_FATIGUE = 1.0             # Maximum fatigue cap
FATIGUE_TO_SERVE_FACTOR = 0.3 # ServeSkill reduction factor
```

### Fatigue Components
1. **Points Load**: Accumulates based on points played per match
2. **3-Set Bonus**: Additional load for matches going to 3 sets
3. **Travel Distance**: Haversine distance between tournament venues
4. **Timezone Shift**: Jet lag penalty for crossing time zones
5. **Decay**: Fatigue decays daily (60% retained)
6. **Rust Penalty**: Penalty for layoffs > 30 days

### Application
- Fatigue reduces **ServeSkill** (not win probability directly)
- Typical adjustment: 0.02-0.09 ServeSkill reduction
- Applied before hold probability calculation

### Tournament Locations
40+ tournaments mapped with lat/lon/timezone for accurate:
- Travel distance calculation
- Timezone shift penalties

---

## 4. Court Speed

### Parameters
Court speed index based on ATP "1st Serve Points Won %" (5-year avg 2021-2025)

### Categories
| Category | 1st Serve Won % | Characteristics |
|----------|-----------------|-----------------|
| Fast | >73% | Serves dominate, short rallies |
| Medium | 70-73% | Balanced play |
| Slow | <70% | Returns/rallies dominate |

### Speed Adjustment
```
speed_adj = court_adjustment × player_speed_preference × 0.5
```

- **Court adjustment**: +0.05 (fast) to -0.05 (slow)
- **Player speed preference**: Calculated from win rates on fast vs slow courts
- Applied to ServeSkill before prediction

### Example Court Speeds
**Fast Courts (>73%):**
- Stuttgart Grass: 77.9%
- Brussels Indoor: 76.7%
- Brisbane: 75.4%
- Dubai: 73.6%
- Wimbledon: 73.3%

**Medium Courts (70-73%):**
- US Open: 71.6%
- Australian Open: 71.6%
- Indian Wells: 71.4%
- Miami: 71.0%

**Slow Courts (<70%):**
- Roland Garros: 67.9%
- Madrid Clay: 69.0%
- Rome Clay: 68.5%
- Monte Carlo: 68.5%

---

## 5. Match Simulation

### Hold Probability
```
P(hold) = sigmoid(0.5 + ServeSkill - ReturnSkill_opponent)
```
- Baseline 0.5 gives ~62% hold rate for average players

### Monte Carlo Simulation
- **N simulations**: 5,000 (default)
- **Best of**: 3 sets (WTA format)
- **Tiebreak**: At 6-6 in each set

### Outputs
- Win probability for each player
- Average total games
- Standard deviation of total games

---

## 6. Head-to-Head Adjustment

### Parameters
```
H2H_MIN_MEETINGS = 2    # Min prior meetings to apply H2H
H2H_WEIGHT = 0.10       # 10% blend weight
```

### Calculation
1. Build H2H database from all player match histories (deduplicated)
2. For each matchup, get record of meetings BEFORE match date
3. Calculate H2H probability with Laplace smoothing:
   ```
   h2h_prob = (p1_wins + 0.5) / (total_meetings + 1)
   ```
4. Blend with current probability:
   ```
   final_prob = (1 - H2H_WEIGHT) × current_prob + H2H_WEIGHT × h2h_prob
   ```

### When Applied
- Only when H2H meetings >= 2
- Adjusts final win probability before output
- Typical adjustment: ±1-5% depending on H2H dominance

---

## 7. Final Prediction Blend

```
P(win) = (1 - elo_weight) × skill_prob + elo_weight × elo_prob
P(win) += clutch_adjustment  # If enabled
P(win) = blend with H2H if meetings >= 2
P(win) = max(0.05, min(0.95, P(win)))  # Clamped to 5-95%
```

---

## 8. Data Sources

### Player Data (`player_data.json`)
- **Source**: Tennis Abstract (tennisabstract.com)
- **Players**: Top 200 WTA
- **Data**: Match history, serve/return stats, ELO ratings, handedness
- **Matches**: 10,385 total (96.4% with serve stats)
- **Handedness**: 166 right, 20 left, 14 unknown (R/L/U)

### Handedness Analysis (hand_coefficients.json)
- Player-specific coefficients for vs-lefty performance
- **Disabled by default** - testing showed it hurts overall accuracy
- Notable struggles vs lefties: Caroline Garcia (16.7%), Sorana Cirstea (33.3%)
- Notable excel vs lefties: Qinwen Zheng (100%), Nina Stojanovic (100%)

### Clutch/Overperformance (clutch_coefficients.json)
- **Enabled by default** - improves betting ROI at high confidence
- Adjusts for players who win more/less than their point stats suggest
- Overperformers (ELO may be inflated): Sabalenka (+12.3%), Rybakina, Svitolina
- Underperformers (ELO may be deflated): Navarro (-7.9%), Kasatkina, Haddad Maia
- Effect: -0.3% overall accuracy, +2.6% high confidence ROI

### Head-to-Head (H2H)
- **Enabled by default** - computed dynamically from match history
- Parameters: `H2H_MIN_MEETINGS = 2`, `H2H_WEIGHT = 0.10` (10% blend)
- Uses Laplace smoothing: `(wins + 0.5) / (total + 1)` to avoid extreme probs
- Effect: -0.3% overall accuracy, +2.4% high confidence accuracy, +3.9% high confidence ROI
- Most valuable for: repeat matchups between top players

### Historical Testing (`data/wta_2025_results.xlsx`)
- **Source**: tennis-data.co.uk
- **Period**: 2025 WTA season
- **Fields**: Results, Pinnacle odds, Bet365 odds

---

## 9. Key Insights from Analysis

1. **Model is well-calibrated** overall (no probability tuning needed)
2. **High confidence picks are reliable** (93% at 80%+ confidence)
3. **Negative CLV picks perform well** (73% accuracy when model < market)
4. **Clay/Grass accuracy higher** than hard court (~69% vs 65%)
5. **Positive CLV strategy** yields best ROI (+21.3%)
6. **Scalable accuracy** - confidence level directly correlates with accuracy

### Features Tested But Not Implemented
- **Rally Length Archetypes** - Data available (97% coverage) but no predictive value
- **Match Charting Data** - Net points, shot types available but redundant with existing stats
- **Serve-Return Interaction** - Nonlinear model unnecessary; linear already captures extremes well
- **Break Point Clutch** - Insufficient data coverage; no signal in testing
- **Short-term Momentum** - Mixed signal, potential regression-to-mean effect

---

## 10. File Structure

```
v3/
├── skill_model.py          # Main prediction model
├── fatigue_v2.py           # Advanced fatigue calculations
├── court_speed.py          # Court speed classifications
├── clutch_model.py         # Clutch/overperformance calculations
├── fetch_players.py        # Data fetching from Tennis Abstract
├── predict_matches.py      # Match prediction script (standard output)
├── analyze_2025.py         # 2025 season analysis vs Pinnacle
├── player_data.json        # Player database (200 players, 10K+ matches)
├── player_skills_cache.pkl # Cached skill calculations
├── clutch_coefficients.json    # Player clutch adjustments
├── hand_coefficients.json      # Handedness adjustments (disabled)
└── data/
    ├── wta_2025_results.xlsx      # Historical results + Pinnacle odds
    └── analysis_2025_results.csv  # Detailed backtest output
```

---

## 11. Usage

```python
from skill_model import SkillPredictor

# Initialize and train
model = SkillPredictor(elo_weight=0.5)
model.train()

# Make prediction
result = model.predict(
    player1="Aryna Sabalenka",
    player2="Iga Swiatek",
    surface="Hard",
    tournament="Australian Open",
    match_date="20260120"
)

print(f"{result['player1']}: {result['p1_win_prob']}%")
print(f"{result['player2']}: {result['p2_win_prob']}%")
print(f"Total Games: {result['total_games']} ± {result['total_std']}")
```

---

## 12. Standard Output Format

When requesting predictions, output in this format:

| P1 | P2 | Pick | Conf | P1 ML | P1 Book | P1 CLV | P2 ML | P2 Book | P2 CLV |
|----|----|----|------|-------|---------|--------|-------|---------|--------|
| Alexandrova | Linette | **Alexandrova** | 61.2% | -157 | -248 | -10.1% | +157 | +199 | **+5.7%** |

**Columns:**
- **P1/P2**: Player names
- **Pick**: Model's predicted winner (bold)
- **Conf**: Model confidence % (higher of P1/P2 probability)
- **P1 ML / P2 ML**: Model's American odds for each player
- **P1 Book / P2 Book**: Sportsbook American odds
- **P1 CLV / P2 CLV**: Closing Line Value (Model% - Book%). **Bold = positive value bet**

**CLV Interpretation:**
- Positive CLV = Model sees more value than book (potential bet)
- Negative CLV = Book sees more value than model (avoid)
