# WTA Tennis Match Prediction Model

A machine learning model for predicting WTA tennis match outcomes, game spreads, and total games.

## Features

- **Winner Prediction**: Probability of each player winning (71.1% accuracy, 0.775 AUC-ROC)
- **Game Spread**: Predicted margin of victory (MAE: 2.06 games)
- **Total Games**: Over/under prediction (75.0% accuracy at O/U 21.5)
- **Mathematically Consistent**: All predictions guaranteed to be internally consistent
- **ELO-Level Adjusted**: Accounts for performance vs strong opponents and opponent-specific level

## Quick Start

```python
from unified_model import UnifiedPredictor

predictor = UnifiedPredictor()
predictor.train()  # Loads cached model or trains fresh

predictor.print_prediction(
    'Iga Swiatek', 
    'Anna Kalinskaya', 
    'Hard',
    'Australian Open', 
    'R16'
)
```

## Installation

```bash
pip install numpy scikit-learn scipy joblib
```

## Files

### Core Model (Recommended)
- `unified_model.py` - Unified prediction model
- `UNIFIED_MODEL_DOCUMENTATION.txt` - Detailed documentation

### Data Files
- `all_players_matches.json` - Match database (9,633 matches)
- `players.json` - Player database (200 WTA players)
- `tennis_abstract_elo.json` - ELO ratings

### Data Collection
- `fetch_all_players.py` - Batch fetch player data
- `fetch_tennis_abstract.py` - Scrape ELO ratings
- `fetch_match_stats.py` - Scrape match statistics

### Legacy Models (Individual)
- `prediction_model.py` - Winner prediction only
- `game_spread_model.py` - Spread prediction only
- `total_games_model.py` - Total games only

### Analysis Tools
- `spread_betting.py` - Spread betting analyzer
- `total_games_betting.py` - O/U betting analyzer

## Model Performance

| Prediction | Metric | Value |
|------------|--------|-------|
| Winner | Accuracy | 71.1% |
| Winner | AUC-ROC | 0.775 |
| Spread | MAE | 2.06 games |
| Total Games | MAE | 3.55 games |
| Total Games | O/U 21.5 | 75.0% |

## Data Source

Match data and ELO ratings from [Tennis Abstract](http://tennisabstract.com/)

## Documentation

See `MODEL_DOCUMENTATION.txt` for overview and `UNIFIED_MODEL_DOCUMENTATION.txt` for detailed specifications including all 46 features used.
