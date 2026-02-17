# Historical Data Directory

This directory contains historical WTA match data for backtesting the prediction model.

## Files

### `wta_2025_results.xlsx`
Complete 2025 WTA season results including:
- 2,505 total matches
- Match results (winner, loser, scores)
- Betting odds from multiple bookmakers (Pinnacle, Bet365, etc.)
- Player rankings and points

### `data_dictionary.txt`
Documentation explaining all columns in the results file:
- Match info: Date, Tournament, Surface, Round, etc.
- Score data: W1, L1, W2, L2, etc.
- Betting odds: PSW/PSL (Pinnacle), B365W/B365L (Bet365), etc.

### `analysis_2025_results.csv`
Generated analysis output containing:
- Model predictions vs actual results
- CLV (Closing Line Value) calculations
- Accuracy tracking per match

## Usage

Run the analysis script from the v3 directory:
```bash
python analyze_2025.py
```

This will:
1. Load the historical data
2. Generate model predictions for each match
3. Compare to Pinnacle closing odds
4. Calculate CLV and accuracy metrics
5. Save results to `analysis_2025_results.csv`

## Key Metrics from 2025 Analysis

| Metric | Value |
|--------|-------|
| Matches Analyzed | 1,859 |
| Overall Accuracy | 66.6% |
| 70%+ Confidence Accuracy | 82.9% |
| 80%+ Confidence Accuracy | 89.1% |
| Average CLV | -0.66% |
| Flat Bet ROI (all picks) | +7.42% |
| Flat Bet ROI (65%+ conf) | +11.10% |
