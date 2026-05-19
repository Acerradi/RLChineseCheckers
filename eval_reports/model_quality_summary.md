# Model Quality Summary

Target: `heuristic`
Target kind: `heuristic`
Overall quality: **Good**
Total games: 500

## Headline

- Overall win/draw/loss: 49.2% / 0.0% / 50.8%
- Chance-adjusted overall win rate: 1.83x random expectation
- Baseline win rate: 60.7%
- Chance-adjusted baseline win rate: 2.33x random expectation
- Checkpoint-opponent win rate: 32.0%
- Chance-adjusted checkpoint win rate: 1.07x random expectation
- Average progress rank: 2.08
- Progress placement score: 0.643 (1.0 first place, 0.0 last place)
- Overall performance score: 0.644 (blends progress rank and final-score rank)
- Top-half by progress rate: 68.0%
- Move-cap/adjudication rate: 0.0%
- Average score: 891.3
- Average training progress: 195.5
- Average pins in goal: 7.04
- Average remaining goal distance: 13.3
- Illegal attempts: 0

## Best And Weakest Cases

- Best: `vs_random` with 2 players, win rate 100.0%, adjusted win 2.00x, performance 1.000
- Weakest: `vs_heuristic` with 6 players, win rate 5.0%, adjusted win 0.30x, performance 0.310

## Scenario Averages

| Scenario | Games | Win | Adj Win | Avg Rank | Place | Perf | Top Half | Loss | Cap | Score | Progress |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `vs_checkpoint:champion.pt` | 100 | 36.0% | 1.16x | 2.51 | 0.525 | 0.525 | 55.0% | 64.0% | 0.0% | 1039.6 | 247.5 |
| `vs_checkpoint:shared_model_final.pt` | 100 | 28.0% | 0.99x | 2.51 | 0.485 | 0.486 | 52.0% | 72.0% | 0.0% | 1027.0 | 243.5 |
| `vs_heuristic` | 100 | 36.0% | 1.17x | 2.58 | 0.512 | 0.512 | 56.0% | 64.0% | 0.0% | 1009.0 | 237.6 |
| `vs_mixed` | 100 | 46.0% | 1.82x | 1.79 | 0.695 | 0.696 | 77.0% | 54.0% | 0.0% | 755.2 | 151.6 |
| `vs_random` | 100 | 100.0% | 4.00x | 1.00 | 1.000 | 1.000 | 100.0% | 0.0% | 0.0% | 625.7 | 97.2 |

## Player Count Averages

| Players | Games | Win | Adj Win | Avg Rank | Place | Perf | Top Half | Loss | Cap | Score | Progress |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 100 | 62.0% | 1.24x | 1.38 | 0.620 | 0.620 | 62.0% | 38.0% | 0.0% | 1100.9 | 267.5 |
| 3 | 100 | 64.0% | 1.92x | 1.55 | 0.725 | 0.725 | 81.0% | 36.0% | 0.0% | 915.1 | 203.4 |
| 4 | 100 | 41.0% | 1.64x | 2.24 | 0.587 | 0.587 | 56.0% | 59.0% | 0.0% | 804.3 | 164.6 |
| 5 | 100 | 40.0% | 2.00x | 2.41 | 0.647 | 0.648 | 75.0% | 60.0% | 0.0% | 818.6 | 170.8 |
| 6 | 100 | 39.0% | 2.34x | 2.81 | 0.638 | 0.639 | 66.0% | 61.0% | 0.0% | 817.6 | 171.2 |
