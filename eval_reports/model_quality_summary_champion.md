# Model Quality Summary

Target: `checkpoints/gnn_h128_l4/self_play/champion.pt`
Overall quality: **Good**
Total games: 500

## Headline

- Overall win/draw/loss: 46.2% / 0.0% / 53.8%
- Chance-adjusted overall win rate: 1.62x random expectation
- Baseline win rate: 54.3%
- Chance-adjusted baseline win rate: 1.97x random expectation
- Checkpoint-opponent win rate: 34.0%
- Chance-adjusted checkpoint win rate: 1.10x random expectation
- Average progress rank: 2.18
- Progress placement score: 0.639 (1.0 first place, 0.0 last place)
- Overall performance score: 0.641 (blends progress rank and final-score rank)
- Top-half by progress rate: 68.0%
- Move-cap/adjudication rate: 0.0%
- Average score: 862.8
- Average training progress: 183.5
- Average pins in goal: 6.77
- Average remaining goal distance: 15.0
- Illegal attempts: 0

## Best And Weakest Cases

- Best: `vs_random` with 2 players, win rate 100.0%, adjusted win 2.00x, performance 1.000
- Weakest: `vs_checkpoint:champion.pt` with 4 players, win rate 0.0%, adjusted win 0.00x, performance 0.400

## Scenario Averages

| Scenario | Games | Win | Adj Win | Avg Rank | Place | Perf | Top Half | Loss | Cap | Score | Progress |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `vs_checkpoint:champion.pt` | 100 | 32.0% | 1.02x | 2.53 | 0.522 | 0.522 | 54.0% | 68.0% | 0.0% | 1007.6 | 234.6 |
| `vs_checkpoint:shared_model_final.pt` | 100 | 36.0% | 1.19x | 2.53 | 0.531 | 0.532 | 56.0% | 64.0% | 0.0% | 993.4 | 228.6 |
| `vs_heuristic` | 100 | 25.0% | 0.68x | 2.88 | 0.440 | 0.447 | 47.0% | 75.0% | 0.0% | 996.3 | 230.4 |
| `vs_mixed` | 100 | 38.0% | 1.23x | 1.96 | 0.701 | 0.705 | 83.0% | 62.0% | 0.0% | 730.5 | 140.8 |
| `vs_random` | 100 | 100.0% | 4.00x | 1.00 | 1.000 | 1.000 | 100.0% | 0.0% | 0.0% | 586.0 | 83.2 |

## Player Count Averages

| Players | Games | Win | Adj Win | Avg Rank | Place | Perf | Top Half | Loss | Cap | Score | Progress |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 100 | 83.0% | 1.66x | 1.17 | 0.830 | 0.830 | 83.0% | 17.0% | 0.0% | 1108.0 | 269.8 |
| 3 | 100 | 48.0% | 1.44x | 1.82 | 0.590 | 0.598 | 70.0% | 52.0% | 0.0% | 825.8 | 172.0 |
| 4 | 100 | 29.0% | 1.16x | 2.32 | 0.560 | 0.560 | 58.0% | 71.0% | 0.0% | 805.9 | 163.3 |
| 5 | 100 | 40.0% | 2.00x | 2.37 | 0.657 | 0.659 | 74.0% | 60.0% | 0.0% | 838.4 | 177.1 |
| 6 | 100 | 31.0% | 1.86x | 3.22 | 0.556 | 0.560 | 55.0% | 69.0% | 0.0% | 735.7 | 135.5 |
