# Model Quality Summary

Target: `checkpoints/gnn_h128_l4/self_play/shared_model_final.pt`
Target kind: `checkpoint`
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

## Opponent Comparisons

Each row compares the target directly against every opponent of that type/model it faced. `Score Better` is the main head-to-head quality metric: 100% means the target always had a strictly higher final score than that opponent.

| Opponent | Pairings | Score Better | Score Tie | Score Worse | Avg Score Margin | Progress Better | Avg Progress Margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| `checkpoint:champion.pt` | 100 | 54.0% | 0.0% | 46.0% | 21.9 | 53.0% | 6.1 |
| `checkpoint:shared_model_final.pt` | 100 | 54.0% | 0.0% | 46.0% | -3.9 | 52.0% | -4.7 |
| `heuristic` | 880 | 44.9% | 0.3% | 54.8% | -20.7 | 43.2% | -10.3 |
| `random` | 420 | 100.0% | 0.0% | 0.0% | 444.1 | 100.0% | 248.4 |

## Player Count Averages

| Players | Games | Win | Adj Win | Avg Rank | Place | Perf | Top Half | Loss | Cap | Score | Progress |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 100 | 83.0% | 1.66x | 1.17 | 0.830 | 0.830 | 83.0% | 17.0% | 0.0% | 1108.0 | 269.8 |
| 3 | 100 | 48.0% | 1.44x | 1.82 | 0.590 | 0.598 | 70.0% | 52.0% | 0.0% | 825.8 | 172.0 |
| 4 | 100 | 29.0% | 1.16x | 2.32 | 0.560 | 0.560 | 58.0% | 71.0% | 0.0% | 805.9 | 163.3 |
| 5 | 100 | 40.0% | 2.00x | 2.37 | 0.657 | 0.659 | 74.0% | 60.0% | 0.0% | 838.4 | 177.1 |
| 6 | 100 | 31.0% | 1.86x | 3.22 | 0.556 | 0.560 | 55.0% | 69.0% | 0.0% | 735.7 | 135.5 |
