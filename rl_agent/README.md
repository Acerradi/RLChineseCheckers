# RL Agent for Chinese Checkers

Self-play PPO agent that trains to play Chinese Checkers. A single trained model works for any player count (2, 3, 4, or 6) — you can train on 1v1 and deploy the same checkpoint in a 6-player game, or mix player counts across training runs.

---

## Before you start

### Working directory

**All commands must be run from the project root (`RLChineseCheckers/`), not from inside `rl_agent/`.**

```bash
cd RLChineseCheckers   # make sure you are here
python -m rl_agent.train
```

Running from inside `rl_agent/` will fail with an import error because Python needs to see `rl_agent` as a package inside the current directory.

### Requirements

```bash
pip install torch numpy
```

---

## Training

### Start a new 1v1 run

```bash
python -m rl_agent.train
```

Training output is printed to the console **and** automatically written to `rl_agent/training.log` at the same time. To use a different log file:

```bash
python -m rl_agent.train --log-file rl_agent/my_run.log
```

To disable the log file and only print to the console:

```bash
python -m rl_agent.train --log-file ""
```

Checkpoints are saved to `rl_agent/checkpoints/` every 500 episodes and as `latest.pt` after every save.

### Reading the training log

Each log line looks like this:

```
ep=    1000  win=0.312  goal_pieces=4.70/10  ep_len=287  trunc=0.021  steps=142000  loss=0.0031  p=-0.0012  v=0.0043  ent=2.8801
```

| Field | Meaning |
|---|---|
| `win` | Rolling win rate over the last 200 episodes |
| `goal_pieces` | Avg agent pieces in the goal zone at episode end (0–10) — the main progress signal before wins appear |
| `ep_len` | Average episode length in steps |
| `trunc` | Fraction of episodes that hit the step limit (1.0 early on means no wins yet) |
| `steps` | Total environment steps taken so far |
| `loss` | Combined PPO loss |
| `p` | Policy (actor) loss |
| `v` | Value (critic) loss |
| `ent` | Policy entropy — higher means more exploration |

### Pause and resume

Press **Ctrl-C** at any time — the trainer catches the signal, writes a final checkpoint, and exits cleanly.

**Resuming is automatic.** If `rl_agent/checkpoints/latest.pt` exists, the next run picks up from where training stopped:

```bash
python -m rl_agent.train   # resumes automatically if latest.pt is present
```

To resume from a specific checkpoint instead:

```bash
python -m rl_agent.train --resume rl_agent/checkpoints/ep_00005000.pt
```

To ignore any existing checkpoint and start from scratch:

```bash
python -m rl_agent.train --fresh
```

The log file is appended to (not overwritten) on resume, so the full training history stays in one file.

### Scale to more players

```bash
# 4-player game
python -m rl_agent.train --n-players 4

# 6-player game
python -m rl_agent.train --n-players 6
```

`--n-players` only controls which game variant is simulated during training.
The network itself always takes a fixed 726-element input (6 × 121 board slots),
so **a checkpoint from any run can be loaded into any other player count**.
Absent player slots are zeroed out automatically.

### Common options

| Flag | Default | Description |
|---|---|---|
| `--n-players` | `2` | Number of players (2 / 3 / 4 / 6) |
| `--episodes` | `100000` | Total training episodes |
| `--resume` | — | Resume from a specific checkpoint (overrides auto-resume) |
| `--fresh` | off | Start from scratch even if `latest.pt` exists |
| `--checkpoint-dir` | `rl_agent/checkpoints` | Where to save checkpoints |
| `--log-file` | `rl_agent/training.log` | Log file path (set to `""` to disable) |
| `--hidden` | `256` | Hidden layer width |
| `--n-layers` | `4` | Number of residual blocks |
| `--lr` | `3e-4` | Adam learning rate |
| `--gamma` | `0.997` | Discount factor |
| `--update-every` | `512` | Agent steps between PPO gradient updates |
| `--max-episode-steps` | `1000` | Step limit per episode |
| `--opponent-sync-every` | `500` | Episodes between syncing the frozen opponent |
| `--save-every` | `500` | Episodes between named checkpoint saves |
| `--win-rate-threshold` | `0.0` | Stop early when rolling win rate reaches this value (`0.0` = disabled) |
| `--device` | `auto` | `cpu` / `cuda` / `mps` / `auto` |

Run `python -m rl_agent.train --help` for the full list.

---

## Evaluation

Evaluate a checkpoint against a random baseline (default) or against itself:

```bash
# vs random opponent (200 games)
python -m rl_agent.evaluate --agent rl_agent/checkpoints/latest.pt

# vs a frozen copy of the same checkpoint
python -m rl_agent.evaluate --agent rl_agent/checkpoints/latest.pt --opponent self

# more games, longer horizon
python -m rl_agent.evaluate --agent rl_agent/checkpoints/latest.pt --episodes 500 --max-steps 2000
```

Output:

```
=== Evaluation Results ===
  n_episodes                     200
  win_rate                       0.7350
  draw_rate                      0.0000
  loss_rate                      0.2650
  truncation_rate                0.0150
  avg_game_length                312.4800
  median_game_length             287.0000
```

### Evaluation options

| Flag | Default | Description |
|---|---|---|
| `--agent` | *(required)* | Path to `.pt` checkpoint |
| `--episodes` | `200` | Number of evaluation games |
| `--n-players` | `2` | Number of players |
| `--opponent` | `random` | `random` / `self` / `latest` |
| `--max-steps` | `1000` | Truncation limit per game |
| `--device` | `cpu` | Inference device |

---

## Architecture

```
obs (121 × n_players floats)
        │
   Linear → ReLU          ← stem
        │
  [ResBlock] × n_layers   ← shared trunk
        │
   ┌────┴────┐
Policy head  Value head
(ACTION_DIM  (scalar)
 log-probs)
```

**State encoding** — float32 vector of fixed length `121 × 7 = 847`.
- Channels 0–5 (121 floats each): binary piece-occupancy. Current player first, then opponents in turn order. Absent player slots are zero. Fixed size means one model works for any player count.
- Channel 6 (121 floats): binary mask of the current player's goal cells. Gives the network explicit geometric knowledge of where to move without relying purely on reward.

**Action encoding** — `pin_index × 121 + destination_cell` (1 210 possible actions). A boolean legal-action mask is applied before softmax so the policy never wastes probability on illegal moves.

**Rewards** — dense reward every step:
- `(total hex distance before − total hex distance after) × 0.01` per move (fires on every step, not just at the goal).
- `+0.05` each time a piece enters the goal zone.
- `+1.0` on winning, `−1.0` on losing (loss applied retroactively by the trainer).

**Value function** — trained with PPO value-loss clipping to prevent large updates from corrupting the advantage estimates that drive the policy.

**Discount factor** — `gamma = 0.997` (not 0.99). With 1 000-step episodes, `0.99^1000 ≈ 0`, making the terminal win signal invisible to distant states. `0.997^1000 ≈ 0.05`, keeping it meaningful.

**Bootstrap correction** — when an episode is truncated by the step limit (not a natural terminal), the trainer passes the value-function estimate of the final state to GAE instead of 0. Using 0 would systematically under-estimate future rewards for truncated episodes.

**Self-play** — the learner always controls seat 0. All other seats are filled by a periodically-synced frozen copy of the learner (updated every `--opponent-sync-every` episodes).

---

## Checkpoint format

Each `.pt` file is a `torch.save` dict with the following keys:

| Key | Content |
|---|---|
| `episode` | Episode number at save time |
| `net` | `PolicyValueNet` `state_dict` |
| `opt` | Adam optimiser `state_dict` |
| `cfg` | `PPOConfig` as a plain dict (architecture + hypers) |
| `extra` | `total_steps`, `win_rate` at save time |

Load in Python:

```python
from rl_agent.agent import PPOAgent

agent = PPOAgent.from_checkpoint("rl_agent/checkpoints/latest.pt")
agent.net.eval()

obs = ...           # float32 numpy array, shape (121 * n_players,)
legal_actions = ... # list of int
action, log_prob, value = agent.select_action(obs, legal_actions)
```

---

## Project layout

```
rl_agent/
├── env.py        # ChineseCheckersEnv  — headless game, imports single system/ read-only
├── network.py    # PolicyValueNet      — MLP actor-critic
├── agent.py      # PPOAgent            — action selection, buffer, PPO update, save/load
├── trainer.py    # SelfPlayTrainer     — episode loop, opponent sync, checkpointing
├── train.py      # CLI entry point     — python -m rl_agent.train
└── evaluate.py   # Evaluation script   — python -m rl_agent.evaluate
```
