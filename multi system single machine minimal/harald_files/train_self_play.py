import random
from collections import deque
import os
import sys
import time
import pickle
import shutil
import torch
import torch.nn.functional as F
import torch.optim as optim
import json
import numpy as np

try:
    from tqdm import tqdm as _tqdm
    def tqdm(iterable=None, **kwargs):
        return _tqdm(iterable, **kwargs)
    def tqdm_write(msg: str):
        _tqdm.write(msg)
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable
    def tqdm_write(msg: str):
        print(msg, flush=True)

class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer (Schaul et al., 2015).

    New samples receive max_priority so they are guaranteed to be seen at least
    once.  After a batch is drawn with sample_with_idx(), call update_priorities()
    with the per-sample TD errors to sharpen the distribution.

    sample() provides a uniform-random fallback (used by the bootstrap phase which
    has no meaningful priority signal).
    """
    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self._buffer: list = []
        self._priorities = np.ones(capacity, dtype=np.float32)
        self._pos = 0
        self._size = 0
        self._max_priority = 1.0

    def add(self, item):
        if self._size < self.capacity:
            self._buffer.append(item)
            self._size += 1
        else:
            self._buffer[self._pos] = item
        self._priorities[self._pos] = self._max_priority
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> list:
        """Uniform random sample — backward-compatible with old ReplayBuffer."""
        n = min(batch_size, self._size)
        indices = np.random.choice(self._size, size=n, replace=False)
        return [self._buffer[i] for i in indices]

    def sample_with_idx(self, batch_size: int) -> tuple[list, list]:
        """Priority-weighted sample; returns (batch, indices) for priority updates."""
        n = min(batch_size, self._size)
        priorities = self._priorities[:self._size]
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        indices = np.random.choice(self._size, size=n, replace=False, p=probs)
        batch = [self._buffer[i] for i in indices]
        return batch, indices.tolist()

    def update_priorities(self, indices: list, errors: list):
        for idx, err in zip(indices, errors):
            if 0 <= idx < self._size:
                p = abs(float(err)) + 1e-6
                self._priorities[idx] = p
                if p > self._max_priority:
                    self._max_priority = p

    def __len__(self) -> int:
        return self._size

# Keep name alias so any external code using ReplayBuffer still works
ReplayBuffer = PrioritizedReplayBuffer


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)


if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from policy_template import build_model, save_model, load_model, GraphState, MyPolicy, HeuristicPolicy, axial_dist, NeuralMCTS
from policies import RandomPolicy
from collections import defaultdict
from environment import ChineseCheckersEnv
import copy

MODEL_HIDDEN_DIM = 128
MODEL_NUM_LAYERS = 4

ARCH_NAME = f"gnn_h{MODEL_HIDDEN_DIM}_l{MODEL_NUM_LAYERS}"

BASE_CHECKPOINT_DIR = os.path.join("checkpoints", ARCH_NAME)
BOOTSTRAP_CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, "bootstrap")
SELFPLAY_CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, "self_play")

TRAIN_PLAYER_COUNTS = [2, 3, 4, 5, 6]
TRAIN_PLAYER_COUNT_WEIGHTS = [0.40, 0.15, 0.15, 0.15, 0.15]

# ============================================================
# OPPONENT POOL CONFIG
# ============================================================

# Warm-start opponent mix
WARMSTART_OPPONENT_WEIGHTS = {"heuristic": 0.50,
                              "random": 0.30,
                              "last_checkpoint": 0.20}

# Warm-start data diversification.
# Prefix rollouts create legal, game-like but more varied positions before the
# heuristic target labels are collected.
WARMSTART_POSITION_NOISE = True
BOOTSTRAP_FROM_MIXED_ROLLOUTS = True
WARMSTART_USE_MIXED_ROLLOUT_PREFIX = WARMSTART_POSITION_NOISE and BOOTSTRAP_FROM_MIXED_ROLLOUTS
WARMSTART_PREFIX_ROLLOUT_MAX_MOVES = 80
WARMSTART_PREFIX_ROLLOUT_PROB = 0.70
WARMSTART_COLLECT_ALL_HEURISTIC_SEATS = True
WARMSTART_MAX_LABEL_MOVES_AFTER_PREFIX = 220
WARMSTART_ROLLOUT_POLICY_WEIGHTS = {"heuristic": 0.35,
                                    "noisy_heuristic": 0.25,
                                    "random": 0.25,
                                    "last_checkpoint": 0.15}

# Self-play opponent mix
SELFPLAY_OPPONENT_WEIGHTS = {"champion": 0.25,
                             "warmstart_final": 0.15,
                             "heuristic": 0.15,
                             "random": 0.10,
                             "random_checkpoint": 0.15,
                             "current_model": 0.20}

# During self-play, train from every current-model seat that is MCTS-guided.
# Opponent seats controlled by heuristic/random/checkpoints remain data generators,
# but are not treated as learner-quality policy labels.
COLLECT_ALL_CURRENT_MODEL_MCTS_SEATS = True

# Use MCTS visit distributions for exploration in training games. Evaluation and
# promotion still use deterministic policies elsewhere.
MCTS_TRAIN_SAMPLE_UNTIL_MOVE = 120
MCTS_TRAIN_TEMPERATURE = 1.0
MCTS_TRAIN_LATE_TEMPERATURE = 0.25

# Keep value targets equivalent to the pre-merge self-play pipeline: every
# stored position for a colour receives that colour's final/progress value.
GAMMA = 1.0

# ============================================================
# CHECKPOINT PROMOTION CONFIG
# ============================================================

PROMOTION_ENABLED = True

PROMOTION_BLOCK_GAMES = 100
PROMOTION_MATCHES = 50
PROMOTION_PLAYER_COUNTS = [2, 3, 4, 5, 6]
PROMOTION_MAX_MOVES = 300

# Primary criterion
PROMOTION_MIN_WINRATE = 0.55

# Secondary criteria
PROMOTION_MAX_TRUNCATION_RATE = 0.15
PROMOTION_MAX_AVG_WIN_MOVES = 220.0
PROMOTION_MIN_PROGRESS_SCORE = 0.0

# Champion path
CHAMPION_NAME = "champion.pt"

# ============================================================
# HYBRID REJECTION CONFIG
# ============================================================

# If a challenger is rejected but not clearly broken, continue training it.
CONTINUE_REJECTED_CHALLENGER = True

# Reset to champion if challenger performs very poorly.
REJECT_RESET_MIN_WINRATE = 0.35

# Reset to champion if challenger creates too many truncated games.
REJECT_RESET_MAX_TRUNCATION_RATE = 0.60

# Optional: reset if adjudication/degenerate rates are too high.
REJECT_RESET_MAX_STALL_ADJ_RATE = 0.35
REJECT_RESET_MAX_STRANDED_HOME_ADJ_RATE = 0.40
REJECT_RESET_MAX_REPETITION_ADJ_RATE = 0.35

# ============================================================
# LEAGUE PROMOTION EVALUATION CONFIG
# ============================================================

PROMOTION_USE_LEAGUE_EVALUATION = True

# The champion is always guaranteed to occupy at least one opponent seat.
# These weights are used for the remaining non-challenger seats.
PROMOTION_LEAGUE_OPPONENT_WEIGHTS = {"champion": 0.35,
                                     "heuristic": 0.20,
                                     "random": 0.10,
                                     "warmstart_final": 0.10,
                                     "random_checkpoint": 0.25}

# Promotion thresholds based on equal-strength baseline.
# Equal-strength expected win rate is 1 / num_players.
PROMOTION_2P_MARGIN = 0.02          # 2p requires 0.52
PROMOTION_OVERALL_MARGIN = 0.04     # overall must beat equal baseline by 4 percentage points

# Avoid promoting models that collapse in a specific player count.
PROMOTION_MIN_BY_COUNT_MARGIN = -0.08
# Example: in 6p, equal is 0.1667. With -0.08 margin, minimum allowed is ~0.0867.

# Rejection/reset thresholds.
# These should be much looser than promotion thresholds.
REJECT_RESET_BELOW_EXPECTED_MARGIN = 0.08
REJECT_RESET_2P_MIN_WINRATE = 0.30

# Policy and training helpers
# ============================================================
def build_policy_from_checkpoint(path: str, device: str = "cpu"):
    model = load_model(path, device=device)
    return MyPolicy(model=model, device=device)

def progress_print(prefix: str, msg: str) -> None:
    print(f"[{prefix}] {msg}", flush=True)

def action_quality_metrics(history, board, focus_colour: str | None = None) -> dict:
    """
    Estimate move quality from game history.

    If focus_colour is provided, only count moves made by that colour.
    """
    if not history:
        return {"forward_moves": 0,
                "backward_moves": 0,
                "jump_moves": 0,
                "undo_moves": 0,
                "progress_score": 0.0}

    progress_score = 0.0
    forward_moves = 0
    backward_moves = 0
    jump_moves = 0
    undo_moves = 0

    last_move_by_colour = {}

    for mv in history:
        colour = mv.get("colour")
        if focus_colour is not None and colour != focus_colour:
            continue

        from_idx = mv.get("from")
        to_idx = mv.get("to")
        if colour is None or from_idx is None or to_idx is None:
            continue

        from_idx = int(from_idx)
        to_idx = int(to_idx)

        if from_idx < 0 or to_idx < 0:
            continue

        from_cell = board.cells[from_idx]
        to_cell = board.cells[to_idx]

        target_colour = board.colour_opposites[colour]
        target_idxs = board.axial_of_colour(target_colour)
        target_cells = [board.cells[i] for i in target_idxs]

        def min_dist(cell):
            return min(axial_dist(cell, tgt) for tgt in target_cells)

        before = min_dist(from_cell)
        after = min_dist(to_cell)
        gain = before - after

        progress_score += gain
        if gain > 0:
            forward_moves += 1
        elif gain < 0:
            backward_moves += 1

        jump_distance = axial_dist(from_cell, to_cell)
        if jump_distance > 1:
            jump_moves += 1

        own_last_move = last_move_by_colour.get(colour)
        if own_last_move is not None and own_last_move == (to_idx, from_idx):
            undo_moves += 1

        last_move_by_colour[colour] = (from_idx, to_idx)

    return {"forward_moves": forward_moves,
            "backward_moves": backward_moves,
            "jump_moves": jump_moves,
            "undo_moves": undo_moves,
            "progress_score": progress_score}

def choose_learner_colour(turn_order, rng: random.Random | None = None) -> str:
    rng = rng or random
    return rng.choice(list(turn_order))

def find_warmstart_final_checkpoint() -> str | None:
    path = os.path.join(BOOTSTRAP_CHECKPOINT_DIR, "shared_model_final.pt")
    return path if os.path.exists(path) else None

def build_league_policy(kind: str, *,
                        champion_ckpt: str,
                        device: str,
                        warmstart_final_path: str | None,
                        random_checkpoint_path: str | None):
    """
    Build one evaluation opponent policy.

    The challenger is handled separately. This function only creates opponents.
    """
    if kind == "champion":
        return build_policy_from_checkpoint(champion_ckpt, device=device)

    if kind == "heuristic":
        return HeuristicPolicy(epsilon=0.0)

    if kind == "random":
        return build_random_policy()

    if kind == "warmstart_final":
        if warmstart_final_path is not None and os.path.exists(warmstart_final_path):
            return build_policy_from_checkpoint(warmstart_final_path, device=device)
        return HeuristicPolicy(epsilon=0.0)

    if kind == "random_checkpoint":
        if random_checkpoint_path is not None and os.path.exists(random_checkpoint_path):
            return build_policy_from_checkpoint(random_checkpoint_path, device=device)
        return HeuristicPolicy(epsilon=0.0)

    raise ValueError(f"Unknown league opponent kind: {kind}")

def sample_league_opponent_kind(*, has_warmstart_final: bool, has_random_checkpoint: bool) -> str:
    allowed = ["champion", "heuristic", "random"]

    if has_warmstart_final:
        allowed.append("warmstart_final")

    if has_random_checkpoint:
        allowed.append("random_checkpoint")

    return weighted_sample_kind(PROMOTION_LEAGUE_OPPONENT_WEIGHTS, allowed)
# ============================================================

# Evaluation and promotion helpers
def evaluate_one_match(champion_ckpt: str,
                       challenger_ckpt: str,
                       num_players: int,
                       device: str,
                       max_moves: int,
                       seed: int = 0,
                       challenger_seat_index: int | None = None) -> dict:
    rng = random.Random(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    env = ChineseCheckersEnv(num_players=num_players)
    env.reset()

    champion_policy = build_policy_from_checkpoint(champion_ckpt, device=device)
    challenger_policy = build_policy_from_checkpoint(challenger_ckpt, device=device)

    turn_order = list(env.turn_order)

    if challenger_seat_index is None:
        challenger_seat_index = seed % len(turn_order)

    challenger_colour = turn_order[challenger_seat_index]

    policies_by_colour = {}
    for colour in turn_order:
        if colour == challenger_colour:
            policies_by_colour[colour] = challenger_policy
        else:
            policies_by_colour[colour] = champion_policy

    result = env.run_policies(policies_by_colour, max_moves=max_moves)
    state = result["state"]
    scores = result["scores"]
    history = result.get("history", [])
    truncated = bool(result.get("truncated", False))

    my_player = next(p for p in state["players"] if p["colour"] == challenger_colour)
    my_status = my_player["status"]

    quality = action_quality_metrics(history=history,
                                     board=env.game.board,
                                     focus_colour=challenger_colour)

    final_score = scores.get(challenger_colour, {}).get("final_score", 0.0)

    adjudication_reason = get_adjudication_reason_from_state(state)

    return {"challenger_colour": challenger_colour,
            "challenger_seat_index": challenger_seat_index,
            "status": my_status,
            "final_score": final_score,
            "move_count": state.get("move_count", 0),
            "truncated": truncated,
            "adjudication_reason": adjudication_reason,
            "adjudication_type": classify_adjudication_reason(adjudication_reason),
            "home_pieces": extract_state_metric_for_colour(state, "home_pieces", challenger_colour, default=0),
            "stranded_home_pieces": extract_state_metric_for_colour(state, "stranded_home_pieces", challenger_colour, default=0),
            "training_progress": extract_state_metric_for_colour(state, "training_progress", challenger_colour, default=0.0),
            "quality": quality}

def evaluate_one_league_match(champion_ckpt: str,
                              challenger_ckpt: str,
                              num_players: int,
                              device: str,
                              max_moves: int,
                              seed: int = 0,
                              challenger_seat_index: int | None = None,
                              checkpoint_dirs: list[str] | None = None) -> dict:
    """
    Evaluate challenger in a mixed-opponent league table.

    Guarantees:
      - challenger occupies exactly one seat
      - champion occupies at least one opponent seat
      - remaining opponent seats are sampled from the league pool
    """
    rng = random.Random(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    env = ChineseCheckersEnv(num_players=num_players)
    env.reset()

    challenger_policy = build_policy_from_checkpoint(challenger_ckpt, device=device)
    champion_policy = build_policy_from_checkpoint(champion_ckpt, device=device)

    turn_order = list(env.turn_order)

    if challenger_seat_index is None:
        challenger_seat_index = seed % len(turn_order)

    challenger_colour = turn_order[challenger_seat_index]
    opponent_colours = [c for c in turn_order if c != challenger_colour]

    if not opponent_colours:
        raise RuntimeError("League evaluation requires at least one opponent")

    # Always force the current champion into one opponent seat.
    champion_colour = opponent_colours[seed % len(opponent_colours)]

    checkpoint_dirs = checkpoint_dirs or [SELFPLAY_CHECKPOINT_DIR, BOOTSTRAP_CHECKPOINT_DIR]

    warmstart_final_path = find_warmstart_final_checkpoint()
    random_checkpoint_path = sample_random_checkpoint(checkpoint_dirs,
                                                      exclude={champion_ckpt, challenger_ckpt})

    has_warmstart_final = warmstart_final_path is not None and os.path.exists(warmstart_final_path)
    has_random_checkpoint = random_checkpoint_path is not None and os.path.exists(random_checkpoint_path)

    policies_by_colour = {}
    opponent_kinds_by_colour = {}

    for colour in turn_order:
        if colour == challenger_colour:
            policies_by_colour[colour] = challenger_policy
            opponent_kinds_by_colour[colour] = "challenger"

        elif colour == champion_colour:
            policies_by_colour[colour] = champion_policy
            opponent_kinds_by_colour[colour] = "champion_forced"

        else:
            kind = sample_league_opponent_kind(has_warmstart_final=has_warmstart_final,
                                               has_random_checkpoint=has_random_checkpoint)

            policies_by_colour[colour] = build_league_policy(kind,
                                                             champion_ckpt=champion_ckpt,
                                                             device=device,
                                                             warmstart_final_path=warmstart_final_path,
                                                             random_checkpoint_path=random_checkpoint_path)
            opponent_kinds_by_colour[colour] = kind

    result = env.run_policies(policies_by_colour, max_moves=max_moves)

    state = result["state"]
    scores = result["scores"]
    history = result.get("history", [])
    truncated = bool(result.get("truncated", False))

    my_player = next(p for p in state["players"] if p["colour"] == challenger_colour)
    my_status = my_player["status"]

    quality = action_quality_metrics(history=history,
                                     board=env.game.board,
                                     focus_colour=challenger_colour)

    final_score = scores.get(challenger_colour, {}).get("final_score", 0.0)

    adjudication_reason = get_adjudication_reason_from_state(state) if "get_adjudication_reason_from_state" in globals() else state.get("adjudication_reason")

    return {"challenger_colour": challenger_colour,
            "challenger_seat_index": challenger_seat_index,
            "status": my_status,
            "final_score": final_score,
            "move_count": state.get("move_count", 0),
            "truncated": truncated,
            "adjudication_reason": adjudication_reason,
            "adjudication_type": classify_adjudication_reason(adjudication_reason) if "classify_adjudication_reason" in globals() else None,
            "home_pieces": extract_state_metric_for_colour(state, "home_pieces", challenger_colour, default=0) if "extract_state_metric_for_colour" in globals() else 0,
            "stranded_home_pieces": extract_state_metric_for_colour(state, "stranded_home_pieces", challenger_colour, default=0) if "extract_state_metric_for_colour" in globals() else 0,
            "training_progress": extract_state_metric_for_colour(state, "training_progress", challenger_colour, default=0.0) if "extract_state_metric_for_colour" in globals() else 0.0,
            "quality": quality,
            "opponent_kinds_by_colour": opponent_kinds_by_colour,
            "forced_champion_colour": champion_colour,
            "evaluation_mode": "league"}

def evaluate_checkpoint_promotion(champion_ckpt: str,
                                  challenger_ckpt: str,
                                  device: str = "cpu",
                                  matches_per_player_count: int = 20,
                                  player_counts=(2, 3, 4, 5, 6),
                                  max_moves: int = 300,
                                  use_league_evaluation: bool = PROMOTION_USE_LEAGUE_EVALUATION) -> dict:
    all_results = []

    for n_players in player_counts:
        print(f"[promotion] evaluating {n_players}-player matches...", flush=True)
        for i in range(matches_per_player_count):
            seat_index = i % n_players
            if use_league_evaluation:
                out = evaluate_one_league_match(champion_ckpt=champion_ckpt,
                                                challenger_ckpt=challenger_ckpt,
                                                num_players=n_players,
                                                device=device,
                                                max_moves=max_moves,
                                                seed=10_000 * n_players + i,
                                                challenger_seat_index=seat_index,
                                                checkpoint_dirs=[SELFPLAY_CHECKPOINT_DIR, BOOTSTRAP_CHECKPOINT_DIR])
            else:
                out = evaluate_one_match(champion_ckpt=champion_ckpt,
                                         challenger_ckpt=challenger_ckpt,
                                         num_players=n_players,
                                         device=device,
                                         max_moves=max_moves,
                                         seed=10_000 * n_players + i,
                                         challenger_seat_index=seat_index)
            out["num_players"] = n_players
            all_results.append(out)

    # -------------------------
    # Aggregate summaries
    # -------------------------
    overall = summarize_promotion_results(all_results)
    by_player_count = summarize_by_player_count(all_results)

    two_player_results = [r for r in all_results if r["num_players"] == 2]
    two_player_summary = summarize_promotion_results(two_player_results)

    two_wins = two_player_summary["wins"]
    two_draws = two_player_summary["draws"]
    two_losses = two_player_summary["losses"]
    two_win_rate = two_player_summary["win_rate"]

    expected_equal_overall = expected_equal_win_rate_for_eval(player_counts)
    overall_margin_vs_equal = overall["win_rate"] - expected_equal_overall

    for n_players, summary in by_player_count.items():
        expected_n = expected_equal_win_rate_for_player_count(n_players)
        summary["expected_equal_win_rate"] = expected_n
        summary["margin_vs_equal"] = summary["win_rate"] - expected_n

    # Keep your current promotion behavior, but add diagnostic gates.
    two_player_expected = expected_equal_win_rate_for_player_count(2)
    two_player_required = two_player_expected + PROMOTION_2P_MARGIN

    primary_pass = two_win_rate >= two_player_required

    secondary_pass = (overall["truncation_rate"] <= PROMOTION_MAX_TRUNCATION_RATE
                      and (overall["avg_win_moves"] == 0.0 or overall["avg_win_moves"] <= PROMOTION_MAX_AVG_WIN_MOVES)
                      and overall["avg_progress_score"] >= PROMOTION_MIN_PROGRESS_SCORE)

    overall_pass = overall["win_rate"] >= (expected_equal_overall + PROMOTION_OVERALL_MARGIN)
    by_count_pass = True
    for n_players, summary in by_player_count.items():
        min_allowed = summary["expected_equal_win_rate"] + PROMOTION_MIN_BY_COUNT_MARGIN
        if summary["win_rate"] < min_allowed:
            by_count_pass = False
            break
    
    promoted = primary_pass and overall_pass and by_count_pass and secondary_pass

    return {"evaluation_mode": "league" if use_league_evaluation else "champion_only",
            "promoted": promoted,
            "primary_pass": primary_pass,
            "secondary_pass": secondary_pass,
            "overall_pass": overall_pass,
            "by_count_pass": by_count_pass,
            "two_player_required": two_player_required,
            "two_player_wins": two_wins,
            "two_player_draws": two_draws,
            "two_player_losses": two_losses,
            "two_player_win_rate": two_win_rate,
            "wins": overall["wins"],
            "draws": overall["draws"],
            "losses": overall["losses"],
            "win_rate": overall["win_rate"],
            "draw_rate": overall["draw_rate"],
            "loss_rate": overall["loss_rate"],
            "truncation_rate": overall["truncation_rate"],
            "avg_win_moves": overall["avg_win_moves"],
            "avg_final_score": overall["avg_final_score"],
            "avg_progress_score": overall["avg_progress_score"],
            "avg_backward_moves": overall["avg_backward_moves"],
            "avg_undo_moves": overall["avg_undo_moves"],
            "avg_jump_moves": overall["avg_jump_moves"],
            "overall": overall,
            "expected_equal_overall": expected_equal_overall,
            "overall_margin_vs_equal": overall_margin_vs_equal,
            "by_player_count": by_player_count,
            "raw_results": all_results}

def print_promotion_report(report: dict) -> None:
    print("\n" + "=" * 80)
    print("CHECKPOINT PROMOTION REPORT")
    print("=" * 80)

    overall = report.get("overall", {})

    print(f"evaluation_mode       : {report.get('evaluation_mode', 'unknown')}")
    print(f"expected_equal_overall: {report.get('expected_equal_overall', 0.0):.3f}")
    print(f"overall_margin_equal  : {report.get('overall_margin_vs_equal', 0.0):+.3f}")
    print(f"overall_pass          : {report.get('overall_pass', False)}")
    print(f"by_count_pass         : {report.get('by_count_pass', False)}")
    print(f"2p required           : {report.get('two_player_required', 0.0):.3f}")

    print(f"promoted              : {report['promoted']}")
    print(f"primary_pass          : {report['primary_pass']}")
    print(f"secondary_pass        : {report['secondary_pass']}")

    print("\n--- Overall ---")
    print(f"wins/draws/losses     : {report['wins']} / {report['draws']} / {report['losses']}")
    print(f"win_rate              : {report['win_rate']:.3f}")
    print(f"draw_rate             : {report['draw_rate']:.3f}")
    print(f"loss_rate             : {report['loss_rate']:.3f}")
    print(f"truncation_rate       : {report['truncation_rate']:.3f}")
    print(f"adjudication_rate     : {overall.get('adjudication_rate', 0.0):.3f}")
    print(f"max_moves_rate        : {overall.get('max_moves_rate', 0.0):.3f}")
    print(f"stall_adj_rate        : {overall.get('stall_adjudication_rate', 0.0):.3f}")
    print(f"stranded_home_adj_rate: {overall.get('stranded_home_adjudication_rate', 0.0):.3f}")
    print(f"repetition_adj_rate   : {overall.get('repetition_adjudication_rate', 0.0):.3f}")

    print("\n--- Move quality ---")
    print(f"avg_moves             : {overall.get('avg_moves', 0.0):.2f}")
    print(f"avg_win_moves         : {report['avg_win_moves']:.2f}")
    print(f"avg_final_score       : {report['avg_final_score']:.2f}")
    print(f"avg_progress          : {report['avg_progress_score']:.2f}")
    print(f"avg_backward          : {report['avg_backward_moves']:.2f}")
    print(f"avg_undo              : {report['avg_undo_moves']:.2f}")
    print(f"avg_jump_moves        : {report['avg_jump_moves']:.2f}")

    print("\n--- Home-piece diagnostics ---")
    print(f"avg_home_pieces       : {overall.get('avg_home_pieces', 0.0):.2f}")
    print(f"avg_stranded_home     : {overall.get('avg_stranded_home_pieces', 0.0):.2f}")
    print(f"max_stranded_home     : {overall.get('max_stranded_home_pieces', 0)}")

    print("\n--- By player count ---")
    by_player_count = report.get("by_player_count", {})
    for n_players in sorted(by_player_count):
        s = by_player_count[n_players]
        print(f"{n_players}p | "
              f"games={s['games']:3d} "
              f"W/D/L={s['wins']:3d}/{s['draws']:3d}/{s['losses']:3d} "
              f"win={s['win_rate']:.3f} "
              f"exp={s.get('expected_equal_win_rate', 1.0 / n_players):.3f} "
              f"margin={s.get('margin_vs_equal', s['win_rate'] - 1.0 / n_players):+.3f} "
              f"trunc={s['truncation_rate']:.3f} "
              f"adj={s['adjudication_rate']:.3f} "
              f"stall={s['stall_adjudication_rate']:.3f} "
              f"stranded_adj={s['stranded_home_adjudication_rate']:.3f} "
              f"avg_moves={s['avg_moves']:.1f} "
              f"avg_stranded={s['avg_stranded_home_pieces']:.2f} "
              f"undo={s['avg_undo_moves']:.2f}")

    print("=" * 80)

def average_or_zero(xs):
    return sum(xs) / len(xs) if xs else 0.0

def expected_equal_win_rate_for_player_count(num_players: int) -> float:
    """
    If all players are equally strong, one seat is expected to win 1 / num_players.
    """
    return 1.0 / float(num_players)

def expected_equal_win_rate_for_eval(player_counts) -> float:
    """
    Expected aggregate win rate if the candidate is equal strength and each
    player count receives the same number of matches.
    """
    counts = list(player_counts)
    if not counts:
        return 0.0
    return sum(expected_equal_win_rate_for_player_count(n) for n in counts) / len(counts)

def league_adjusted_score(win_rate: float, player_counts) -> float:
    """
    Positive means above equal-strength expectation.
    Negative means below equal-strength expectation.
    """
    return win_rate - expected_equal_win_rate_for_eval(player_counts)

def safe_rate(count: int, total: int) -> float:
    return count / total if total > 0 else 0.0

def get_adjudication_reason_from_state(state: dict) -> str | None:
    """
    Works with the adjudication fields suggested for core.py.
    Falls back cleanly if those fields are not present yet.
    """
    reason = state.get("adjudication_reason")
    if reason:
        return str(reason)

    event = state.get("last_adjudication_event") or {}
    reason = event.get("reason")
    return str(reason) if reason else None

def classify_adjudication_reason(reason: str | None) -> str:
    if not reason:
        return "none"

    reason_upper = reason.upper()

    if "MAX_MOVES" in reason_upper:
        return "max_moves"
    if "STALL" in reason_upper:
        return "stall"
    if "STRANDED_HOME" in reason_upper:
        return "stranded_home"
    if "REPETITION" in reason_upper:
        return "repetition"

    return "other"

def extract_state_metric_for_colour(state: dict, metric_name: str, colour: str, default=0):
    """
    Handles state fields shaped like:
        state["stranded_home_pieces"] = {"red": 1, "blue": 0}
    """
    metric = state.get(metric_name, {})
    if isinstance(metric, dict):
        return metric.get(colour, default)
    return default

def summarize_promotion_results(results: list[dict]) -> dict:
    """
    Summarize promotion/evaluation results for either all games or one player-count slice.
    """
    n = len(results)
    if n == 0:
        return {
            "games": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "win_rate": 0.0,
            "draw_rate": 0.0,
            "loss_rate": 0.0,
            "truncation_rate": 0.0,
            "adjudication_rate": 0.0,
            "max_moves_rate": 0.0,
            "stall_adjudication_rate": 0.0,
            "stranded_home_adjudication_rate": 0.0,
            "repetition_adjudication_rate": 0.0,
            "avg_moves": 0.0,
            "avg_win_moves": 0.0,
            "avg_final_score": 0.0,
            "avg_progress_score": 0.0,
            "avg_backward_moves": 0.0,
            "avg_undo_moves": 0.0,
            "avg_jump_moves": 0.0,
            "avg_home_pieces": 0.0,
            "avg_stranded_home_pieces": 0.0,
            "max_stranded_home_pieces": 0,
        }

    wins = sum(r["status"] == "WIN" for r in results)
    draws = sum(r["status"] == "DRAW" for r in results)
    losses = sum(r["status"] not in ("WIN", "DRAW") for r in results)

    truncated = sum(bool(r.get("truncated", False)) for r in results)

    reason_classes = [classify_adjudication_reason(r.get("adjudication_reason")) for r in results]
    adjudicated = sum(cls != "none" for cls in reason_classes)
    max_moves = sum(cls == "max_moves" for cls in reason_classes)
    stall = sum(cls == "stall" for cls in reason_classes)
    stranded_home_adj = sum(cls == "stranded_home" for cls in reason_classes)
    repetition = sum(cls == "repetition" for cls in reason_classes)

    win_move_counts = [r["move_count"] for r in results if r["status"] == "WIN"]

    stranded_values = [r.get("stranded_home_pieces", 0) for r in results]
    home_values = [r.get("home_pieces", 0) for r in results]

    return {
        "games": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": safe_rate(wins, n),
        "draw_rate": safe_rate(draws, n),
        "loss_rate": safe_rate(losses, n),
        "truncation_rate": safe_rate(truncated, n),
        "adjudication_rate": safe_rate(adjudicated, n),
        "max_moves_rate": safe_rate(max_moves, n),
        "stall_adjudication_rate": safe_rate(stall, n),
        "stranded_home_adjudication_rate": safe_rate(stranded_home_adj, n),
        "repetition_adjudication_rate": safe_rate(repetition, n),
        "avg_moves": average_or_zero([r["move_count"] for r in results]),
        "avg_win_moves": average_or_zero(win_move_counts),
        "avg_final_score": average_or_zero([r["final_score"] for r in results]),
        "avg_progress_score": average_or_zero([r["quality"]["progress_score"] for r in results]),
        "avg_backward_moves": average_or_zero([r["quality"]["backward_moves"] for r in results]),
        "avg_undo_moves": average_or_zero([r["quality"]["undo_moves"] for r in results]),
        "avg_jump_moves": average_or_zero([r["quality"]["jump_moves"] for r in results]),
        "avg_home_pieces": average_or_zero(home_values),
        "avg_stranded_home_pieces": average_or_zero(stranded_values),
        "max_stranded_home_pieces": max(stranded_values) if stranded_values else 0,
    }

def summarize_by_player_count(results: list[dict]) -> dict[int, dict]:
    out = {}
    for n_players in sorted(set(r["num_players"] for r in results)):
        subset = [r for r in results if r["num_players"] == n_players]
        out[n_players] = summarize_promotion_results(subset)
    return out

def append_jsonl(path: str, item: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, default=str) + "\n")

def challenger_collapsed(report: dict) -> tuple[bool, list[str]]:
    """
    Decide whether a rejected challenger is bad enough that we should discard it
    and restart the next block from the champion.

    Returns:
        collapsed: bool
        reasons: list[str]
    """
    reasons = []

    win_rate = report.get("win_rate", 0.0)
    truncation_rate = report.get("truncation_rate", 0.0)

    overall = report.get("overall", {}) or {}

    stall_adj_rate = overall.get("stall_adjudication_rate", 0.0)
    stranded_home_adj_rate = overall.get("stranded_home_adjudication_rate", 0.0)
    repetition_adj_rate = overall.get("repetition_adjudication_rate", 0.0)

    expected_equal = report.get("expected_equal_overall", 0.0)

    if expected_equal > 0:
        reset_floor = expected_equal - REJECT_RESET_BELOW_EXPECTED_MARGIN
    else:
        reset_floor = REJECT_RESET_MIN_WINRATE

    if win_rate < reset_floor:
        reasons.append(f"win_rate={win_rate:.3f} < reset_floor={reset_floor:.3f} "
                       f"(expected_equal={expected_equal:.3f})")

    two_player_win_rate = report.get("two_player_win_rate", 0.0)

    if two_player_win_rate < REJECT_RESET_2P_MIN_WINRATE:
        reasons.append(f"two_player_win_rate={two_player_win_rate:.3f} < {REJECT_RESET_2P_MIN_WINRATE:.3f}")

    if truncation_rate > REJECT_RESET_MAX_TRUNCATION_RATE:
        reasons.append(f"truncation_rate={truncation_rate:.3f} > {REJECT_RESET_MAX_TRUNCATION_RATE:.3f}")

    if stall_adj_rate > REJECT_RESET_MAX_STALL_ADJ_RATE:
        reasons.append(f"stall_adj_rate={stall_adj_rate:.3f} > {REJECT_RESET_MAX_STALL_ADJ_RATE:.3f}")

    if stranded_home_adj_rate > REJECT_RESET_MAX_STRANDED_HOME_ADJ_RATE:
        reasons.append(f"stranded_home_adj_rate={stranded_home_adj_rate:.3f} > {REJECT_RESET_MAX_STRANDED_HOME_ADJ_RATE:.3f}")

    if repetition_adj_rate > REJECT_RESET_MAX_REPETITION_ADJ_RATE:
        reasons.append(f"repetition_adj_rate={repetition_adj_rate:.3f} > {REJECT_RESET_MAX_REPETITION_ADJ_RATE:.3f}")

    return len(reasons) > 0, reasons

class TrainableAgent:
    def __init__(self, name: str, device: str = "cpu", lr: float = 1e-3, hidden_dim: int = 128, num_layers: int = 4):
        self.name = name
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.model = build_model(device=device,
                                 hidden_dim=hidden_dim,
                                 num_layers=num_layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.policy = MyPolicy(model=self.model,
                               device=device,
                               hidden_dim=hidden_dim,
                               num_layers=num_layers)

    def graph_from_observation(self, observation):
        gs = self.policy.graph_builder.build(observation)
        return GraphState(x=gs.x.to(self.device),
                          edge_index=gs.edge_index.to(self.device),
                          legal_actions=gs.legal_actions,
                          controlled_colour=gs.controlled_colour,
                          meta=gs.meta,
                          action_features=gs.action_features.to(self.device))

    def _graph_to_device(self, gs):
        return GraphState(x=gs.x.to(self.device),
                          edge_index=gs.edge_index.to(self.device),
                          legal_actions=gs.legal_actions,
                          action_features=gs.action_features.to(self.device),
                          controlled_colour=gs.controlled_colour,
                          meta=gs.meta)

    def train_policy_batch(self, batch):
        """
        Supervised imitation only. No value loss.
        Best for warm-start.
        """
        if not batch:
            return 0.0

        self.model.train()
        self.optimizer.zero_grad()

        losses = []
        for item in batch:
            gs = self._graph_to_device(item["graph_state"])
            target_action_idx = torch.tensor([item["action_index"]], device=self.device)

            logits, _ = self.model(gs)
            policy_loss = F.cross_entropy(logits.unsqueeze(0), target_action_idx)
            losses.append(policy_loss)

        batch_loss = torch.stack(losses).mean()
        batch_loss.backward()
        self.optimizer.step()

        return float(batch_loss.item())

    def train_policy_value_batch_soft(self, batch, value_weight: float = 0.1,
                                      return_metrics: bool = False,
                                      return_per_sample_errors: bool = False):
        """
        Policy + value training where the policy target is a probability distribution
        over legal actions.

        If return_metrics=True, returns a dict with total/policy/value loss.
        If return_per_sample_errors=True, the dict also contains 'per_sample_errors'
        (list of per-item total losses) for updating prioritized replay priorities.
        """
        if not batch:
            if return_metrics:
                out = {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
                if return_per_sample_errors:
                    out["per_sample_errors"] = []
                return out
            return 0.0

        self.model.train()
        self.optimizer.zero_grad()

        total_losses = []
        policy_losses = []
        value_losses = []

        for item in batch:
            gs = self._graph_to_device(item["graph_state"])
            target_policy = item["target_policy"].to(self.device)
            target_value = torch.tensor(item["target_value"], dtype=torch.float32, device=self.device)

            logits, pred_value = self.model(gs)

            log_probs = F.log_softmax(logits, dim=0)
            policy_loss = -(target_policy * log_probs).sum()
            value_loss = F.mse_loss(pred_value, target_value)

            total_loss = policy_loss + value_weight * value_loss

            total_losses.append(total_loss)
            policy_losses.append(policy_loss.detach())
            value_losses.append(value_loss.detach())

        batch_loss = torch.stack(total_losses).mean()
        batch_loss.backward()
        self.optimizer.step()

        metrics = {"total_loss": float(batch_loss.item()),
                   "policy_loss": float(torch.stack(policy_losses).mean().item()),
                   "value_loss": float(torch.stack(value_losses).mean().item())}

        if return_per_sample_errors:
            metrics["per_sample_errors"] = [float(tl.detach().item()) for tl in total_losses]

        if return_metrics:
            return metrics

        return metrics["total_loss"]

    def select_action_with_index(self, observation):
        graph_state = self.graph_from_observation(observation)
        self.model.eval()
        with torch.no_grad():
            logits, value = self.model(graph_state)

        if logits.numel() == 0:
            raise RuntimeError("No legal actions available")

        action_idx = int(torch.argmax(logits).item())
        pin_id, _, to_idx = graph_state.legal_actions[action_idx]
        return (pin_id, to_idx), action_idx, graph_state

    def load(self, path: str):
        self.model = load_model(path,
                                device=self.device,
                                hidden_dim=self.hidden_dim,
                                num_layers=self.num_layers)
        self.policy = MyPolicy(model=self.model,
                               device=self.device,
                               hidden_dim=self.hidden_dim,
                               num_layers=self.num_layers)

    def save(self, path: str):
        save_model(self.model, path,
                   hidden_dim=self.hidden_dim,
                   num_layers=self.num_layers)

    def save_full(self, path: str):
        """Save model weights + optimizer state for complete resume."""
        self.save(path)
        train_state = {"optimizer": self.optimizer.state_dict()}
        torch.save(train_state, path + ".train")

    def load_full(self, path: str):
        """Load model weights and, if present, optimizer state."""
        self.load(path)
        opt_path = path + ".train"
        if os.path.exists(opt_path):
            try:
                train_state = torch.load(opt_path, map_location=self.device)
                self.optimizer.load_state_dict(train_state["optimizer"])
            except Exception as e:
                print(f"[resume] could not load optimizer state from {opt_path}: {e}", flush=True)

# Helpers for checkpoint management and opponent sampling
# ============================================================
def collect_checkpoint_candidates(checkpoint_dirs: list[str], include_champion: bool = False) -> list[str]:
    candidates = []

    for checkpoint_dir in checkpoint_dirs:
        if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
            continue

        for name in os.listdir(checkpoint_dir):
            full = os.path.join(checkpoint_dir, name)

            if name == "champion.pt":
                if include_champion:
                    candidates.append(full)
                continue

            if name == "shared_model_final.pt":
                candidates.append(full)
                continue

            if name.startswith("shared_model_") and name.endswith(".pt"):
                candidates.append(full)
                continue

            if name.startswith("challenger_block_") and name.endswith(".pt"):
                candidates.append(full)
                continue

    # deterministic ordering
    candidates = sorted(set(candidates))
    return candidates

def sample_random_checkpoint(checkpoint_dirs: list[str], exclude: set[str] | None = None) -> str | None:
    exclude = exclude or set()
    candidates = [p for p in collect_checkpoint_candidates(checkpoint_dirs, include_champion=False) if p not in exclude]
    if not candidates:
        return None
    return random.choice(candidates)

def build_random_policy(seed: int | None = None):
    try:
        return RandomPolicy(seed=seed)
    except TypeError:
        return RandomPolicy()

def build_checkpoint_policy_if_exists(path: str | None, device: str):
    if path is None or not os.path.exists(path):
        return None
    return build_policy_from_checkpoint(path, device=device)

def weighted_sample_kind(weight_map: dict[str, float], allowed_kinds: list[str]) -> str:
    kinds = [k for k in allowed_kinds if k in weight_map]
    weights = [weight_map[k] for k in kinds]
    total = sum(weights)

    if total <= 0:
        return random.choice(allowed_kinds)

    weights = [w / total for w in weights]
    return random.choices(kinds, weights=weights, k=1)[0]

def sample_warmstart_seat_plan(turn_order: list[str], has_last_checkpoint: bool) -> tuple[str, dict[str, str]]:
    """
    One tracked learner seat (teacher-controlled for imitation),
    other seats are sampled from the warm-start opponent pool.
    """
    learner_colour = random.choice(list(turn_order))

    allowed = ["heuristic", "random"]
    if has_last_checkpoint:
        allowed.append("last_checkpoint")

    seat_plan = {}
    for colour in turn_order:
        if colour == learner_colour:
            seat_plan[colour] = "teacher_seat"
        else:
            seat_plan[colour] = weighted_sample_kind(WARMSTART_OPPONENT_WEIGHTS, allowed)

    return learner_colour, seat_plan

def sample_selfplay_seat_plan(turn_order: list[str],
                              has_champion: bool,
                              has_warmstart_final: bool,
                              has_random_checkpoint: bool) -> tuple[str, dict[str, str]]:
    """
    One tracked learner seat (MCTS-labeled current model),
    other seats are sampled from the self-play opponent pool.
    """
    learner_colour = random.choice(list(turn_order))

    allowed = ["heuristic", "random", "current_model"]

    if has_champion:
        allowed.append("champion")
    if has_warmstart_final:
        allowed.append("warmstart_final")
    if has_random_checkpoint:
        allowed.append("random_checkpoint")

    seat_plan = {}
    for colour in turn_order:
        if colour == learner_colour:
            seat_plan[colour] = "tracked_current"
        else:
            seat_plan[colour] = weighted_sample_kind(SELFPLAY_OPPONENT_WEIGHTS, allowed)

    return learner_colour, seat_plan

def build_warmstart_policy_cache(seat_plan: dict[str, str],
                                 heuristic: HeuristicPolicy,
                                 last_checkpoint_path: str | None,
                                 device: str) -> dict[str, MyPolicy | HeuristicPolicy | RandomPolicy]:
    cache = {}

    last_checkpoint_policy = build_checkpoint_policy_if_exists(last_checkpoint_path, device=device)

    for colour, kind in seat_plan.items():
        if kind == "teacher_seat":
            cache[colour] = heuristic
        elif kind == "heuristic":
            cache[colour] = heuristic
        elif kind == "random":
            cache[colour] = build_random_policy()
        elif kind == "last_checkpoint":
            cache[colour] = last_checkpoint_policy if last_checkpoint_policy is not None else heuristic
        else:
            raise ValueError(f"Unknown warm-start seat kind: {kind}")

    return cache

def add_heuristic_imitation_example(learner: TrainableAgent,
                                    observation: dict,
                                    chosen_action: tuple[int, int],
                                    examples_by_colour: dict) -> None:
    colour = observation["colour"]
    gs = learner.graph_from_observation(observation)

    action_index = None
    for i, (pin_id, _, to_idx) in enumerate(gs.legal_actions):
        if (pin_id, to_idx) == chosen_action:
            action_index = i
            break

    if action_index is None:
        raise RuntimeError("Heuristic action not found in legal action list")

    examples_by_colour[colour].append({"graph_state": gs,
                                       "action_index": action_index})

def sample_warmstart_rollout_kind(has_last_checkpoint: bool) -> str:
    allowed = ["heuristic", "noisy_heuristic", "random"]
    if has_last_checkpoint:
        allowed.append("last_checkpoint")
    return weighted_sample_kind(WARMSTART_ROLLOUT_POLICY_WEIGHTS, allowed)

def run_warmstart_mixed_rollout_prefix(env: ChineseCheckersEnv,
                                       *,
                                       max_prefix_moves: int,
                                       prefix_probability: float,
                                       heuristic: HeuristicPolicy,
                                       noisy_heuristic: HeuristicPolicy,
                                       last_checkpoint_path: str | None,
                                       device: str) -> dict:
    """
    Advance the environment with mixed legal policies before collecting labels.

    This creates broader, still-reachable positions for warm-start imitation.
    """
    if max_prefix_moves <= 0 or random.random() > prefix_probability:
        return {"prefix_moves": 0,
                "prefix_truncated": False,
                "prefix_kind_counts": {}}

    last_checkpoint_policy = build_checkpoint_policy_if_exists(last_checkpoint_path, device=device)
    has_last_checkpoint = last_checkpoint_policy is not None
    random_policy = build_random_policy()

    kind_counts = defaultdict(int)
    prefix_moves = random.randint(0, max_prefix_moves)
    moves_done = 0

    for _ in range(prefix_moves):
        if env.game.status == "FINISHED":
            break

        colour = env.current_turn_colour
        if colour is None:
            break

        obs = env.observe(colour)
        kind = sample_warmstart_rollout_kind(has_last_checkpoint=has_last_checkpoint)
        kind_counts[kind] += 1

        if kind == "heuristic":
            action = heuristic.select_action(obs)
        elif kind == "noisy_heuristic":
            action = noisy_heuristic.select_action(obs)
        elif kind == "random":
            action = random_policy.select_action(obs)
        elif kind == "last_checkpoint" and last_checkpoint_policy is not None:
            action = last_checkpoint_policy.select_action(obs)
        else:
            action = heuristic.select_action(obs)

        step_result = env.step(colour, action)
        moves_done += 1
        if step_result.done:
            break

    return {"prefix_moves": moves_done,
            "prefix_truncated": env.game.status == "FINISHED",
            "prefix_kind_counts": dict(kind_counts)}

def build_selfplay_policy_cache(seat_plan: dict[str, str],
                                learner: TrainableAgent,
                                heuristic: HeuristicPolicy,
                                champion_path: str | None,
                                warmstart_final_path: str | None,
                                random_checkpoint_path: str | None,
                                device: str) -> dict[str, MyPolicy | HeuristicPolicy | RandomPolicy]:
    cache = {}

    current_model_policy = MyPolicy(model=learner.model,
                                    device=device,
                                    hidden_dim=MODEL_HIDDEN_DIM,
                                    num_layers=MODEL_NUM_LAYERS)

    champion_policy = build_checkpoint_policy_if_exists(champion_path, device=device)
    warmstart_policy = build_checkpoint_policy_if_exists(warmstart_final_path, device=device)
    random_checkpoint_policy = build_checkpoint_policy_if_exists(random_checkpoint_path, device=device)

    for colour, kind in seat_plan.items():
        if kind == "tracked_current":
            cache[colour] = "tracked_current"   # sentinel; handled explicitly in loop
        elif kind == "champion":
            cache[colour] = champion_policy if champion_policy is not None else current_model_policy
        elif kind == "warmstart_final":
            cache[colour] = warmstart_policy if warmstart_policy is not None else heuristic
        elif kind == "heuristic":
            cache[colour] = heuristic
        elif kind == "random":
            cache[colour] = build_random_policy()
        elif kind == "random_checkpoint":
            cache[colour] = random_checkpoint_policy if random_checkpoint_policy is not None else heuristic
        elif kind == "current_model":
            cache[colour] = current_model_policy
        else:
            raise ValueError(f"Unknown self-play seat kind: {kind}")

    return cache
# ============================================================

"""
MCTS for self play
"""
class LocalSearchEnv:
    """
    Adapter so NeuralMCTS can search inside the existing in-process environment.
    """
    def __init__(self, env: ChineseCheckersEnv):
        self.env = env

    def clone(self):
        return LocalSearchEnv(copy.deepcopy(self.env))

    def current_player_colour(self) -> str:
        return self.env.current_turn_colour

    def observe(self, colour: str):
        return self.env.observe(colour)

    def step(self, colour: str, action):
        step_result = self.env.step(colour, action)
        return step_result.reward, step_result.done, step_result.info

    def value_for_colour(self, colour: str) -> float:
        """
        Multiplayer-safe terminal/search value from the requested player's perspective.
        """
        game = self.env.game
        if game is None:
            return 0.0

        state = game.to_public_state()
        me = next((p for p in state["players"] if p["colour"] == colour), None)

        if me is not None:
            if me["status"] == "WIN":
                return 1.0
            if me["status"] == "DRAW":
                return 0.0
            if me["status"] == "LOSS":
                return -1.0

        return game.normalized_training_value(colour)

def mcts_label_for_observation(learner: TrainableAgent,
                               env: ChineseCheckersEnv,
                               colour: str,
                               num_simulations: int = 64,
                               allow_heuristic_fallback: bool = False):
    """
    Run MCTS from the current environment state and return:
      - chosen action
      - graph_state
      - target_policy distribution aligned with graph_state.legal_actions
      - metadata
    """
    search_env = LocalSearchEnv(copy.deepcopy(env))

    mcts = NeuralMCTS(model=learner.model,
                      graph_builder=learner.policy.graph_builder,
                      device=learner.device,
                      num_simulations=num_simulations)

    try:
        best_action, actions, probs = mcts.search(search_env)
        mcts_failed = False

    except Exception as e:
        if not allow_heuristic_fallback:
            raise RuntimeError(f"MCTS failed for colour={colour}, move_count={env.game.move_count if env.game else 'NA'}") from e

        obs = env.observe(colour)
        heuristic = HeuristicPolicy(epsilon=0.0)
        best_action = heuristic.select_action(obs)
        gs = learner.graph_from_observation(obs)

        target_policy = torch.zeros(len(gs.legal_actions), dtype=torch.float32)
        for i, (pin_id, _, to_idx) in enumerate(gs.legal_actions):
            if (pin_id, to_idx) == best_action:
                target_policy[i] = 1.0
                break

        return best_action, gs, target_policy, {"mcts_failed": True}

    obs = env.observe(colour)
    gs = learner.graph_from_observation(obs)

    action_to_prob = {tuple(a): float(p) for a, p in zip(actions, probs.tolist())}

    target_policy = []
    for pin_id, _, to_idx in gs.legal_actions:
        target_policy.append(action_to_prob.get((pin_id, to_idx), 0.0))

    target_policy = torch.tensor(target_policy, dtype=torch.float32)
    if target_policy.sum() > 0:
        target_policy = target_policy / target_policy.sum()
    else:
        target_policy = torch.ones(len(gs.legal_actions), dtype=torch.float32)
        target_policy = target_policy / target_policy.sum()

    return best_action, gs, target_policy, {"mcts_failed": mcts_failed}

def sample_action_from_target_policy(graph_state: GraphState,
                                     target_policy: torch.Tensor,
                                     best_action: tuple[int, int],
                                     move_count: int,
                                     sample_until_move: int,
                                     temperature: float,
                                     late_temperature: float) -> tuple[tuple[int, int], bool]:
    """
    Pick the played training action from the MCTS visit distribution.

    Early/mid-game uses temperature sampling for exploration. Late game falls
    back to deterministic MCTS argmax so finishing behavior stays sharp.
    """
    if move_count >= sample_until_move:
        return best_action, False

    temp = temperature if move_count < sample_until_move // 2 else late_temperature
    if temp <= 0.0 or target_policy.numel() == 0:
        return best_action, False

    probs = target_policy.detach().float().cpu().clamp(min=0.0)
    if probs.sum() <= 0:
        return best_action, False

    if temp != 1.0:
        probs = probs.pow(1.0 / temp)
        if probs.sum() <= 0:
            return best_action, False

    probs = probs / probs.sum()
    action_idx = int(torch.multinomial(probs, num_samples=1).item())
    pin_id, _, to_idx = graph_state.legal_actions[action_idx]
    return (pin_id, to_idx), (pin_id, to_idx) != best_action

def terminal_value_from_result(game, final_state, colour: str, truncated: bool) -> float:
    """
    Multiplayer-safe value target from this colour's perspective.
    """
    me = next(p for p in final_state["players"] if p["colour"] == colour)

    if me["status"] == "WIN":
        return 1.0
    if me["status"] == "DRAW":
        return 0.0
    if me["status"] == "LOSS":
        return -1.0

    # If the game was truncated but not already adjudicated, use clean progress.
    if truncated:
        return game.normalized_training_value(colour)

    return -1.0

def policy_examples_from_game(per_colour_examples,
                              game,
                              final_state,
                              truncated: bool) -> list[dict]:
    """
    Attach target values after a game finishes or truncates.
    """
    out = []
    for colour, examples in per_colour_examples.items():
        target_value = terminal_value_from_result(game, final_state, colour, truncated=truncated)
        for ex in examples:
            ex["target_value"] = target_value
            out.append(ex)
    return out

def bootstrap_imitation(num_games: int = 500,
                        batch_size: int = 64,
                        checkpoint_every: int = 50,
                        checkpoint_dir: str = "checkpoints/bootstrap",
                        device: str = "cpu",
                        start_checkpoint: str | None = None,
                        max_moves_per_game: int = 500,
                        start_game_index: int = 0,
                        positions_per_cycle: int = 2500,
                        updates_per_cycle: int = 200,
                        num_players: int = 6,
                        use_mixed_rollout_prefix: bool = WARMSTART_USE_MIXED_ROLLOUT_PREFIX,
                        prefix_rollout_max_moves: int = WARMSTART_PREFIX_ROLLOUT_MAX_MOVES,
                        prefix_rollout_prob: float = WARMSTART_PREFIX_ROLLOUT_PROB,
                        collect_all_heuristic_seats: bool = WARMSTART_COLLECT_ALL_HEURISTIC_SEATS,
                        max_label_moves_after_prefix: int = WARMSTART_MAX_LABEL_MOVES_AFTER_PREFIX):
    """
    Warm-start using heuristic imitation on one tracked seat, but with mixed opponents.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    heuristic = HeuristicPolicy(epsilon=0.0)
    noisy_heuristic = HeuristicPolicy(epsilon=4.0)
    learner = TrainableAgent(name="shared_model",
                            device=device,
                            hidden_dim=MODEL_HIDDEN_DIM,
                            num_layers=MODEL_NUM_LAYERS)
    
    buffer = ReplayBuffer(capacity=100000)

    maybe_load_into_learner(learner, start_checkpoint)
    positions_since_update = 0

    pbar = tqdm(range(1, num_games + 1), desc="bootstrap", unit="game", leave=True)
    for local_game_idx in pbar:
        game_idx = start_game_index + local_game_idx
        num_players = random.choices(TRAIN_PLAYER_COUNTS, weights=TRAIN_PLAYER_COUNT_WEIGHTS, k=1)[0]
        env = ChineseCheckersEnv(num_players=num_players)
        env.reset()

        last_checkpoint_path = find_latest_checkpoint(BOOTSTRAP_CHECKPOINT_DIR)
        has_last_checkpoint = last_checkpoint_path is not None and os.path.exists(last_checkpoint_path)

        learner_colour, seat_plan = sample_warmstart_seat_plan(turn_order=env.turn_order,
                                                               has_last_checkpoint=has_last_checkpoint)
        policy_cache = build_warmstart_policy_cache(seat_plan=seat_plan,
                                                   heuristic=heuristic,
                                                   last_checkpoint_path=last_checkpoint_path,
                                                   device=device)

        rollout_info = {"prefix_moves": 0,
                        "prefix_truncated": False,
                        "prefix_kind_counts": {}}
        if use_mixed_rollout_prefix:
            rollout_info = run_warmstart_mixed_rollout_prefix(env,
                                                              max_prefix_moves=prefix_rollout_max_moves,
                                                              prefix_probability=prefix_rollout_prob,
                                                              heuristic=heuristic,
                                                              noisy_heuristic=noisy_heuristic,
                                                              last_checkpoint_path=last_checkpoint_path,
                                                              device=device)

        progress_print("bootstrap",
                       f"starting game={game_idx} num_players={num_players} "
                       f"learner_colour={learner_colour} seat_plan={seat_plan} "
                       f"prefix_moves={rollout_info['prefix_moves']} "
                       f"prefix_kinds={rollout_info['prefix_kind_counts']}")
        per_colour_examples = defaultdict(list)
        done = False
        game_start = time.time()
        label_moves_done = 0

        while (not done
               and env.game.status != "FINISHED"
               and env.game.move_count < max_moves_per_game
               and label_moves_done < max_label_moves_after_prefix):
            colour = env.current_turn_colour
            obs = env.observe(colour)

            chosen_action = policy_cache[colour].select_action(obs)
            should_collect_label = (
                colour == learner_colour
                or (collect_all_heuristic_seats and seat_plan[colour] in {"teacher_seat", "heuristic"})
            )
            if should_collect_label:
                add_heuristic_imitation_example(learner=learner,
                                                observation=obs,
                                                chosen_action=chosen_action,
                                                examples_by_colour=per_colour_examples)

            step_result = env.step(colour, chosen_action)
            done = step_result.done
            label_moves_done += 1

        truncated = env.game.move_count >= max_moves_per_game
        if truncated:
            if env.game.status != "FINISHED":
                env.game.adjudicate_by_progress("MAX_MOVES_REACHED")

        final_state = env.game.to_public_state()
        examples = policy_examples_from_game(per_colour_examples=per_colour_examples,
                                             game=env.game,
                                             final_state=final_state,
                                             truncated=truncated)

        for ex in examples:
            buffer.add(ex)
        positions_since_update += len(examples)

        avg_loss = None
        if positions_since_update >= positions_per_cycle and len(buffer) >= batch_size:
            losses = []
            for update_idx in range(updates_per_cycle):
                batch = buffer.sample(batch_size)
                loss = learner.train_policy_batch(batch)
                losses.append(loss)

                if (update_idx + 1) % 10 == 0 or (update_idx + 1) == updates_per_cycle:
                    running_avg = sum(losses) / len(losses)
                    tqdm_write(f"[bootstrap] update {update_idx+1}/{updates_per_cycle} "
                               f"loss={loss:.4f} avg={running_avg:.4f}")

            avg_loss = sum(losses) / len(losses)
            positions_since_update = 0

        elapsed = time.time() - game_start
        loss_str = f"{avg_loss:.4f}" if avg_loss is not None else "NA"
        print(f"[bootstrap] game={game_idx} moves={env.game.move_count} "
              f"prefix_moves={rollout_info['prefix_moves']} "
              f"label_moves={label_moves_done} "
              f"examples={len(examples)} buffer={len(buffer)} "
              f"positions_since_update={positions_since_update} "
              f"avg_loss={loss_str} truncated={truncated} time={elapsed:.2f}s")

        if game_idx % checkpoint_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"shared_model_{game_idx}.pt")
            tqdm_write(f"[bootstrap] saving checkpoint: {ckpt_path}")
            learner.save_full(ckpt_path)

    learner.save_full(os.path.join(checkpoint_dir, "shared_model_final.pt"))


def evaluate_warmstart_topk_match(checkpoint_path: str,
                                  eval_games: int = 20,
                                  max_positions: int = 1000,
                                  device: str = "cpu",
                                  k: int = 3,
                                  use_mixed_rollout_prefix: bool = WARMSTART_USE_MIXED_ROLLOUT_PREFIX,
                                  prefix_rollout_max_moves: int = WARMSTART_PREFIX_ROLLOUT_MAX_MOVES,
                                  prefix_rollout_prob: float = WARMSTART_PREFIX_ROLLOUT_PROB) -> dict:
    """
    Measures how often the model's chosen action is among the heuristic's top-k legal moves
    on states generated by heuristic play.

    Uses ADAPTIVE k:
      - many pieces outside target  -> wider top-k
      - few pieces outside target   -> narrower top-k
      - endgame (1-2 outside)       -> exact best move required
    """
    heuristic = HeuristicPolicy(epsilon=0.0)
    noisy_heuristic = HeuristicPolicy(epsilon=4.0)

    model = load_model(checkpoint_path, device=device)
    learner = MyPolicy(model=model, device=device)

    matches = 0
    exact_matches = 0
    fixed_top3_matches = 0
    total = 0
    rank_sum = 0.0
    legal_action_sum = 0

    adaptive_breakdown = {"k1": 0,
                          "k3": 0,
                          "k5": 0}
    prefix_moves_total = 0

    for _ in range(eval_games):
        num_players = random.choices(TRAIN_PLAYER_COUNTS, weights=TRAIN_PLAYER_COUNT_WEIGHTS, k=1)[0]
        env = ChineseCheckersEnv(num_players=num_players)
        env.reset()
        if use_mixed_rollout_prefix:
            rollout_info = run_warmstart_mixed_rollout_prefix(env,
                                                              max_prefix_moves=prefix_rollout_max_moves,
                                                              prefix_probability=prefix_rollout_prob,
                                                              heuristic=heuristic,
                                                              noisy_heuristic=noisy_heuristic,
                                                              last_checkpoint_path=checkpoint_path,
                                                              device=device)
            prefix_moves_total += rollout_info["prefix_moves"]

        done = False

        while not done and total < max_positions:
            colour = env.current_turn_colour
            obs = env.observe(colour)

            state = obs["state"]
            legal_moves = obs["legal_moves"]
            my_positions = state["pins"][colour]

            outside_count = heuristic._count_pieces_outside_target(colour, state)

            # Adaptive top-k window
            if outside_count <= 2:
                effective_k = 1
                adaptive_breakdown["k1"] += 1
            elif outside_count <= 5:
                effective_k = 3
                adaptive_breakdown["k3"] += 1
            else:
                effective_k = 5
                adaptive_breakdown["k5"] += 1

            scored_actions = []
            for pin_id, to_list in legal_moves.items():
                pid = int(pin_id)
                from_idx = my_positions[pid]
                for to_idx in to_list:
                    s = heuristic._score_action(colour, from_idx, int(to_idx), state)
                    scored_actions.append(((pid, int(to_idx)), s))

            if not scored_actions:
                step_result = env.step(colour, heuristic.select_action(obs))
                done = step_result.done
                continue

            scored_actions.sort(key=lambda x: x[1], reverse=True)

            topk_actions = {action for action, _ in scored_actions[:effective_k]}
            fixed_top3_actions = {action for action, _ in scored_actions[:3]}
            best_action = scored_actions[0][0]
            model_action = learner.select_action(obs)

            if tuple(model_action) in topk_actions:
                matches += 1
            if tuple(model_action) == best_action:
                exact_matches += 1
            if tuple(model_action) in fixed_top3_actions:
                fixed_top3_matches += 1

            ranked_actions = [action for action, _ in scored_actions]
            if tuple(model_action) in ranked_actions:
                rank_sum += ranked_actions.index(tuple(model_action)) + 1
            else:
                rank_sum += len(ranked_actions) + 1
            legal_action_sum += len(ranked_actions)

            total += 1

            step_result = env.step(colour, heuristic.select_action(obs))
            done = step_result.done

        if total >= max_positions:
            break

    match_rate = matches / total if total > 0 else 0.0
    return {"topk_match_rate": match_rate,
            "exact_match_rate": exact_matches / total if total > 0 else 0.0,
            "fixed_top3_match_rate": fixed_top3_matches / total if total > 0 else 0.0,
            "avg_heuristic_rank": rank_sum / total if total > 0 else 0.0,
            "avg_legal_actions": legal_action_sum / total if total > 0 else 0.0,
            "matches": matches,
            "exact_matches": exact_matches,
            "fixed_top3_matches": fixed_top3_matches,
            "total": total,
            "avg_prefix_moves": prefix_moves_total / eval_games if eval_games > 0 else 0.0,
            "adaptive_breakdown": adaptive_breakdown}


def warmstart_until_good_enough(target_action_match: float = 0.80,
                                bootstrap_chunk_games: int = 50,
                                max_bootstrap_games: int = 2000,
                                batch_size: int = 64,
                                checkpoint_every: int = 25,
                                checkpoint_dir: str = "checkpoints/bootstrap",
                                device: str = "cpu",
                                max_moves_per_game: int = 500,
                                use_mixed_rollout_prefix: bool = WARMSTART_USE_MIXED_ROLLOUT_PREFIX,
                                prefix_rollout_max_moves: int = WARMSTART_PREFIX_ROLLOUT_MAX_MOVES,
                                prefix_rollout_prob: float = WARMSTART_PREFIX_ROLLOUT_PROB,
                                collect_all_heuristic_seats: bool = WARMSTART_COLLECT_ALL_HEURISTIC_SEATS,
                                max_label_moves_after_prefix: int = WARMSTART_MAX_LABEL_MOVES_AFTER_PREFIX,
                                ) -> tuple[str, dict]:
    """
    Repeatedly runs bootstrap imitation in chunks, evaluating after each chunk,
    until the model matches the heuristic often enough.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    total_games = 0
    final_ckpt = os.path.join(checkpoint_dir, "shared_model_final.pt")
    max_moves_per_game=max_moves_per_game

    while total_games < max_bootstrap_games:
        print(f"\nRunning bootstrap chunk: {bootstrap_chunk_games} games")
        current_checkpoint = final_ckpt if os.path.exists(final_ckpt) else None

        bootstrap_imitation(num_games=bootstrap_chunk_games,
                            batch_size=batch_size,
                            checkpoint_every=checkpoint_every,
                            checkpoint_dir=checkpoint_dir,
                            device=device,
                            max_moves_per_game=max_moves_per_game,
                            start_checkpoint=current_checkpoint,
                            start_game_index=total_games,
                            use_mixed_rollout_prefix=use_mixed_rollout_prefix,
                            prefix_rollout_max_moves=prefix_rollout_max_moves,
                            prefix_rollout_prob=prefix_rollout_prob,
                            collect_all_heuristic_seats=collect_all_heuristic_seats,
                            max_label_moves_after_prefix=max_label_moves_after_prefix)
        
        total_games += bootstrap_chunk_games

        if not os.path.exists(final_ckpt):
            raise FileNotFoundError(f"Bootstrap checkpoint missing: {final_ckpt}")

        action_eval = evaluate_warmstart_topk_match(
            checkpoint_path=final_ckpt,
            eval_games=20,
            max_positions=1000,
            device=device,
            k=3,
            use_mixed_rollout_prefix=use_mixed_rollout_prefix,
            prefix_rollout_max_moves=prefix_rollout_max_moves,
            prefix_rollout_prob=prefix_rollout_prob)

        print(f"[warmstart eval] games={total_games} "
              f"adaptive_match_rate={action_eval['topk_match_rate']:.3f} "
              f"({action_eval['matches']}/{action_eval['total']}) "
              f"exact={action_eval['exact_match_rate']:.3f} "
              f"top3={action_eval['fixed_top3_match_rate']:.3f} "
              f"avg_rank={action_eval['avg_heuristic_rank']:.2f} "
              f"avg_legal={action_eval['avg_legal_actions']:.1f} "
              f"avg_prefix={action_eval['avg_prefix_moves']:.1f} "
              f"k1={action_eval['adaptive_breakdown']['k1']} "
              f"k3={action_eval['adaptive_breakdown']['k3']} "
              f"k5={action_eval['adaptive_breakdown']['k5']}")

        if action_eval["topk_match_rate"] >= target_action_match:
            print("Warm-start threshold reached.")
            return final_ckpt, action_eval

    print("Reached maximum bootstrap games without hitting target threshold.")
    return final_ckpt, action_eval


def self_play_refinement(start_checkpoint: str | None = None,
                         num_games: int = 2000,
                         batch_size: int = 64,
                         checkpoint_every: int = 100,
                         checkpoint_dir: str = "checkpoints/self_play",
                         device: str = "cpu",
                         max_moves_per_game: int = 300,
                         start_game_index: int = 0,
                         positions_per_cycle: int = 500,
                         updates_per_cycle: int = 50,
                         value_weight: float = 0.5,
                         num_players: int = 6,
                         mcts_simulations: int = 128,
                         collect_all_current_model_mcts_seats: bool = COLLECT_ALL_CURRENT_MODEL_MCTS_SEATS,
                         mcts_sample_until_move: int = MCTS_TRAIN_SAMPLE_UNTIL_MOVE,
                         mcts_temperature: float = MCTS_TRAIN_TEMPERATURE,
                         mcts_late_temperature: float = MCTS_TRAIN_LATE_TEMPERATURE):
    """
    MCTS-labeled self-play:
    - policy targets come from MCTS visit distributions
    - value targets come from final outcome / normalized truncated score
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    learner = TrainableAgent(name="shared_model",
                             device=device,
                             hidden_dim=MODEL_HIDDEN_DIM,
                             num_layers=MODEL_NUM_LAYERS)
    heuristic = HeuristicPolicy(epsilon=0.0)
    buffer = ReplayBuffer(capacity=200000)

    maybe_load_into_learner(learner, start_checkpoint)

    positions_since_update = 0
    total_updates_run = 0

    run_metrics = {"games": 0,
                   "truncated": 0,
                   "resigned": 0,
                   "adjudicated": 0,
                   "stall_adjudicated": 0,
                   "stranded_home_adjudicated": 0,
                   "repetition_adjudicated": 0,
                   "max_moves_adjudicated": 0,
                   "mcts_calls": 0,
                   "mcts_failures": 0,
                   "mcts_sampled_non_argmax": 0,
                   "mcts_trained_seat_moves": 0,
                   "total_moves": 0,
                   "total_examples": 0,
                   "total_stranded_home": 0.0,
                   "max_stranded_home": 0,
                   "updates": 0,
                   "loss_total_sum": 0.0,
                   "loss_policy_sum": 0.0,
                   "loss_value_sum": 0.0}

    warmstart_final_path = os.path.join(BOOTSTRAP_CHECKPOINT_DIR, "shared_model_final.pt")
    champion_path = os.path.join(checkpoint_dir, CHAMPION_NAME)

    pbar = tqdm(range(1, num_games + 1), desc="self-play", unit="game", leave=True)
    for local_game_idx in pbar:
        game_idx = start_game_index + local_game_idx
        num_players = random.choices(TRAIN_PLAYER_COUNTS,
                                     weights=TRAIN_PLAYER_COUNT_WEIGHTS,
                                     k=1)[0]

        env = ChineseCheckersEnv(num_players=num_players)
        env.reset()

        random_checkpoint_path = sample_random_checkpoint(checkpoint_dirs=[BOOTSTRAP_CHECKPOINT_DIR, SELFPLAY_CHECKPOINT_DIR],
                                                          exclude={champion_path} if os.path.exists(champion_path) else set())
        learner_colour, seat_plan = sample_selfplay_seat_plan(turn_order=env.turn_order,
                                                              has_champion=os.path.exists(champion_path),
                                                              has_warmstart_final=os.path.exists(warmstart_final_path),
                                                              has_random_checkpoint=random_checkpoint_path is not None)
        policy_cache = build_selfplay_policy_cache(seat_plan=seat_plan,
                                                   learner=learner,
                                                   heuristic=heuristic,
                                                   champion_path=champion_path if os.path.exists(champion_path) else None,
                                                   warmstart_final_path=warmstart_final_path if os.path.exists(warmstart_final_path) else None,
                                                   random_checkpoint_path=random_checkpoint_path,
                                                   device=device)

        progress_print("self-play",
                       f"starting game={game_idx} num_players={num_players} "
                       f"learner_colour={learner_colour} seat_plan={seat_plan} "
                       f"mcts_simulations={mcts_simulations}")

        per_colour_examples = defaultdict(list)
        done = False
        resigned = False
        game_start = time.time()
        printed_first_mcts = False

        while not done and env.game.move_count < max_moves_per_game:
            colour = env.current_turn_colour
            obs = env.observe(colour)

            seat_kind = seat_plan[colour]
            should_collect_mcts = (
                seat_kind == "tracked_current"
                or (collect_all_current_model_mcts_seats and seat_kind == "current_model")
            )

            if should_collect_mcts:
                if not printed_first_mcts:
                    progress_print("self-play",
                                   f"first MCTS call game={game_idx} move={env.game.move_count} "
                                   f"colour={colour} seat_kind={seat_kind}")
                    printed_first_mcts = True

                best_action, gs, target_policy, mcts_meta = mcts_label_for_observation(learner=learner,
                                                                                       env=env,
                                                                                       colour=colour,
                                                                                       num_simulations=mcts_simulations,
                                                                                       allow_heuristic_fallback=False)

                chosen_action, sampled_non_argmax = sample_action_from_target_policy(graph_state=gs,
                                                                                     target_policy=target_policy,
                                                                                     best_action=best_action,
                                                                                     move_count=env.game.move_count,
                                                                                     sample_until_move=mcts_sample_until_move,
                                                                                     temperature=mcts_temperature,
                                                                                     late_temperature=mcts_late_temperature)

                per_colour_examples[colour].append({"graph_state": gs,
                                                    "target_policy": target_policy,
                                                    "mcts_failed": mcts_meta["mcts_failed"],
                                                    "seat_kind": seat_kind,
                                                    "sampled_non_argmax": sampled_non_argmax})
                run_metrics["mcts_calls"] += 1
                run_metrics["mcts_failures"] += int(mcts_meta["mcts_failed"])
                run_metrics["mcts_sampled_non_argmax"] += int(sampled_non_argmax)
                run_metrics["mcts_trained_seat_moves"] += 1
            else:
                chosen_action = policy_cache[colour].select_action(obs)

            step_result = env.step(colour, chosen_action)
            done = step_result.done

        truncated = (not resigned) and env.game.move_count >= max_moves_per_game
        if truncated or resigned:
            if env.game.status != "FINISHED":
                env.game.adjudicate_by_progress("MAX_MOVES_REACHED")

        final_state = env.game.to_public_state()

        adjudication_reason = get_adjudication_reason_from_state(final_state)
        adjudication_type = classify_adjudication_reason(adjudication_reason)

        run_metrics["games"] += 1
        run_metrics["truncated"] += int(truncated)
        run_metrics["resigned"] += int(resigned)
        run_metrics["total_moves"] += env.game.move_count

        if adjudication_type != "none":
            run_metrics["adjudicated"] += 1
        if adjudication_type == "stall":
            run_metrics["stall_adjudicated"] += 1
        elif adjudication_type == "stranded_home":
            run_metrics["stranded_home_adjudicated"] += 1
        elif adjudication_type == "repetition":
            run_metrics["repetition_adjudicated"] += 1
        elif adjudication_type == "max_moves":
            run_metrics["max_moves_adjudicated"] += 1

        stranded_map = final_state.get("stranded_home_pieces", {})
        if isinstance(stranded_map, dict) and stranded_map:
            avg_stranded_this_game = sum(stranded_map.values()) / len(stranded_map)
            max_stranded_this_game = max(stranded_map.values())
        else:
            avg_stranded_this_game = 0.0
            max_stranded_this_game = 0

        run_metrics["total_stranded_home"] += avg_stranded_this_game
        run_metrics["max_stranded_home"] = max(run_metrics["max_stranded_home"], max_stranded_this_game)
        # Compute discounted returns backward through each color's move sequence.
        # Resigned learner gets -1.0 as terminal value regardless of progress.
        # Target values are clipped to [-1, 1] to stay within the tanh value head's range.
        # Without clipping, accumulated stranded_home_penalty step rewards (e.g. -0.6/move
        # over 36 moves with GAMMA=0.99) produce targets as extreme as -20, making MSE ~200.
        examples = []
        for colour, exs in per_colour_examples.items():
            if colour == learner_colour and resigned:
                terminal_val = -1.0
            else:
                terminal_val = terminal_value_from_result(env.game, final_state, colour, truncated=truncated)
            g = terminal_val
            for ex in reversed(exs):
                g = ex.pop("step_reward", 0.0) + GAMMA * g
                ex["target_value"] = float(max(-1.0, min(1.0, g)))
            examples.extend(exs)

        for ex in examples:
            buffer.add(ex)
        run_metrics["total_examples"] += len(examples)

        positions_since_update += len(examples)

        avg_loss = None
        updates_this_game = 0

        if positions_since_update >= positions_per_cycle and len(buffer) >= batch_size:
            losses = []
            for update_idx in range(updates_per_cycle):
                # Prioritized sampling: returns batch + storage indices
                batch, sample_indices = buffer.sample_with_idx(batch_size)
                loss_metrics = learner.train_policy_value_batch_soft(
                    batch,
                    value_weight=value_weight,
                    return_metrics=True,
                    return_per_sample_errors=True)

                # Update replay priorities with per-sample total loss as proxy TD error
                buffer.update_priorities(sample_indices, loss_metrics["per_sample_errors"])

                loss = loss_metrics["total_loss"]
                losses.append(loss)

                run_metrics["updates"] += 1
                run_metrics["loss_total_sum"] += loss_metrics["total_loss"]
                run_metrics["loss_policy_sum"] += loss_metrics["policy_loss"]
                run_metrics["loss_value_sum"] += loss_metrics["value_loss"]

                updates_this_game += 1
                total_updates_run += 1

                if (update_idx + 1) % 10 == 0 or (update_idx + 1) == updates_per_cycle:
                    running_avg = sum(losses) / len(losses)
                    tqdm_write(f"[self-play] update {update_idx+1}/{updates_per_cycle} "
                               f"loss={loss:.4f} avg={running_avg:.4f} "
                               f"policy={loss_metrics['policy_loss']:.4f} "
                               f"value={loss_metrics['value_loss']:.4f}")

            avg_loss = sum(losses) / len(losses)
            positions_since_update = 0

        elapsed = time.time() - game_start
        loss_str = f"{avg_loss:.4f}" if avg_loss is not None else "NA"

        mcts_failure_rate_so_far = safe_rate(run_metrics["mcts_failures"], run_metrics["mcts_calls"])

        print(f"[self-play] game={game_idx} moves={env.game.move_count} "
              f"examples={len(examples)} buffer={len(buffer)} "
              f"positions_since_update={positions_since_update} "
              f"updates_this_game={updates_this_game} total_updates={total_updates_run} "
              f"loss={loss_str} truncated={truncated} "
              f"adj={adjudication_type} "
              f"mcts_fail_rate={mcts_failure_rate_so_far:.3f} "
              f"mcts_sampled={run_metrics['mcts_sampled_non_argmax']} "
              f"avg_stranded_this_game={avg_stranded_this_game:.2f} "
              f"max_stranded_this_game={max_stranded_this_game} "
              f"time={elapsed:.2f}s")

        if game_idx % checkpoint_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"shared_model_{game_idx}.pt")
            tqdm_write(f"[self-play] saving checkpoint: {ckpt_path}")
            learner.save_full(ckpt_path)

    games = run_metrics["games"]
    updates = run_metrics["updates"]

    print("\n" + "=" * 80)
    print("SELF-PLAY RUN SUMMARY")
    print("=" * 80)
    print(f"games                      : {games}")
    print(f"avg_moves                  : {safe_rate(run_metrics['total_moves'], games):.2f}")
    print(f"total_examples             : {run_metrics['total_examples']}")
    print(f"avg_examples_per_game      : {safe_rate(run_metrics['total_examples'], games):.2f}")
    print(f"truncation_rate            : {safe_rate(run_metrics['truncated'], games):.3f}")
    print(f"resign_rate                : {safe_rate(run_metrics['resigned'], games):.3f}")
    print(f"adjudication_rate          : {safe_rate(run_metrics['adjudicated'], games):.3f}")
    print(f"max_moves_adj_rate         : {safe_rate(run_metrics['max_moves_adjudicated'], games):.3f}")
    print(f"stall_adj_rate             : {safe_rate(run_metrics['stall_adjudicated'], games):.3f}")
    print(f"stranded_home_adj_rate     : {safe_rate(run_metrics['stranded_home_adjudicated'], games):.3f}")
    print(f"repetition_adj_rate        : {safe_rate(run_metrics['repetition_adjudicated'], games):.3f}")
    print(f"mcts_calls                 : {run_metrics['mcts_calls']}")
    print(f"mcts_failures              : {run_metrics['mcts_failures']}")
    print(f"mcts_failure_rate          : {safe_rate(run_metrics['mcts_failures'], run_metrics['mcts_calls']):.3f}")
    print(f"mcts_trained_seat_moves    : {run_metrics['mcts_trained_seat_moves']}")
    print(f"mcts_sampled_non_argmax    : {run_metrics['mcts_sampled_non_argmax']}")
    print(f"mcts_sampled_non_argmax_rate: {safe_rate(run_metrics['mcts_sampled_non_argmax'], run_metrics['mcts_trained_seat_moves']):.3f}")
    print(f"avg_stranded_home          : {safe_rate(run_metrics['total_stranded_home'], games):.3f}")
    print(f"max_stranded_home          : {run_metrics['max_stranded_home']}")

    if updates > 0:
        print(f"avg_total_loss             : {run_metrics['loss_total_sum'] / updates:.4f}")
        print(f"avg_policy_loss            : {run_metrics['loss_policy_sum'] / updates:.4f}")
        print(f"avg_value_loss             : {run_metrics['loss_value_sum'] / updates:.4f}")
    else:
        print("avg_total_loss             : NA")
        print("avg_policy_loss            : NA")
        print("avg_value_loss             : NA")

    print("=" * 80)

    learner.save_full(os.path.join(checkpoint_dir, "shared_model_final.pt"))

def self_play_with_promotion(start_checkpoint: str,
                             total_blocks: int = 20,
                             games_per_block: int = 100,
                             batch_size: int = 64,
                             checkpoint_dir: str = "checkpoints/self_play",
                             device: str = "cpu",
                             max_moves_per_game: int = 300,
                             positions_per_cycle: int = 2500,
                             updates_per_cycle: int = 150,
                             value_weight: float = 0.5,
                             mcts_simulations: int = 128,
                             collect_all_current_model_mcts_seats: bool = COLLECT_ALL_CURRENT_MODEL_MCTS_SEATS,
                             mcts_sample_until_move: int = MCTS_TRAIN_SAMPLE_UNTIL_MOVE,
                             mcts_temperature: float = MCTS_TRAIN_TEMPERATURE,
                             mcts_late_temperature: float = MCTS_TRAIN_LATE_TEMPERATURE):
    os.makedirs(checkpoint_dir, exist_ok=True)

    champion_ckpt = os.path.join(checkpoint_dir, CHAMPION_NAME)
    if not os.path.exists(champion_ckpt):
        copy_checkpoint_with_optimizer(start_checkpoint, champion_ckpt)
        print(f"[promotion] initialized champion from {start_checkpoint}")

    # Resume from persisted block state if available
    block_state = load_block_state(checkpoint_dir)
    if block_state is not None:
        resume_block = block_state["block_idx"] + 1
        total_games_done = block_state["total_games_done"]
        current_start_ckpt = block_state["current_start_ckpt"]
        if not os.path.exists(current_start_ckpt):
            print(f"[promotion] resume ckpt missing ({current_start_ckpt}), falling back to champion")
            current_start_ckpt = champion_ckpt
        print(f"[promotion] resuming from block {resume_block} (games done: {total_games_done})")
    else:
        resume_block = 1
        total_games_done = 0
        current_start_ckpt = champion_ckpt

    for block_idx in range(resume_block, total_blocks + 1):
        print("\n" + "#" * 80)
        print(f"[promotion] STARTING BLOCK {block_idx}/{total_blocks}")
        print(f"[promotion] actual champion = {champion_ckpt}")
        print(f"[promotion] training starts from = {current_start_ckpt}")
        print("#" * 80)

        challenger_ckpt = os.path.join(checkpoint_dir, f"challenger_block_{block_idx}.pt")

        # Train one block from champion
        self_play_refinement(start_checkpoint=current_start_ckpt,
                             num_games=games_per_block,
                             batch_size=batch_size,
                             checkpoint_every=games_per_block,  # save once at block end
                             checkpoint_dir=checkpoint_dir,
                             device=device,
                             max_moves_per_game=max_moves_per_game,
                             start_game_index=total_games_done,
                             positions_per_cycle=positions_per_cycle,
                             updates_per_cycle=updates_per_cycle,
                             value_weight=value_weight,
                             mcts_simulations=mcts_simulations,
                             collect_all_current_model_mcts_seats=collect_all_current_model_mcts_seats,
                             mcts_sample_until_move=mcts_sample_until_move,
                             mcts_temperature=mcts_temperature,
                             mcts_late_temperature=mcts_late_temperature)

        # self_play_refinement writes shared_model_final.pt
        produced_ckpt = os.path.join(checkpoint_dir, "shared_model_final.pt")
        if not os.path.exists(produced_ckpt):
            raise FileNotFoundError(f"Expected challenger checkpoint missing: {produced_ckpt}")

        copy_checkpoint_with_optimizer(produced_ckpt, challenger_ckpt)
        print(f"[promotion] challenger saved to {challenger_ckpt}")

        # Evaluate challenger against champion
        report = evaluate_checkpoint_promotion(champion_ckpt=champion_ckpt,
                                               challenger_ckpt=challenger_ckpt,
                                               device=device,
                                               matches_per_player_count=PROMOTION_MATCHES,
                                               player_counts=PROMOTION_PLAYER_COUNTS,
                                               max_moves=PROMOTION_MAX_MOVES,
                                               use_league_evaluation=PROMOTION_USE_LEAGUE_EVALUATION)
        print_promotion_report(report)

        report_path = os.path.join(checkpoint_dir, "promotion_metrics.jsonl")
        append_jsonl(report_path, {"block_idx": block_idx,
                                    "champion_ckpt": champion_ckpt,
                                    "challenger_ckpt": challenger_ckpt,
                                    "promoted": report["promoted"],
                                    "overall": report.get("overall", {}),
                                    "by_player_count": report.get("by_player_count", {})})
        print(f"[promotion] metrics written to {report_path}")

        # Promote, continue, or reset
        if report["promoted"]:
            copy_checkpoint_with_optimizer(challenger_ckpt, champion_ckpt)
            current_start_ckpt = champion_ckpt
            print("[promotion] challenger PROMOTED -> new champion")
            print(f"[promotion] next block will start from champion: {current_start_ckpt}")

        else:
            collapsed, collapse_reasons = challenger_collapsed(report)

            if CONTINUE_REJECTED_CHALLENGER and not collapsed:
                current_start_ckpt = challenger_ckpt
                print("[promotion] challenger REJECTED narrowly -> continuing challenger training")
                print(f"[promotion] next block will start from rejected challenger: {current_start_ckpt}")

            else:
                current_start_ckpt = champion_ckpt
                print("[promotion] challenger REJECTED badly -> resetting to champion")
                print(f"[promotion] next block will start from champion: {current_start_ckpt}")

                if collapse_reasons:
                    print("[promotion] reset reasons:")
                    for reason in collapse_reasons:
                        print(f"  - {reason}")

        total_games_done += games_per_block

        # Persist block state so training can resume after a crash
        save_block_state(checkpoint_dir,
                         block_idx=block_idx,
                         total_games_done=total_games_done,
                         current_start_ckpt=current_start_ckpt)

    print("\n[promotion] training complete.")
    print(f"[promotion] final champion: {champion_ckpt}")

def maybe_load_into_learner(learner: TrainableAgent, checkpoint_path: str | None):
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        learner.load_full(checkpoint_path)
        print(f"[resume] loaded weights + optimizer from {checkpoint_path}", flush=True)


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    if not os.path.isdir(checkpoint_dir):
        return None

    candidates = []
    for name in os.listdir(checkpoint_dir):
        if name.startswith("shared_model_") and name.endswith(".pt"):
            stem = name[len("shared_model_"):-3]
            if stem.isdigit():
                candidates.append((int(stem), os.path.join(checkpoint_dir, name)))

    if not candidates:
        final_path = os.path.join(checkpoint_dir, "shared_model_final.pt")
        return final_path if os.path.exists(final_path) else None

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def save_block_state(checkpoint_dir: str, block_idx: int, total_games_done: int, current_start_ckpt: str):
    """Persist block index so self_play_with_promotion can resume after a crash."""
    state = {"block_idx": block_idx,
             "total_games_done": total_games_done,
             "current_start_ckpt": current_start_ckpt}
    path = os.path.join(checkpoint_dir, "block_state.json")
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def load_block_state(checkpoint_dir: str) -> dict | None:
    path = os.path.join(checkpoint_dir, "block_state.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def copy_checkpoint_with_optimizer(src: str, dst: str):
    """Copy a .pt checkpoint and its accompanying .pt.train optimizer state if present."""
    shutil.copyfile(src, dst)
    src_train = src + ".train"
    if os.path.exists(src_train):
        shutil.copyfile(src_train, dst + ".train")


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    RUN_WARMSTART = False
    RUN_SELF_PLAY_PROMOTION = True

    WARMSTART_DIR = BOOTSTRAP_CHECKPOINT_DIR
    SELFPLAY_DIR = SELFPLAY_CHECKPOINT_DIR

    WARMSTART_CHUNK_GAMES = 50
    WARMSTART_MAX_GAMES = 10000
    WARMSTART_BATCH_SIZE = 64
    WARMSTART_CHECKPOINT_EVERY = 25
    WARMSTART_MAX_MOVES = 300
    WARMSTART_TARGET_ACTION_MATCH = 0.90
    WARMSTART_POSITION_NOISE = True
    BOOTSTRAP_FROM_MIXED_ROLLOUTS = True
    WARMSTART_USE_MIXED_ROLLOUT_PREFIX = WARMSTART_POSITION_NOISE and BOOTSTRAP_FROM_MIXED_ROLLOUTS
    WARMSTART_PREFIX_ROLLOUT_MAX_MOVES = 80
    WARMSTART_PREFIX_ROLLOUT_PROB = 0.65
    WARMSTART_COLLECT_ALL_HEURISTIC_SEATS = True
    WARMSTART_MAX_LABEL_MOVES_AFTER_PREFIX = 220

    SELFPLAY_TOTAL_BLOCKS = 100
    SELFPLAY_GAMES_PER_BLOCK = 100
    SELFPLAY_BATCH_SIZE = 64
    SELFPLAY_MAX_MOVES = 300
    SELFPLAY_POSITIONS_PER_CYCLE = 300
    SELFPLAY_UPDATES_PER_CYCLE = 100
    SELFPLAY_VALUE_WEIGHT = 0.05
    SELFPLAY_MCTS_SIMULATIONS = 128
    SELFPLAY_COLLECT_ALL_CURRENT_MODEL_MCTS_SEATS = True
    SELFPLAY_MCTS_SAMPLE_UNTIL_MOVE = 120
    SELFPLAY_MCTS_TEMPERATURE = 1.0
    SELFPLAY_MCTS_LATE_TEMPERATURE = 0.25

    MANUAL_START_CHECKPOINT = os.path.join(SELFPLAY_DIR, "champion.pt")

    print(f"Architecture name      : {ARCH_NAME}", flush=True)
    print(f"Warm-start directory   : {WARMSTART_DIR}", flush=True)
    print(f"Self-play directory    : {SELFPLAY_DIR}", flush=True)

    start_checkpoint = MANUAL_START_CHECKPOINT

    if RUN_WARMSTART:
        final_ckpt, warmstart_info = warmstart_until_good_enough(target_action_match=WARMSTART_TARGET_ACTION_MATCH,
                                                                 bootstrap_chunk_games=WARMSTART_CHUNK_GAMES,
                                                                 max_bootstrap_games=WARMSTART_MAX_GAMES,
                                                                 batch_size=WARMSTART_BATCH_SIZE,
                                                                 checkpoint_every=WARMSTART_CHECKPOINT_EVERY,
                                                                 checkpoint_dir=WARMSTART_DIR,
                                                                 device=DEVICE,
                                                                 max_moves_per_game=WARMSTART_MAX_MOVES,
                                                                 use_mixed_rollout_prefix=WARMSTART_USE_MIXED_ROLLOUT_PREFIX,
                                                                 prefix_rollout_max_moves=WARMSTART_PREFIX_ROLLOUT_MAX_MOVES,
                                                                 prefix_rollout_prob=WARMSTART_PREFIX_ROLLOUT_PROB,
                                                                 collect_all_heuristic_seats=WARMSTART_COLLECT_ALL_HEURISTIC_SEATS,
                                                                 max_label_moves_after_prefix=WARMSTART_MAX_LABEL_MOVES_AFTER_PREFIX)
        start_checkpoint = final_ckpt
        print(f"Warm-start completed: {final_ckpt}", flush=True)

    if RUN_SELF_PLAY_PROMOTION:
        if start_checkpoint is None:
            raise ValueError("No starting checkpoint provided for self-play.")

        print(f"Using start checkpoint: {start_checkpoint}", flush=True)
        if not os.path.exists(start_checkpoint):
            raise FileNotFoundError(f"Start checkpoint not found: {start_checkpoint}")

        self_play_with_promotion(start_checkpoint=start_checkpoint,
                                 total_blocks=SELFPLAY_TOTAL_BLOCKS,
                                 games_per_block=SELFPLAY_GAMES_PER_BLOCK,
                                 batch_size=SELFPLAY_BATCH_SIZE,
                                 checkpoint_dir=SELFPLAY_DIR,
                                 device=DEVICE,
                                 max_moves_per_game=SELFPLAY_MAX_MOVES,
                                 positions_per_cycle=SELFPLAY_POSITIONS_PER_CYCLE,
                                 updates_per_cycle=SELFPLAY_UPDATES_PER_CYCLE,
                                 value_weight=SELFPLAY_VALUE_WEIGHT,
                                 mcts_simulations=SELFPLAY_MCTS_SIMULATIONS,
                                 collect_all_current_model_mcts_seats=SELFPLAY_COLLECT_ALL_CURRENT_MODEL_MCTS_SEATS,
                                 mcts_sample_until_move=SELFPLAY_MCTS_SAMPLE_UNTIL_MOVE,
                                 mcts_temperature=SELFPLAY_MCTS_TEMPERATURE,
                                 mcts_late_temperature=SELFPLAY_MCTS_LATE_TEMPERATURE)


if __name__ == "__main__":
    main()
