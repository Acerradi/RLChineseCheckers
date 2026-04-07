import random
from collections import deque
import os
import sys
import time
import torch
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, item):
        self.buffer.append(item)

    def sample(self, batch_size: int):
        n = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, n)

    def __len__(self):
        return len(self.buffer)


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)


if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from policy_template import build_model, save_model, load_model, GraphState, MyPolicy, HeuristicPolicy, axial_dist, NeuralMCTS

MODEL_HIDDEN_DIM = 128
MODEL_NUM_LAYERS = 4

ARCH_NAME = f"gnn_h{MODEL_HIDDEN_DIM}_l{MODEL_NUM_LAYERS}"

BASE_CHECKPOINT_DIR = os.path.join("checkpoints", ARCH_NAME)
BOOTSTRAP_CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, "bootstrap")
SELFPLAY_CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, "self_play")

TRAIN_PLAYER_COUNTS = [2, 3, 4, 5, 6]
TRAIN_PLAYER_COUNT_WEIGHTS = [0.30, 0.20, 0.15, 0.15, 0.20]

# ============================================================
# CHECKPOINT PROMOTION CONFIG
# ============================================================

PROMOTION_ENABLED = True

PROMOTION_BLOCK_GAMES = 100
PROMOTION_MATCHES = 20 #20
PROMOTION_PLAYER_COUNTS = [2, 3, 4, 5, 6]
PROMOTION_MAX_MOVES = 300

# Primary criterion
PROMOTION_MIN_WINRATE = 0.55

# Secondary criteria
PROMOTION_MAX_TRUNCATION_RATE = 0.15
PROMOTION_MAX_AVG_WIN_MOVES = 220.0
PROMOTION_MIN_PROGRESS_SCORE = 0.0

# Harsh truncated-game punishment for value targets
TRUNCATED_VALUE = -0.75

# Champion path
CHAMPION_NAME = "champion.pt"

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

    last_from = None
    last_to = None
    last_colour = None

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

        if last_colour == colour and last_from == to_idx and last_to == from_idx:
            undo_moves += 1

        last_from = from_idx
        last_to = to_idx
        last_colour = colour

    return {"forward_moves": forward_moves,
            "backward_moves": backward_moves,
            "jump_moves": jump_moves,
            "undo_moves": undo_moves,
            "progress_score": progress_score}

def choose_learner_colour(turn_order, rng: random.Random | None = None) -> str:
    rng = rng or random
    return rng.choice(list(turn_order))

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

    quality = action_quality_metrics(
        history=history,
        board=env.game.board,
        focus_colour=challenger_colour,
    )

    final_score = scores.get(challenger_colour, {}).get("final_score", 0.0)

    return {"challenger_colour": challenger_colour,
            "challenger_seat_index": challenger_seat_index,
            "status": my_status,
            "final_score": final_score,
            "move_count": state.get("move_count", 0),
            "truncated": truncated,
            "quality": quality}

def evaluate_checkpoint_promotion(champion_ckpt: str,
                                  challenger_ckpt: str,
                                  device: str = "cpu",
                                  matches_per_player_count: int = 20,
                                  player_counts=(2, 3, 4, 5, 6),
                                  max_moves: int = 300) -> dict:
    all_results = []

    for n_players in player_counts:
        print(f"[promotion] evaluating {n_players}-player matches...", flush=True)
        for i in range(matches_per_player_count):
            seat_index = i % n_players
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
    # Primary gate: 2-player only
    # -------------------------
    two_player_results = [r for r in all_results if r["num_players"] == 2]

    two_wins = sum(r["status"] == "WIN" for r in two_player_results)
    two_draws = sum(r["status"] == "DRAW" for r in two_player_results)
    two_losses = sum(r["status"] not in ("WIN", "DRAW") for r in two_player_results)
    two_win_rate = two_wins / len(two_player_results) if two_player_results else 0.0

    # -------------------------
    # Secondary robustness: 3-6 players
    # -------------------------
    multi_results = [r for r in all_results if r["num_players"] >= 3]

    truncations = sum(r["truncated"] for r in all_results)
    truncation_rate = truncations / len(all_results) if all_results else 0.0

    win_move_counts = [r["move_count"] for r in all_results if r["status"] == "WIN"]
    avg_win_moves = average_or_zero(win_move_counts)

    avg_final_score = average_or_zero([r["final_score"] for r in all_results])
    avg_progress_score = average_or_zero([r["quality"]["progress_score"] for r in all_results])
    avg_backward_moves = average_or_zero([r["quality"]["backward_moves"] for r in all_results])
    avg_undo_moves = average_or_zero([r["quality"]["undo_moves"] for r in all_results])
    avg_jump_moves = average_or_zero([r["quality"]["jump_moves"] for r in all_results])

    wins = sum(r["status"] == "WIN" for r in all_results)
    draws = sum(r["status"] == "DRAW" for r in all_results)
    losses = sum(r["status"] not in ("WIN", "DRAW") for r in all_results)

    win_rate = wins / len(all_results) if all_results else 0.0
    draw_rate = draws / len(all_results) if all_results else 0.0
    loss_rate = losses / len(all_results) if all_results else 0.0

    primary_pass = two_win_rate >= PROMOTION_MIN_WINRATE
    secondary_pass = (truncation_rate <= PROMOTION_MAX_TRUNCATION_RATE
        and (avg_win_moves == 0.0 or avg_win_moves <= PROMOTION_MAX_AVG_WIN_MOVES)
        and avg_progress_score >= PROMOTION_MIN_PROGRESS_SCORE)
    promoted = primary_pass and secondary_pass

    return {"promoted": promoted,
            "primary_pass": primary_pass,
            "secondary_pass": secondary_pass,
            "two_player_wins": two_wins,
            "two_player_draws": two_draws,
            "two_player_losses": two_losses,
            "two_player_win_rate": two_win_rate,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": win_rate,
            "draw_rate": draw_rate,
            "loss_rate": loss_rate,
            "truncation_rate": truncation_rate,
            "avg_win_moves": avg_win_moves,
            "avg_final_score": avg_final_score,
            "avg_progress_score": avg_progress_score,
            "avg_backward_moves": avg_backward_moves,
            "avg_undo_moves": avg_undo_moves,
            "avg_jump_moves": avg_jump_moves,
            "raw_results": all_results}

def print_promotion_report(report: dict) -> None:
    print("\n" + "=" * 80)
    print("CHECKPOINT PROMOTION REPORT")
    print("=" * 80)
    print(f"promoted              : {report['promoted']}")
    print(f"primary_pass          : {report['primary_pass']}")
    print(f"secondary_pass        : {report['secondary_pass']}")
    print(f"2p wins/draws/losses  : {report['two_player_wins']} / {report['two_player_draws']} / {report['two_player_losses']}")
    print(f"2p win_rate           : {report['two_player_win_rate']:.3f}")
    print(f"all wins/draws/losses : {report['wins']} / {report['draws']} / {report['losses']}")
    print(f"all win_rate          : {report['win_rate']:.3f}")
    print(f"draw_rate             : {report['draw_rate']:.3f}")
    print(f"loss_rate             : {report['loss_rate']:.3f}")
    print(f"truncation_rate       : {report['truncation_rate']:.3f}")
    print(f"avg_win_moves         : {report['avg_win_moves']:.2f}")
    print(f"avg_final_score       : {report['avg_final_score']:.2f}")
    print(f"avg_progress          : {report['avg_progress_score']:.2f}")
    print(f"avg_backward          : {report['avg_backward_moves']:.2f}")
    print(f"avg_undo              : {report['avg_undo_moves']:.2f}")
    print(f"avg_jump_moves        : {report['avg_jump_moves']:.2f}")

def average_or_zero(xs):
    return sum(xs) / len(xs) if xs else 0.0

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

    def train_policy_value_batch_soft(self, batch, value_weight: float = 0.1):
        """
        Policy + value training where the policy target is a probability distribution
        over legal actions (e.g. from MCTS visit counts).
        """
        if not batch:
            return 0.0

        self.model.train()
        self.optimizer.zero_grad()

        losses = []
        for item in batch:
            gs = self._graph_to_device(item["graph_state"])
            target_policy = item["target_policy"].to(self.device)   # shape [num_actions]
            target_value = torch.tensor(item["target_value"], dtype=torch.float32, device=self.device)

            logits, pred_value = self.model(gs)

            log_probs = F.log_softmax(logits, dim=0)
            policy_loss = -(target_policy * log_probs).sum()

            value_loss = F.mse_loss(pred_value, target_value)
            losses.append(policy_loss + value_weight * value_loss)

        batch_loss = torch.stack(losses).mean()
        batch_loss.backward()
        self.optimizer.step()

        return float(batch_loss.item())

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


from collections import defaultdict
from environment import ChineseCheckersEnv
import copy

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

def mcts_label_for_observation(learner: TrainableAgent, env: ChineseCheckersEnv, colour: str, num_simulations: int = 16):
    """
    Run MCTS from the current environment state and return:
      - chosen action (best action from search)
      - graph_state
      - target_policy distribution aligned with graph_state.legal_actions
    """
    search_env = LocalSearchEnv(copy.deepcopy(env))

    mcts = NeuralMCTS(model=learner.model,
                      graph_builder=learner.policy.graph_builder,
                      device=learner.device,
                      num_simulations=num_simulations)

    try:
        best_action, actions, probs = mcts.search(search_env)
    except Exception:
        obs = env.observe(colour)
        heuristic = HeuristicPolicy(epsilon=0.0)
        best_action = heuristic.select_action(obs)
        gs = learner.graph_from_observation(obs)

        target_policy = torch.zeros(len(gs.legal_actions), dtype=torch.float32)
        for i, (pin_id, _, to_idx) in enumerate(gs.legal_actions):
            if (pin_id, to_idx) == best_action:
                target_policy[i] = 1.0
                break

        return best_action, gs, target_policy

    obs = env.observe(colour)
    gs = learner.graph_from_observation(obs)

    # Align MCTS action probabilities with graph_state.legal_actions
    action_to_prob = {tuple(a): float(p) for a, p in zip(actions, probs.tolist())}

    target_policy = []
    for pin_id, _, to_idx in gs.legal_actions:
        target_policy.append(action_to_prob.get((pin_id, to_idx), 0.0))

    target_policy = torch.tensor(target_policy, dtype=torch.float32)
    if target_policy.sum() > 0:
        target_policy = target_policy / target_policy.sum()
    else:
        # fallback: uniform over legal actions
        target_policy = torch.ones(len(gs.legal_actions), dtype=torch.float32)
        target_policy = target_policy / target_policy.sum()

    return best_action, gs, target_policy

def terminal_value_from_result(game, final_state, colour: str, truncated: bool) -> float:
    """
    For truncated games, use normalized final score instead of a blanket penalty.
    """
    if truncated:
        return normalized_final_score(game, colour)

    me = next(p for p in final_state["players"] if p["colour"] == colour)
    if me["status"] == "WIN":
        return 1.0
    if me["status"] == "DRAW":
        return 0.0
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
                        num_players: int = 6):
    """
    Warm-start using heuristic imitation only.
    Training is triggered by positions collected, not by every Nth game.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    heuristic = HeuristicPolicy(epsilon=0.0)
    learner = TrainableAgent(name="shared_model",
                            device=device,
                            hidden_dim=MODEL_HIDDEN_DIM,
                            num_layers=MODEL_NUM_LAYERS)
    
    buffer = ReplayBuffer(capacity=100000)

    maybe_load_into_learner(learner, start_checkpoint)

    positions_since_update = 0

    for local_game_idx in range(1, num_games + 1):
        game_idx = start_game_index + local_game_idx
        num_players = random.choices(TRAIN_PLAYER_COUNTS, weights=TRAIN_PLAYER_COUNT_WEIGHTS, k=1)[0]
        env = ChineseCheckersEnv(num_players=num_players)

        env.reset()
        per_colour_examples = defaultdict(list)

        done = False
        game_start = time.time()

        while not done and env.game.move_count < max_moves_per_game:
            colour = env.current_turn_colour
            obs = env.observe(colour)

            heuristic_action = heuristic.select_action(obs)
            gs = learner.graph_from_observation(obs)

            action_index = None
            for i, (pin_id, _, to_idx) in enumerate(gs.legal_actions):
                if (pin_id, to_idx) == heuristic_action:
                    action_index = i
                    break

            if action_index is None:
                raise RuntimeError("Heuristic action not found in legal action list")

            per_colour_examples[colour].append({
                "graph_state": gs,
                "action_index": action_index,
            })

            step_result = env.step(colour, heuristic_action)
            done = step_result.done

        truncated = env.game.move_count >= max_moves_per_game
        if truncated:
            print(f"[bootstrap] game={game_idx} hit move cap ({max_moves_per_game})")

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
            progress_print("bootstrap",
                          f"starting update cycle: positions_since_update={positions_since_update} "
                          f"buffer={len(buffer)} updates={updates_per_cycle}")

            losses = []
            for update_idx in range(updates_per_cycle):
                batch = buffer.sample(batch_size)
                loss = learner.train_policy_batch(batch)
                losses.append(loss)

                if (update_idx + 1) % 10 == 0 or (update_idx + 1) == updates_per_cycle:
                    running_avg = sum(losses) / len(losses)
                    progress_print("bootstrap",
                                  f"update {update_idx+1}/{updates_per_cycle} "
                                  f"current_loss={loss:.4f} avg_loss={running_avg:.4f}")

            avg_loss = sum(losses) / len(losses)
            positions_since_update = 0

            progress_print("bootstrap",
                           f"finished update cycle avg_loss={avg_loss:.4f}")

        elapsed = time.time() - game_start
        loss_str = f"{avg_loss:.4f}" if avg_loss is not None else "NA"
        print(f"[bootstrap] game={game_idx} moves={env.game.move_count} "
              f"examples={len(examples)} buffer={len(buffer)} "
              f"positions_since_update={positions_since_update} "
              f"avg_loss={loss_str} truncated={truncated} time={elapsed:.2f}s")

        if game_idx % checkpoint_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"shared_model_{game_idx}.pt")
            print(f"[bootstrap] saving checkpoint: {ckpt_path}")
            learner.save(ckpt_path)

    learner.save(os.path.join(checkpoint_dir, "shared_model_final.pt"))


def evaluate_warmstart_topk_match(checkpoint_path: str,
                                  eval_games: int = 20,
                                  max_positions: int = 1000,
                                  device: str = "cpu",
                                  k: int = 3) -> dict:
    """
    Measures how often the model's chosen action is among the heuristic's top-k legal moves
    on states generated by heuristic play.

    Returns:{"topk_match_rate": float,"matches": int,"total": int,"k": int,}
    """
    env = ChineseCheckersEnv(num_players=6)

    # Make heuristic deterministic for evaluation
    heuristic = HeuristicPolicy(epsilon=0.0)

    model = load_model(checkpoint_path, device=device)
    learner = MyPolicy(model=model, device=device)

    matches = 0
    total = 0

    for _ in range(eval_games):
        env.reset()
        done = False

        while not done and total < max_positions:
            colour = env.current_turn_colour
            obs = env.observe(colour)

            state = obs["state"]
            legal_moves = obs["legal_moves"]
            my_positions = state["pins"][colour]

            # Score every legal move with the heuristic
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

            # Sort descending by heuristic score
            scored_actions.sort(key=lambda x: x[1], reverse=True)

            topk_actions = {action for action, _ in scored_actions[:k]}
            model_action = learner.select_action(obs)

            if tuple(model_action) in topk_actions:
                matches += 1

            total += 1

            # Keep state generation heuristic-driven
            step_result = env.step(colour, heuristic.select_action(obs))
            done = step_result.done

        if total >= max_positions:
            break

    match_rate = matches / total if total > 0 else 0.0
    return {"topk_match_rate": match_rate,"matches": matches,"total": total,"k": k}


def warmstart_until_good_enough(target_action_match: float = 0.80,
                                bootstrap_chunk_games: int = 50,
                                max_bootstrap_games: int = 2000,
                                batch_size: int = 64,
                                checkpoint_every: int = 25,
                                checkpoint_dir: str = "checkpoints/bootstrap",
                                device: str = "cpu",
                                max_moves_per_game: int = 500,
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
                            start_game_index=total_games)
        
        total_games += bootstrap_chunk_games

        if not os.path.exists(final_ckpt):
            raise FileNotFoundError(f"Bootstrap checkpoint missing: {final_ckpt}")

        action_eval = evaluate_warmstart_topk_match(
            checkpoint_path=final_ckpt,
            eval_games=20,
            max_positions=1000,
            device=device,
            k=3,
        )

        print(f"[warmstart eval] games={total_games} "
              f"top{action_eval['k']}_match_rate={action_eval['topk_match_rate']:.3f} "
              f"({action_eval['matches']}/{action_eval['total']})")

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
                         value_weight: float = 0.1,
                         num_players: int = 6,
                         mcts_simulations: int = 16):
    """
    MCTS-labeled self-play:
    - learner acts for all seats
    - policy targets come from MCTS visit distributions
    - value targets come from final outcome / normalized truncated score
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    learner = TrainableAgent(name="shared_model",
                             device=device,
                             hidden_dim=MODEL_HIDDEN_DIM,
                             num_layers=MODEL_NUM_LAYERS)
    buffer = ReplayBuffer(capacity=200000)

    maybe_load_into_learner(learner, start_checkpoint)

    positions_since_update = 0
    total_updates_run = 0

    for local_game_idx in range(1, num_games + 1):
        game_idx = start_game_index + local_game_idx
        num_players = random.choices(TRAIN_PLAYER_COUNTS,
                                     weights=TRAIN_PLAYER_COUNT_WEIGHTS,
                                     k=1)[0]

        env = ChineseCheckersEnv(num_players=num_players)
        env.reset()

        progress_print("self-play",
                       f"starting game={game_idx} num_players={num_players}")

        per_colour_examples = defaultdict(list)
        done = False
        game_start = time.time()

        while not done and env.game.move_count < max_moves_per_game:
            colour = env.current_turn_colour

            # Search-improved target + chosen action
            chosen_action, gs, target_policy = mcts_label_for_observation(learner=learner,
                                                                          env=env,
                                                                          colour=colour,
                                                                          num_simulations=mcts_simulations)

            per_colour_examples[colour].append({"graph_state": gs, "target_policy": target_policy})

            step_result = env.step(colour, chosen_action)
            done = step_result.done

        truncated = env.game.move_count >= max_moves_per_game
        if truncated:
            print(f"[self-play] game={game_idx} hit move cap ({max_moves_per_game})")

        final_state = env.game.to_public_state()

        examples = []
        for colour, exs in per_colour_examples.items():
            target_value = terminal_value_from_result(env.game, final_state, colour, truncated=truncated)
            for ex in exs:
                ex["target_value"] = target_value
                examples.append(ex)

        for ex in examples:
            buffer.add(ex)

        positions_since_update += len(examples)

        avg_loss = None
        updates_this_game = 0

        if positions_since_update >= positions_per_cycle and len(buffer) >= batch_size:
            progress_print("self-play",
                           f"starting update cycle: positions_since_update={positions_since_update} "
                           f"buffer={len(buffer)} updates={updates_per_cycle}")

            losses = []
            for update_idx in range(updates_per_cycle):
                batch = buffer.sample(batch_size)
                loss = learner.train_policy_value_batch_soft(batch, value_weight=value_weight)
                losses.append(loss)
                updates_this_game += 1
                total_updates_run += 1

                if (update_idx + 1) % 10 == 0 or (update_idx + 1) == updates_per_cycle:
                    running_avg = sum(losses) / len(losses)
                    progress_print("self-play",
                                   f"update {update_idx+1}/{updates_per_cycle} "
                                   f"current_loss={loss:.4f} avg_loss={running_avg:.4f}")

            avg_loss = sum(losses) / len(losses)
            positions_since_update = 0

            progress_print("self-play",
                           f"finished update cycle avg_loss={avg_loss:.4f}")

        elapsed = time.time() - game_start
        loss_str = f"{avg_loss:.4f}" if avg_loss is not None else "NA"

        print(f"[self-play] game={game_idx} moves={env.game.move_count} "
              f"examples={len(examples)} buffer={len(buffer)} "
              f"positions_since_update={positions_since_update} "
              f"updates_this_game={updates_this_game} total_updates={total_updates_run} "
              f"loss={loss_str} truncated={truncated} time={elapsed:.2f}s")

        if game_idx % checkpoint_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"shared_model_{game_idx}.pt")
            print(f"[self-play] saving checkpoint: {ckpt_path}")
            learner.save(ckpt_path)

    learner.save(os.path.join(checkpoint_dir, "shared_model_final.pt"))

def self_play_with_promotion(start_checkpoint: str,
                             total_blocks: int = 20,
                             games_per_block: int = 100,
                             batch_size: int = 64,
                             checkpoint_dir: str = "checkpoints/self_play",
                             device: str = "cpu",
                             max_moves_per_game: int = 300,
                             positions_per_cycle: int = 2500,
                             updates_per_cycle: int = 150,
                             value_weight: float = 0.1,
                             mcts_simulations: int = 16):
    os.makedirs(checkpoint_dir, exist_ok=True)

    champion_ckpt = os.path.join(checkpoint_dir, CHAMPION_NAME)
    if not os.path.exists(champion_ckpt):
        # initialize champion from warm-start or supplied starting checkpoint
        import shutil
        shutil.copyfile(start_checkpoint, champion_ckpt)
        print(f"[promotion] initialized champion from {start_checkpoint}")

    current_start_ckpt = champion_ckpt
    total_games_done = 0

    for block_idx in range(1, total_blocks + 1):
        print("\n" + "#" * 80)
        print(f"[promotion] STARTING BLOCK {block_idx}/{total_blocks}")
        print(f"[promotion] champion = {current_start_ckpt}")
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
                             mcts_simulations=mcts_simulations)

        # self_play_refinement writes shared_model_final.pt
        produced_ckpt = os.path.join(checkpoint_dir, "shared_model_final.pt")
        if not os.path.exists(produced_ckpt):
            raise FileNotFoundError(f"Expected challenger checkpoint missing: {produced_ckpt}")

        import shutil
        shutil.copyfile(produced_ckpt, challenger_ckpt)
        print(f"[promotion] challenger saved to {challenger_ckpt}")

        # Evaluate challenger against champion
        report = evaluate_checkpoint_promotion(champion_ckpt=champion_ckpt,
                                               challenger_ckpt=challenger_ckpt,
                                               device=device,
                                               matches_per_player_count=PROMOTION_MATCHES,
                                               player_counts=PROMOTION_PLAYER_COUNTS,
                                               max_moves=PROMOTION_MAX_MOVES)
        print_promotion_report(report)

        # Promote or reject
        if report["promoted"]:
            shutil.copyfile(challenger_ckpt, champion_ckpt)
            current_start_ckpt = champion_ckpt
            print(f"[promotion] challenger PROMOTED -> new champion")
        else:
            current_start_ckpt = champion_ckpt
            print(f"[promotion] challenger REJECTED -> keeping old champion")

        total_games_done += games_per_block

    print("\n[promotion] training complete.")
    print(f"[promotion] final champion: {champion_ckpt}")

def normalized_final_score(game, colour: str) -> float:
    """Extract the final score for the given colour from the game state and normalize it to [-1, 1]."""
    player = next(p for p in game.players if p.colour == colour)
    score_dict = game.scores.get(player.player_id, {})
    raw = score_dict.get("final_score", 0.0)

    # Example normalization
    return max(-1.0, min(1.0, raw / 1200.0))


def checkpoint_game_index(path: str) -> int:
    name = os.path.basename(path)
    if name.startswith("shared_model_") and name.endswith(".pt"):
        stem = name[len("shared_model_"):-3]
        if stem.isdigit():
            return int(stem)
    return 0


def maybe_load_into_learner(learner: TrainableAgent, checkpoint_path: str | None):
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        learner.load(checkpoint_path)


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


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    RUN_WARMSTART = True
    RUN_SELF_PLAY_PROMOTION = True

    WARMSTART_DIR = BOOTSTRAP_CHECKPOINT_DIR
    SELFPLAY_DIR = SELFPLAY_CHECKPOINT_DIR

    WARMSTART_CHUNK_GAMES = 50
    WARMSTART_MAX_GAMES = 1000
    WARMSTART_BATCH_SIZE = 64
    WARMSTART_CHECKPOINT_EVERY = 25
    WARMSTART_MAX_MOVES = 300
    WARMSTART_TARGET_ACTION_MATCH = 0.90

    SELFPLAY_TOTAL_BLOCKS = 50
    SELFPLAY_GAMES_PER_BLOCK = 100
    SELFPLAY_BATCH_SIZE = 64
    SELFPLAY_MAX_MOVES = 300
    SELFPLAY_POSITIONS_PER_CYCLE = 300
    SELFPLAY_UPDATES_PER_CYCLE = 30
    SELFPLAY_VALUE_WEIGHT = 0.05

    MANUAL_START_CHECKPOINT = os.path.join(BOOTSTRAP_CHECKPOINT_DIR, "shared_model_final.pt")

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
                                                                 max_moves_per_game=WARMSTART_MAX_MOVES)
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
                                 mcts_simulations=8)


if __name__ == "__main__":
    main()