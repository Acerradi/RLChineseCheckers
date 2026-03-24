import random
from collections import deque


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


import os
import sys
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


from policy_template import build_model, save_model, load_model, GraphState, MyPolicy, HeuristicPolicy


class TrainableAgent:
    def __init__(self, name: str, device: str = "cpu", lr: float = 1e-3):
        self.name = name
        self.device = device
        self.model = build_model(device=device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.policy = MyPolicy(model=self.model, device=device)

    def graph_from_observation(self, observation):
        gs = self.policy.graph_builder.build(observation)
        return GraphState(
            x=gs.x.to(self.device),
            edge_index=gs.edge_index.to(self.device),
            legal_actions=gs.legal_actions,
            controlled_colour=gs.controlled_colour,
            meta=gs.meta,
        )

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

    def train_policy_value_batch(self, batch, learning_weight: float = 0.5):
        if not batch:
            return 0.0

        self.model.train()
        total_loss = 0.0

        for item in batch:
            gs = GraphState(
                x=item["graph_state"].x.to(self.device),
                edge_index=item["graph_state"].edge_index.to(self.device),
                legal_actions=item["graph_state"].legal_actions,
                controlled_colour=item["graph_state"].controlled_colour,
                meta=item["graph_state"].meta,
            )

            target_action_idx = torch.tensor([item["action_index"]], device=self.device)
            target_value = torch.tensor(item["target_value"], dtype=torch.float32, device=self.device)

            logits, pred_value = self.model(gs)

            policy_loss = F.cross_entropy(logits.unsqueeze(0), target_action_idx)
            value_loss = F.mse_loss(pred_value, target_value)

            loss = policy_loss + value_loss * learning_weight

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())

        return total_loss / len(batch)

    def save(self, path: str):
        save_model(self.model, path)


from collections import defaultdict
from environment import ChineseCheckersEnv

def terminal_value_from_status(final_state, colour: str) -> float:
    """Determine the target value for a player based on the final game state and their colour."""
    me = next(p for p in final_state["players"] if p["colour"] == colour)
    if me["status"] == "WIN":
        return 1.0
    if me["status"] == "DRAW":
        return 0.0
    return -1.0


def bootstrap_imitation(num_games: int = 500,
                        batch_size: int = 64,
                        checkpoint_every: int = 50,
                        checkpoint_dir: str = "checkpoints/bootstrap",
                        device: str = "cpu",
                        start_checkpoint: str | None = None,
                        max_moves_per_game: int = 500
                        ):
    """Run a single round of bootstrap imitation learning using the heuristic policy to generate training data."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = ChineseCheckersEnv(num_players=6)
    heuristic = HeuristicPolicy(epsilon=0.0)
    learner = TrainableAgent(name="shared_model", device=device)
    buffer = ReplayBuffer(capacity=50000)

    if start_checkpoint is not None and os.path.exists(start_checkpoint):
        learner.model.load_state_dict(torch.load(start_checkpoint, map_location=device))
        learner.model.eval()

    for game_idx in range(1, num_games + 1):
        env.reset()
        per_colour_examples = defaultdict(list)

        done = False
        game_start = time.time()
        while not done and env.game.move_count < max_moves_per_game:
            colour = env.current_turn_colour
            obs = env.observe(colour)

            # Heuristic chooses the action
            heuristic_action = heuristic.select_action(obs)

            # Convert that action into the legal-action index for supervision
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

        if env.game.move_count >= max_moves_per_game:
            print(f"[bootstrap] game={game_idx} hit move cap ({max_moves_per_game})")

        final_state = env.game.to_public_state()

        for colour, examples in per_colour_examples.items():
            target_value = terminal_value_from_status(final_state, colour)

            #env.game.compute_scores()
            #target_value = normalized_final_score(env.game, colour)

            for ex in examples:
                ex["target_value"] = target_value
                buffer.add(ex)

        updates_per_game = 10
        losses = []

        for _ in range(updates_per_game):
            batch = buffer.sample(batch_size)
            loss = learner.train_policy_value_batch(batch)
            losses.append(loss)

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        elapsed = time.time() - game_start
        if game_idx % 1 == 0:
            print(f"[bootstrap] game={game_idx} moves={env.game.move_count} "
                  f"buffer={len(buffer)} avg_loss={avg_loss:.4f} "
                  f"updates={updates_per_game} time={elapsed:.2f}s"
                  )
        
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


def warmstart_until_good_enough(target_action_match: float = 0.85,
                                bootstrap_chunk_games: int = 100,
                                max_bootstrap_games: int = 2000,
                                batch_size: int = 64,
                                checkpoint_every: int = 50,
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
                            start_checkpoint=current_checkpoint
                            )
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


def self_play_refinement(start_checkpoint: str,
                         num_games: int = 2000,
                         batch_size: int = 64,
                         checkpoint_every: int = 100,
                         checkpoint_dir: str = "checkpoints/self_play",
                         device: str = "cpu",
                         heuristic_mix: float = 0.15,
                         max_moves_per_game: int = 500
                         ):
    """Continue training a policy by having it play against itself, with some heuristic action noise for diversity."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = ChineseCheckersEnv(num_players=6)
    heuristic = HeuristicPolicy()
    learner = TrainableAgent(name="shared_model", device=device)
    learner.model.load_state_dict(torch.load(start_checkpoint, map_location=device))
    learner.model.eval()

    buffer = ReplayBuffer(capacity=100000)

    for game_idx in range(1, num_games + 1):
        env.reset()
        per_colour_examples = defaultdict(list)

        done = False
        game_start = time.time()
        while not done and env.game.move_count < max_moves_per_game:
            colour = env.current_turn_colour
            obs = env.observe(colour)

            # Mostly use the learned model, sometimes use the heuristic for diversity
            if random.random() < heuristic_mix:
                action = heuristic.select_action(obs)
                gs = learner.graph_from_observation(obs)
                action_index = None
                for i, (pin_id, _, to_idx) in enumerate(gs.legal_actions):
                    if (pin_id, to_idx) == action:
                        action_index = i
                        break
            else:
                action, action_index, gs = learner.select_action_with_index(obs)

            if action_index is None:
                raise RuntimeError("Action index not found")

            per_colour_examples[colour].append({"graph_state": gs,
                                                "action_index": action_index,
                                                })

            step_result = env.step(colour, action)
            done = step_result.done

        if env.game.move_count >= max_moves_per_game:
            print(f"[self-play] game={game_idx} hit move cap ({max_moves_per_game})")

        final_state = env.game.to_public_state()

        for colour, examples in per_colour_examples.items():
            target_value = terminal_value_from_status(final_state, colour)
            
            #env.game.compute_scores()
            #target_value = normalized_final_score(env.game, colour)

            for ex in examples:
                ex["target_value"] = target_value
                buffer.add(ex)

        updates_per_game = 10
        losses = []

        for _ in range(updates_per_game):
            batch = buffer.sample(batch_size)
            loss = learner.train_policy_value_batch(batch)
            losses.append(loss)

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        elapsed = time.time() - game_start
        print(f"[self-play] game={game_idx} moves={env.game.move_count} "
              f"buffer={len(buffer)} loss={avg_loss:.4f} time={elapsed:.2f}s")

        if game_idx % checkpoint_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"shared_model_{game_idx}.pt")
            print(f"[self-play] saving checkpoint: {ckpt_path}")
            learner.save(ckpt_path)

    learner.save(os.path.join(checkpoint_dir, "shared_model_final.pt"))


def normalized_final_score(game, colour: str) -> float:
    """Extract the final score for the given colour from the game state and normalize it to [-1, 1]."""
    player = next(p for p in game.players if p.colour == colour)
    score_dict = game.scores.get(player.player_id, {})
    raw = score_dict.get("final_score", 0.0)

    # Example normalization
    return max(-1.0, min(1.0, raw / 1200.0))


def main():
    random.seed(42)
    torch.manual_seed(42)

    device = "cpu"

    root_ckpt_dir = "multi_system_single_machine_minimal/harald_files/checkpoints"
    bootstrap_dir = os.path.join(root_ckpt_dir, "bootstrap")
    selfplay_dir = os.path.join(root_ckpt_dir, "self_play")

    os.makedirs(bootstrap_dir, exist_ok=True)
    os.makedirs(selfplay_dir, exist_ok=True)

    print("=" * 80)
    print("STAGE 1: HEURISTIC WARM-START UNTIL GOOD ENOUGH")
    print("=" * 80)

    warmstart_checkpoint, warmstart_eval = warmstart_until_good_enough(
        target_action_match=0.75,
        bootstrap_chunk_games=100,
        max_bootstrap_games=2000,
        batch_size=64,
        checkpoint_every=50,
        checkpoint_dir=bootstrap_dir,
        max_moves_per_game=500,
        device=device)

    print("\nWarm-start evaluation summary:")
    print(warmstart_eval)

    print("=" * 80)
    print("STAGE 2: SELF-PLAY REFINEMENT")
    print("=" * 80)

    self_play_refinement(start_checkpoint=warmstart_checkpoint,
                         num_games=2000,
                         batch_size=64,
                         checkpoint_every=100,
                         checkpoint_dir=selfplay_dir,
                         device=device,
                         heuristic_mix=0.15,
                         max_moves_per_game=500)

    print("\nTraining complete.")
    print(f"Warm-start checkpoint used: {warmstart_checkpoint}")
    print(f"Final self-play checkpoint: {os.path.join(selfplay_dir, 'shared_model_final.pt')}")

if __name__ == "__main__":
    main()