from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from checkers_board import HexBoard
from policies import BasePolicy


COLOURS = ["red", "lawn green", "yellow", "blue", "gray0", "purple"]
ZONE_TYPES = ["board"] + COLOURS
OCCUPANT_TYPES = ["empty"] + COLOURS

COLOUR_TO_IDX = {c: i for i, c in enumerate(COLOURS)}
ZONE_TO_IDX = {z: i for i, z in enumerate(ZONE_TYPES)}
OCC_TO_IDX = {o: i for i, o in enumerate(OCCUPANT_TYPES)}


@dataclass
class GraphState:
    x: torch.Tensor
    edge_index: torch.Tensor
    legal_actions: List[Tuple[int, int, int]]   # (pin_id, from_idx, to_idx)
    action_features: torch.Tensor               # [num_actions, action_feat_dim]
    controlled_colour: str
    meta: Dict[str, Any]

class BoardGraphBuilder:
    """
    Converts your live/training observation into a graph the GNN can consume.

    Important:
    You may need to adjust `_build_edge_index()` depending on how your
    HexBoard exposes neighbors.
    """

    def __init__(self, board: HexBoard):
        self.board = board
        self.edge_index = self._build_edge_index()

    def _build_edge_index(self) -> torch.Tensor:
        edges = []

        # Your board stores cells as BoardPosition objects with axial coords q, r.
        coords = [(cell.q, cell.r) for cell in self.board.cells]
        coord_to_idx = self.board.index_of

        # Axial hex neighbor directions
        hex_dirs = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, -1),
            (-1, 1),
        ]

        for i, (q, r) in enumerate(coords):
            for dq, dr in hex_dirs:
                nbr = (q + dq, r + dr)
                if nbr in coord_to_idx:
                    j = coord_to_idx[nbr]
                    edges.append((i, j))

        if not edges:
            raise ValueError("No graph edges found after coordinate-based construction")

        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _occupancy_map(self, state: Dict[str, Any]) -> Dict[int, str]:
        occ = {}
        for colour, positions in state["pins"].items():
            for idx in positions:
                occ[idx] = colour
        return occ

    def _target_cells(self, colour: str):
        target_colour = self.board.colour_opposites[colour]
        target_idxs = self.board.axial_of_colour(target_colour)
        return [self.board.cells[i] for i in target_idxs]

    def _home_cells(self, colour: str):
        home_idxs = self.board.axial_of_colour(colour)
        return [self.board.cells[i] for i in home_idxs]

    def _min_dist_to_goal(self, idx: int, colour: str) -> int:
        here = self.board.cells[idx]
        targets = self._target_cells(colour)
        return min(axial_dist(here, tgt) for tgt in targets)

    def _min_dist_to_home(self, idx: int, colour: str) -> int:
        here = self.board.cells[idx]
        homes = self._home_cells(colour)
        return min(axial_dist(here, home) for home in homes)

    def _goal_zone_depth(self, idx: int, colour: str) -> int:
        here = self.board.cells[idx]
        homes = self._home_cells(colour)
        return min(axial_dist(here, home) for home in homes)

    def build(self, observation: Dict[str, Any]) -> GraphState:
        state = observation["state"]
        legal_moves = observation["legal_moves"]
        controlled_colour = observation["colour"]

        goal_dist = {idx: self._min_dist_to_goal(idx, controlled_colour)
            for idx in range(len(self.board.cells))}
        depth_dist = {idx: self._goal_zone_depth(idx, controlled_colour)
            for idx in range(len(self.board.cells))}

        occ = self._occupancy_map(state)
        my_target_zone = self.board.colour_opposites[controlled_colour]
        R = float(self.board.R)

        x_rows = []
        for idx, cell in enumerate(self.board.cells):
            row = []

            # 1) Occupant one-hot
            occupant = occ.get(idx, "empty")
            occ_onehot = [0.0] * len(OCCUPANT_TYPES)
            occ_onehot[OCC_TO_IDX[occupant]] = 1.0
            row.extend(occ_onehot)

            # 2) Zone one-hot
            zone = getattr(cell, "postype", "board")
            zone_onehot = [0.0] * len(ZONE_TYPES)
            zone_onehot[ZONE_TO_IDX.get(zone, 0)] = 1.0
            row.extend(zone_onehot)

            # 3) Existing scalar features
            row.append(1.0 if occupant == controlled_colour else 0.0)
            row.append(1.0 if zone == my_target_zone else 0.0)
            row.append(1.0 if state.get("current_turn_colour") == controlled_colour else 0.0)
            row.append(min(state.get("move_count", 0) / 200.0, 1.0))

            # 4) New geometry features
            q = float(cell.q)
            r = float(cell.r)
            s = float(-cell.q - cell.r)

            row.append(q / (2.0 * R))
            row.append(r / (2.0 * R))
            row.append(s / (2.0 * R))

            # 5) Distance-to-goal / distance-to-home features
            dist_goal = float(goal_dist[idx])
            dist_home = float(depth_dist[idx])

            # Normalization: board is small, 16 is a safe rough divisor
            row.append(min(dist_goal / 16.0, 1.0))
            row.append(min(dist_home / 16.0, 1.0))

            x_rows.append(row)

        x = torch.tensor(x_rows, dtype=torch.float32)

        my_positions = state["pins"][controlled_colour]
        pin_id_to_from = {int(pin_id): my_positions[int(pin_id)] for pin_id in legal_moves.keys()}

        legal_actions: List[Tuple[int, int, int]] = []
        action_feature_rows: List[List[float]] = []

        last_move = state.get("last_move")
        last_from = int(last_move.get("from", -1)) if last_move is not None else -1
        last_to = int(last_move.get("to", -1)) if last_move is not None else -1

        target_zone = self.board.colour_opposites[controlled_colour]

        for pin_id, to_list in legal_moves.items():
            pid = int(pin_id)
            from_idx = pin_id_to_from[pid]

            from_cell = self.board.cells[from_idx]
            from_zone = getattr(from_cell, "postype", "board")
            from_goal_dist = goal_dist[from_idx]
            from_depth = depth_dist[from_idx]

            for to_idx in to_list:
                to_idx = int(to_idx)
                to_cell = self.board.cells[to_idx]
                to_zone = getattr(to_cell, "postype", "board")
                to_goal_dist = goal_dist[to_idx]
                to_depth = depth_dist[to_idx]
                
                progress_gain = float(from_goal_dist - to_goal_dist)
                jump_distance = float(axial_dist(from_cell, to_cell))
                is_jump = 1.0 if jump_distance > 1.0 else 0.0

                from_in_target = 1.0 if from_zone == target_zone else 0.0
                to_in_target = 1.0 if to_zone == target_zone else 0.0
                enters_target = 1.0 if from_zone != target_zone and to_zone == target_zone else 0.0
                leaves_target = 1.0 if from_zone == target_zone and to_zone != target_zone else 0.0

                depth_gain = float(to_depth - from_depth)
                is_immediate_undo = 1.0 if (from_idx == last_to and to_idx == last_from) else 0.0

                # Normalize some numeric move features
                feat_row = [progress_gain / 8.0,
                            min(jump_distance / 8.0, 1.0),
                            is_jump,
                            from_in_target,
                            to_in_target,
                            enters_target,
                            leaves_target,
                            depth_gain / 8.0,
                            is_immediate_undo]

                legal_actions.append((pid, from_idx, to_idx))
                action_feature_rows.append(feat_row)

        if action_feature_rows:
            action_features = torch.tensor(action_feature_rows, dtype=torch.float32)
        else:
            action_features = torch.empty((0, 9), dtype=torch.float32)

        return GraphState(x=x,
                          edge_index=self.edge_index,
                          legal_actions=legal_actions,
                          action_features=action_features,
                          controlled_colour=controlled_colour,
                          meta=state)


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.msg = nn.Linear(in_dim, out_dim)
        self.self_proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index

        messages = self.msg(x[src])
        agg = torch.zeros(x.size(0), messages.size(-1), device=x.device)
        agg.index_add_(0, dst, messages)

        deg = torch.zeros(x.size(0), device=x.device)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        deg = deg.clamp(min=1.0).unsqueeze(-1)

        out = self.self_proj(x) + agg / deg
        out = self.norm(out)
        return F.relu(out)


def axial_dist(cell_a, cell_b) -> int:
    dq = abs(cell_a.q - cell_b.q)
    dr = abs(cell_a.r - cell_b.r)
    ds = abs((-cell_a.q - cell_a.r) - (-cell_b.q - cell_b.r))
    return max(dq, dr, ds)


class HeuristicPolicy(BasePolicy):
    def __init__(self, board: Optional[HexBoard] = None, epsilon: float = 0.05):
        self.board = board or HexBoard()
        self.epsilon = epsilon
        self.rng = random.Random()

    def _target_cells(self, colour: str):
        target_colour = self.board.colour_opposites[colour]
        target_idxs = self.board.axial_of_colour(target_colour)
        return [self.board.cells[i] for i in target_idxs]

    def _min_dist_to_goal(self, idx: int, colour: str) -> int:
        here = self.board.cells[idx]
        target_colour = self.board.colour_opposites[colour]

        # If already inside target zone, use geometric target distance as fallback
        # and let the special in-goal logic decide whether to settle deeper.
        if getattr(here, "postype", "board") == target_colour:
            return min(axial_dist(here, tgt) for tgt in self._target_cells(colour))

        open_targets = [i for i in self._open_target_indices(colour)]
        if open_targets:
            return min(axial_dist(here, self.board.cells[i]) for i in open_targets)

        return min(axial_dist(here, tgt) for tgt in self._target_cells(colour))

    def _goal_zone_depth(self, idx: int, colour: str) -> int:
        """
        Higher is better once inside the target triangle.

        We measure depth using distance from the player's own home triangle:
        deeper into the opposite triangle generally means farther from home.
        """
        home_idxs = self.board.axial_of_colour(colour)
        home_cells = [self.board.cells[i] for i in home_idxs]
        here = self.board.cells[idx]
        return min(axial_dist(here, home) for home in home_cells)

    def _count_pieces_outside_target(self, colour: str, state: Dict[str, Any]) -> int:
        target_zone = self.board.colour_opposites[colour]
        my_positions = state["pins"][colour]
        return sum(1 for idx in my_positions if getattr(self.board.cells[idx], "postype", "board") != target_zone)

    def _count_pieces_in_home(self, colour: str, state: Dict[str, Any]) -> int:
        my_positions = state["pins"][colour]
        return sum(1 for idx in my_positions
            if getattr(self.board.cells[idx], "postype", "board") == colour)

    def _open_target_indices(self, colour: str):
        target_colour = self.board.colour_opposites[colour]
        target_idxs = self.board.axial_of_colour(target_colour)
        return [i for i in target_idxs if not self.board.cells[i].occupied]

    def _min_dist_to_open_target(self, idx: int, colour: str) -> int:
        here = self.board.cells[idx]
        open_target_idxs = self._open_target_indices(colour)

        if open_target_idxs:
            return min(axial_dist(here, self.board.cells[i]) for i in open_target_idxs)

        # fallback if target is full
        return self._min_dist_to_goal(idx, colour)

    def _recent_repetition_penalty(self, colour: str, from_idx: int, to_idx: int, state: Dict[str, Any]) -> float:
        """
        Penalize immediate undo strongly, and repeated oscillation patterns mildly.
        Uses last_move only because that is what your public state currently exposes.
        """
        penalty = 0.0
        last_move = state.get("last_move")
        if last_move is not None and last_move.get("colour") == colour:
            last_from = int(last_move.get("from", -1))
            last_to = int(last_move.get("to", -1))

            # immediate undo
            if last_from == to_idx and last_to == from_idx:
                penalty -= 20.0

            # weak penalty for returning to the same origin or destination pattern
            if last_to == to_idx:
                penalty -= 4.0
            if last_from == from_idx:
                penalty -= 2.0

        return penalty

    def _score_action(self, colour: str, from_idx: int, to_idx: int, state: Dict[str, Any]) -> float:
        before = self._min_dist_to_goal(from_idx, colour)
        after = self._min_dist_to_goal(to_idx, colour)

        before_open = self._min_dist_to_open_target(from_idx, colour)
        after_open = self._min_dist_to_open_target(to_idx, colour)

        progress_gain = before - after
        open_target_gain = before_open - after_open
        jump_distance = axial_dist(self.board.cells[from_idx], self.board.cells[to_idx])

        target_zone = self.board.colour_opposites[colour]
        from_zone = getattr(self.board.cells[from_idx], "postype", "board")
        to_zone = getattr(self.board.cells[to_idx], "postype", "board")

        outside_count = self._count_pieces_outside_target(colour, state)
        home_count = self._count_pieces_in_home(colour, state)
        move_count = int(state.get("move_count", 0))


        score = 0.0

        # Main priorities
        score += 14.0 * progress_gain # primary reward for moves that reduce distance to goal
        score += 10.0 * open_target_gain # reward moves that approach open targets, even if they don't immediately reduce goal distance
        if progress_gain < 0:
            score += 10.0 * progress_gain   # penalize backward moves
        if open_target_gain < 0:
            score += 14.0 * open_target_gain  # stronger penalty for moves that don't approach an open target
        if progress_gain == 0 and not (from_zone == target_zone and to_zone == target_zone):
            score -= 2.5   # small penalty for non-progressing moves outside the target zone
        
        # Long jumps
        jump_bonus = 2.5 * max(0, jump_distance - 1)
        if progress_gain > 0 or open_target_gain > 0:
            score += jump_bonus
        elif progress_gain < 0 and open_target_gain < 0:
            score -= jump_bonus
        
        # Entering/leaving target zone
        if from_zone != target_zone and to_zone == target_zone:
            score += 10.0
            if outside_count <= 2:
                # bonus for entering target zone when few pieces are left outside
                score += 10.0
        if from_zone == target_zone and to_zone != target_zone:
            score -= 20.0

        # Moves inside the target zone: prefer deeper cells and discourage lateral shuffling
        if from_zone == target_zone and to_zone == target_zone:
            from_depth = self._goal_zone_depth(from_idx, colour)
            to_depth = self._goal_zone_depth(to_idx, colour)
            depth_gain = to_depth - from_depth

            score += 10.0 * depth_gain

            if depth_gain == 0:
                score -= 8.0   # discourage lateral goal-zone shuffling
            elif depth_gain < 0:
                score += 12.0 * depth_gain  # stronger penalty for moving "outward"

            if outside_count > 1 and depth_gain <= 0:
                score -= 6.0
            
            if outside_count == 1 and depth_gain < 0:
                score -= 8.0

        # Bonus for leaving home zone
        if from_zone == colour and to_zone != colour:
            score += 12.0

        if home_count > 0:
            # urgency grows over time if pieces remain in home
            home_urgency = min(move_count / 40.0, 4.0)

            if from_zone == colour:
                score += 6.0 * home_urgency
            else:
                # mild pressure to not ignore evacuation forever
                score -= 1.0 * home_urgency


        # Penalize immediate undo of the last move
        score += self._recent_repetition_penalty(colour, from_idx, to_idx, state)
        
        # If already in target zone, discourage moves that don't gain progress
        if from_zone == target_zone and to_zone == target_zone and progress_gain <= 0 and open_target_gain <= 0:
            score -= 8.0

        # Add some noise for tiebreaking and exploration
        score += self.rng.uniform(-self.epsilon, self.epsilon)
        return score

    def select_action(self, observation: Dict[str, Any]) -> Tuple[int, int]:
        colour = observation["colour"]
        state = observation["state"]
        legal_moves = observation["legal_moves"]

        my_positions = state["pins"][colour]

        best_action = None
        best_score = -float("inf")

        for pin_id, to_list in legal_moves.items():
            pid = int(pin_id)
            from_idx = my_positions[pid]
            for to_idx in to_list:
                s = self._score_action(colour, from_idx, int(to_idx), state)
                if s > best_score:
                    best_score = s
                    best_action = (pid, int(to_idx))

        if best_action is None:
            raise RuntimeError("No legal moves available")

        return best_action

class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        d = in_dim
        for _ in range(num_layers):
            self.layers.append(GraphConv(d, hidden_dim))
            d = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


class MovePolicyValueNet(nn.Module):
    def __init__(self, node_feat_dim: int, 
                 action_feat_dim: int = 9,
                 hidden_dim: int = 128,
                 num_layers: int = 4):
        
        super().__init__()
        self.action_feat_dim = action_feat_dim

        self.encoder = GraphEncoder(in_dim=node_feat_dim,
                                    hidden_dim=hidden_dim,
                                    num_layers=num_layers)

        self.global_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                         nn.ReLU())

        self.action_proj = nn.Sequential(nn.Linear(action_feat_dim, hidden_dim),
                                         nn.ReLU())

        self.policy_mlp = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, 1))

        self.value_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, 1),
                                       nn.Tanh())

    def encode(self, graph_state: GraphState) -> Tuple[torch.Tensor, torch.Tensor]:
        node_emb = self.encoder(graph_state.x, graph_state.edge_index)
        global_emb = self.global_proj(node_emb.mean(dim=0))
        return node_emb, global_emb

    def forward(self, graph_state: GraphState) -> Tuple[torch.Tensor, torch.Tensor]:
        node_emb, global_emb = self.encode(graph_state)

        scores = []
        for i, (_, from_idx, to_idx) in enumerate(graph_state.legal_actions):
            action_feat = graph_state.action_features[i]
            action_emb = self.action_proj(action_feat)

            feat = torch.cat([node_emb[from_idx],
                              node_emb[to_idx],
                              global_emb,
                              action_emb], dim=-1)
            score = self.policy_mlp(feat).squeeze(-1)
            scores.append(score)

        if scores:
            policy_logits = torch.stack(scores, dim=0)
        else:
            policy_logits = torch.empty(0, dtype=node_emb.dtype, device=node_emb.device)

        value = self.value_mlp(global_emb).squeeze(-1)
        return policy_logits, value

@torch.no_grad()
def evaluate_graph(model: MovePolicyValueNet, graph_state: GraphState) -> Tuple[torch.Tensor, float]:
    model.eval()
    logits, value = model(graph_state)

    if logits.numel() == 0:
        priors = torch.empty(0, dtype=torch.float32)
    else:
        priors = torch.softmax(logits, dim=0)

    return priors.cpu(), float(value.cpu())


class MyPolicy(BasePolicy):
    def __init__(self,
                 model: Optional[MovePolicyValueNet] = None,
                 board: Optional[HexBoard] = None,
                 device: str = "cpu",
                 use_mcts: bool = False,
                 mcts_simulations: int = 28,
                 hidden_dim: int = 128,
                 num_layers: int = 4,):
        
        self.device = torch.device(device)
        self.board = board or HexBoard()
        self.graph_builder = BoardGraphBuilder(self.board)

        if model is None:
            node_feat_dim = len(OCCUPANT_TYPES) + len(ZONE_TYPES) + 9
            model = MovePolicyValueNet(node_feat_dim=node_feat_dim,
                                       action_feat_dim=9,
                                       hidden_dim=hidden_dim,
                                       num_layers=num_layers)

        self.model = model.to(self.device)
        self.model.eval()

        self.use_mcts = use_mcts
        self.mcts_simulations = mcts_simulations

    @torch.no_grad()
    def select_action(self, observation: Dict[str, Any]) -> Tuple[int, int]:
        graph_state = self.graph_builder.build(observation)
        graph_state = GraphState(x=graph_state.x.to(self.device),
                                 edge_index=graph_state.edge_index.to(self.device),
                                 legal_actions=graph_state.legal_actions,
                                 action_features=graph_state.action_features.to(self.device),
                                 controlled_colour=graph_state.controlled_colour,
                                 meta=graph_state.meta)

        logits, _ = self.model(graph_state)

        if logits.numel() == 0:
            raise RuntimeError("No legal moves available")

        # Simple anti-undo penalty based on public state
        state = observation["state"]
        last_move = state.get("last_move")
        if last_move is not None:
            penalized_logits = logits.clone()
            last_from = int(last_move.get("from", -1))
            last_to = int(last_move.get("to", -1))

            for i, (pin_id, from_idx, to_idx) in enumerate(graph_state.legal_actions):
                if from_idx == last_to and to_idx == last_from:
                    penalized_logits[i] -= 15.0

            logits = penalized_logits

        best_idx = int(torch.argmax(logits).item())
        pin_id, _, to_idx = graph_state.legal_actions[best_idx]
        return pin_id, to_idx

    def select_action_with_mcts(self, env: SearchEnvironment) -> Tuple[int, int]:
        """
        Local training/evaluation path.
        Requires your Path 2 environment to implement clone/current_player_colour/observe/step.
        """
        mcts = NeuralMCTS(
            model=self.model,
            graph_builder=self.graph_builder,
            device=str(self.device),
            num_simulations=self.mcts_simulations,
        )
        best_action, actions, probs = mcts.search(env)
        return best_action


def build_model(device: str = "cpu", hidden_dim: int = 128, num_layers: int = 4) -> MovePolicyValueNet:
    node_feat_dim = len(OCCUPANT_TYPES) + len(ZONE_TYPES) + 9
    model = MovePolicyValueNet(node_feat_dim=node_feat_dim,
                               action_feat_dim=9,
                               hidden_dim=hidden_dim,
                               num_layers=num_layers)
    return model.to(device)

def save_model(model: MovePolicyValueNet, path: str, hidden_dim: int = 128, num_layers: int = 4) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
    }
    torch.save(payload, path)

def load_model(path: str, device: str = "cpu", hidden_dim: int = 128, num_layers: int = 4) -> MovePolicyValueNet:
    payload = torch.load(path, map_location=device)

    # backward compatibility: old checkpoints may just be raw state_dicts
    if isinstance(payload, dict) and "model_state_dict" in payload:
        ckpt_hidden = payload.get("hidden_dim", hidden_dim)
        ckpt_layers = payload.get("num_layers", num_layers)

        if ckpt_hidden != hidden_dim or ckpt_layers != num_layers:
            raise ValueError(
                f"Checkpoint architecture mismatch: "
                f"checkpoint has hidden_dim={ckpt_hidden}, num_layers={ckpt_layers}, "
                f"but requested hidden_dim={hidden_dim}, num_layers={num_layers}"
            )

        state_dict = payload["model_state_dict"]
    else:
        # old raw state_dict checkpoint
        state_dict = payload

    model = build_model(device=device, hidden_dim=hidden_dim, num_layers=num_layers)
    model.load_state_dict(state_dict)
    model.eval()
    return model

class SearchEnvironment:
    """
    Adapter interface for MCTS.
    Your local training environment can implement this.
    """

    def clone(self):
        raise NotImplementedError

    def current_player_colour(self) -> str:
        raise NotImplementedError

    def observe(self, colour: str) -> Dict[str, Any]:
        raise NotImplementedError

    def step(self, colour: str, action: Tuple[int, int]):
        """
        Returns:
            reward, done, info
        or something similar depending on your env wrapper.
        """
        raise NotImplementedError


@dataclass
class ChildStats:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    child: Optional["MCTSNode"] = None

    @property
    def q(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass
class MCTSNode:
    current_colour: str
    expanded: bool = False
    terminal: bool = False
    terminal_value: float = 0.0
    actions: List[Tuple[int, int]] = None
    children: Dict[Tuple[int, int], ChildStats] = None

    def __post_init__(self):
        if self.actions is None:
            self.actions = []
        if self.children is None:
            self.children = {}

class NeuralMCTS:
    def __init__(self, model: MovePolicyValueNet,
                 graph_builder: BoardGraphBuilder,
                 device: str = "cpu",
                 c_puct: float = 1.5,
                 num_simulations: int = 64):
        
        self.model = model
        self.graph_builder = graph_builder
        self.device = torch.device(device)
        self.c_puct = c_puct
        self.num_simulations = num_simulations

    def _graph_for_observation(self, obs: Dict[str, Any]) -> GraphState:
        gs = self.graph_builder.build(obs)
        return GraphState(
            x=gs.x.to(self.device),
            edge_index=gs.edge_index.to(self.device),
            legal_actions=gs.legal_actions,
            action_features=gs.action_features.to(self.device),
            controlled_colour=gs.controlled_colour,
            meta=gs.meta,
        )

    def _evaluate_value_for_colour(self, env: SearchEnvironment, colour: str) -> float:
        """
        Evaluate the position from `colour`'s perspective.

        This is what makes the MCTS multiplayer-safe: the backed-up value is
        always from the root learner's perspective, not from alternating players.
        """
        obs = env.observe(colour)
        gs = self._graph_for_observation(obs)

        self.model.eval()
        with torch.no_grad():
            _, value = self.model(gs)

        return float(value.detach().cpu())

    def _expand(self, node: MCTSNode, env: SearchEnvironment, root_colour: str) -> float:
        """
        Expand legal actions for node.current_colour, but return value from
        root_colour's perspective.
        """
        obs = env.observe(node.current_colour)
        graph_state = self._graph_for_observation(obs)

        priors, _ = evaluate_graph(self.model, graph_state)
        actions = [(pin_id, to_idx) for pin_id, _, to_idx in graph_state.legal_actions]

        if not actions:
            node.terminal = True
            node.terminal_value = self._evaluate_value_for_colour(env, root_colour)
            node.expanded = True
            return node.terminal_value

        node.actions = actions
        for action, prior in zip(actions, priors.tolist()):
            node.children[action] = ChildStats(prior=prior)

        node.expanded = True
        return self._evaluate_value_for_colour(env, root_colour)

    def _select(self, node: MCTSNode) -> Tuple[int, int]:
        total_visits = sum(ch.visit_count for ch in node.children.values())
        total_visits = max(total_visits, 1)

        best_action = None
        best_score = -float("inf")

        for action, stats in node.children.items():
            u = self.c_puct * stats.prior * math.sqrt(total_visits) / (1 + stats.visit_count)
            score = stats.q + u
            if score > best_score:
                best_score = score
                best_action = action

        if best_action is None:
            raise RuntimeError("MCTS selection failed: no best action")

        return best_action

    def _terminal_value_for_root(
        self,
        *,
        root_colour: str,
        reward: float,
        done: bool,
        info: Dict[str, Any],
        env: SearchEnvironment,
    ) -> float:
        """
        Return terminal value from root_colour's perspective.

        Prefer explicit environment adjudication/progress if available.
        Fall back to the immediate reward only if no better info exists.
        """
        if not done:
            return self._evaluate_value_for_colour(env, root_colour)

        # If LocalSearchEnv implements value_for_colour, use it.
        if hasattr(env, "value_for_colour"):
            return float(env.value_for_colour(root_colour))

        # Fallback for compatibility.
        return float(reward)

    def _simulate(self, env: SearchEnvironment, node: MCTSNode, root_colour: str) -> float:
        if node.terminal:
            return node.terminal_value

        if not node.expanded:
            return self._expand(node, env, root_colour)

        action = self._select(node)
        next_env = env.clone()
        reward, done, info = next_env.step(node.current_colour, action)

        stats = node.children[action]

        if stats.child is None:
            next_colour = next_env.current_player_colour() if not done else node.current_colour
            stats.child = MCTSNode(
                current_colour=next_colour,
                terminal=done,
                terminal_value=0.0,
            )

        if done:
            value = self._terminal_value_for_root(
                root_colour=root_colour,
                reward=reward,
                done=done,
                info=info,
                env=next_env,
            )
            stats.child.terminal_value = value
        else:
            value = self._simulate(next_env, stats.child, root_colour)

        # Critical change:
        # Do NOT negate. Value is already from root_colour's perspective.
        stats.visit_count += 1
        stats.value_sum += value
        return value

    def search(self, env: SearchEnvironment) -> Tuple[Tuple[int, int], List[Tuple[int, int]], torch.Tensor]:
        root_colour = env.current_player_colour()
        root = MCTSNode(current_colour=root_colour)

        for _ in range(self.num_simulations):
            self._simulate(env.clone(), root, root_colour)

        actions = list(root.children.keys())
        visits = torch.tensor([root.children[a].visit_count for a in actions], dtype=torch.float32)

        if visits.numel() == 0:
            raise RuntimeError("MCTS found no legal actions")

        probs = visits / visits.sum()
        best_idx = int(torch.argmax(visits).item())
        return actions[best_idx], actions, probs
