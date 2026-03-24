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

    def build(self, observation: Dict[str, Any]) -> GraphState:
        state = observation["state"]
        legal_moves = observation["legal_moves"]
        controlled_colour = observation["colour"]

        occ = self._occupancy_map(state)
        my_target_zone = self.board.colour_opposites[controlled_colour]

        x_rows = []
        for idx, cell in enumerate(self.board.cells):
            row = []

            occupant = occ.get(idx, "empty")
            occ_onehot = [0.0] * len(OCCUPANT_TYPES)
            occ_onehot[OCC_TO_IDX[occupant]] = 1.0
            row.extend(occ_onehot)

            zone = getattr(cell, "postype", "board")
            zone_onehot = [0.0] * len(ZONE_TYPES)
            zone_onehot[ZONE_TO_IDX.get(zone, 0)] = 1.0
            row.extend(zone_onehot)

            row.append(1.0 if occupant == controlled_colour else 0.0)
            row.append(1.0 if zone == my_target_zone else 0.0)
            row.append(1.0 if state.get("current_turn_colour") == controlled_colour else 0.0)
            row.append(min(state.get("move_count", 0) / 200.0, 1.0))

            x_rows.append(row)

        x = torch.tensor(x_rows, dtype=torch.float32)

        my_positions = state["pins"][controlled_colour]
        pin_id_to_from = {int(pin_id): my_positions[int(pin_id)] for pin_id in legal_moves.keys()}

        legal_actions: List[Tuple[int, int, int]] = []
        for pin_id, to_list in legal_moves.items():
            pid = int(pin_id)
            from_idx = pin_id_to_from[pid]
            for to_idx in to_list:
                legal_actions.append((pid, from_idx, int(to_idx)))

        return GraphState(
            x=x,
            edge_index=self.edge_index,
            legal_actions=legal_actions,
            controlled_colour=controlled_colour,
            meta=state,
        )


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
        return min(axial_dist(here, tgt) for tgt in self._target_cells(colour))

    def _score_action(self, colour: str, from_idx: int, to_idx: int, state: Dict[str, Any]) -> float:
        """Heuristic scoring of a potential move for the given colour."""
        before = self._min_dist_to_goal(from_idx, colour)
        after = self._min_dist_to_goal(to_idx, colour)

        progress_gain = before - after
        jump_distance = axial_dist(self.board.cells[from_idx], self.board.cells[to_idx])

        target_zone = self.board.colour_opposites[colour]
        from_zone = getattr(self.board.cells[from_idx], "postype", "board")
        to_zone = getattr(self.board.cells[to_idx], "postype", "board")

        # Main priorities
        score = 0.0
        score += 10.0 * progress_gain          # move toward goal
        score += 2.5 * max(0, jump_distance-1) # prefer larger jumps
        score += 4.0 if to_zone == target_zone else 0.0 # prefer entering target zone

        # Avoid undoing progress
        if progress_gain < 0:
            score += 6.0 * progress_gain       # stronger penalty if move goes backward

        # Avoid leaving target zone once entered
        if from_zone == target_zone and to_zone != target_zone:
            score -= 15.0

        # Small preference for central mobility / non-trivial motion
        score += 0.25 * jump_distance

        # Tiny noise for exploration / tie-breaking
        score += self.rng.uniform(-self.epsilon, self.epsilon)

        return score

    def select_action(self, observation: Dict[str, Any]) -> Tuple[int, int]:
        """Select a move by scoring all legal moves with the heuristic and picking the best."""
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
    """
    GNN encoder + legal-move policy head + scalar value head.
    """

    def __init__(self, node_feat_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = GraphEncoder(node_feat_dim, hidden_dim=hidden_dim, num_layers=4)

        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.policy_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.value_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def encode(self, graph_state: GraphState) -> Tuple[torch.Tensor, torch.Tensor]:
        node_emb = self.encoder(graph_state.x, graph_state.edge_index)
        global_emb = self.global_proj(node_emb.mean(dim=0))
        return node_emb, global_emb

    def forward(self, graph_state: GraphState) -> Tuple[torch.Tensor, torch.Tensor]:
        node_emb, global_emb = self.encode(graph_state)

        scores = []
        for _, from_idx, to_idx in graph_state.legal_actions:
            feat = torch.cat(
                [node_emb[from_idx], node_emb[to_idx], global_emb],
                dim=-1,
            )
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
                 mcts_simulations: int = 64):
        
        self.device = torch.device(device)
        self.board = board or HexBoard()
        self.graph_builder = BoardGraphBuilder(self.board)

        if model is None:
            node_feat_dim = len(OCCUPANT_TYPES) + len(ZONE_TYPES) + 4
            model = MovePolicyValueNet(node_feat_dim=node_feat_dim, hidden_dim=128)

        self.model = model.to(self.device)
        self.model.eval()

        self.use_mcts = use_mcts
        self.mcts_simulations = mcts_simulations

    @torch.no_grad()
    def select_action(self, observation: Dict[str, Any]) -> Tuple[int, int]:
        """
        Live deployment path.
        Uses policy head only, because the live socket setup does not naturally
        give us a clonable in-memory environment for tree search.
        """
        graph_state = self.graph_builder.build(observation)
        graph_state = GraphState(
            x=graph_state.x.to(self.device),
            edge_index=graph_state.edge_index.to(self.device),
            legal_actions=graph_state.legal_actions,
            controlled_colour=graph_state.controlled_colour,
            meta=graph_state.meta,
        )

        logits, value = self.model(graph_state)

        if logits.numel() == 0:
            raise RuntimeError("No legal moves available")

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


def build_model(device: str = "cpu", hidden_dim: int = 128) -> MovePolicyValueNet:
    node_feat_dim = len(OCCUPANT_TYPES) + len(ZONE_TYPES) + 4
    model = MovePolicyValueNet(node_feat_dim=node_feat_dim, hidden_dim=hidden_dim)
    return model.to(device)

def save_model(model: MovePolicyValueNet, path: str) -> None:
    torch.save(model.state_dict(), path)

def load_model(path: str, device: str = "cpu", hidden_dim: int = 128) -> MovePolicyValueNet:
    model = build_model(device=device, hidden_dim=hidden_dim)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
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
    def __init__(
        self,
        model: MovePolicyValueNet,
        graph_builder: BoardGraphBuilder,
        device: str = "cpu",
        c_puct: float = 1.5,
        num_simulations: int = 64,
    ):
        self.model = model
        self.graph_builder = graph_builder
        self.device = torch.device(device)
        self.c_puct = c_puct
        self.num_simulations = num_simulations

    def _expand(self, node: MCTSNode, env: SearchEnvironment) -> float:
        obs = env.observe(node.current_colour)
        graph_state = self.graph_builder.build(obs)
        graph_state = GraphState(
            x=graph_state.x.to(self.device),
            edge_index=graph_state.edge_index.to(self.device),
            legal_actions=graph_state.legal_actions,
            controlled_colour=graph_state.controlled_colour,
            meta=graph_state.meta,
        )

        priors, value = evaluate_graph(self.model, graph_state)
        actions = [(pin_id, to_idx) for pin_id, _, to_idx in graph_state.legal_actions]

        if not actions:
            node.terminal = True
            node.terminal_value = value
            node.expanded = True
            return value

        node.actions = actions
        for action, prior in zip(actions, priors.tolist()):
            node.children[action] = ChildStats(prior=prior)

        node.expanded = True
        return value

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

        return best_action

    def _simulate(self, env: SearchEnvironment, node: MCTSNode) -> float:
        if node.terminal:
            return node.terminal_value

        if not node.expanded:
            return self._expand(node, env)

        action = self._select(node)
        next_env = env.clone()
        reward, done, info = next_env.step(node.current_colour, action)

        stats = node.children[action]

        if stats.child is None:
            next_colour = next_env.current_player_colour() if not done else node.current_colour
            stats.child = MCTSNode(current_colour=next_colour, terminal=done, terminal_value=reward if done else 0.0)

        child_value = reward if done else self._simulate(next_env, stats.child)

        # simple alternating-sign backup; workable first approximation
        backed_up = -child_value

        stats.visit_count += 1
        stats.value_sum += backed_up
        return backed_up

    def search(self, env: SearchEnvironment) -> Tuple[Tuple[int, int], List[Tuple[int, int]], torch.Tensor]:
        root = MCTSNode(current_colour=env.current_player_colour())

        for _ in range(self.num_simulations):
            self._simulate(env.clone(), root)

        actions = list(root.children.keys())
        visits = torch.tensor([root.children[a].visit_count for a in actions], dtype=torch.float32)

        if visits.numel() == 0:
            raise RuntimeError("MCTS found no legal actions")

        probs = visits / visits.sum()
        best_idx = int(torch.argmax(visits).item())
        return actions[best_idx], actions, probs


