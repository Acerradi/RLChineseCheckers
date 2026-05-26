from __future__ import annotations

import json
import os
import random
import sys
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

for path in (CURRENT_DIR, PARENT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from environment import ChineseCheckersEnv
from evaluate_model_quality import (
    EntrantSpec,
    build_target_spec,
    make_policy,
    target_label,
)


def safe_mean(values: Iterable[float]) -> float:
    xs = list(values)
    return mean(xs) if xs else 0.0


def safe_std(values: Iterable[float]) -> float:
    xs = list(values)
    return pstdev(xs) if len(xs) > 1 else 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: str, payload: Any) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_text(path: str, text: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def opponent_spec(item: Any, index: int) -> EntrantSpec:
    if isinstance(item, str):
        if item not in {"random", "heuristic"}:
            raise ValueError(f"Unknown opponent string: {item}")
        return EntrantSpec(f"{item}_{index}", item)

    if isinstance(item, dict):
        kind = item["kind"]
        name = item.get("name", f"{kind}_{index}")
        checkpoint = item.get("checkpoint")
        return EntrantSpec(name, kind, checkpoint)

    raise TypeError(f"Unsupported opponent spec: {item!r}")


def build_entrants_for_collection(
    *,
    target: EntrantSpec,
    opponent_pattern: List[Any],
    num_players: int,
) -> List[EntrantSpec]:
    if not opponent_pattern:
        raise ValueError("Opponent collection must contain at least one opponent pattern entry")

    entrants = [target]
    opponent_index = 0
    while len(entrants) < num_players:
        pattern_item = opponent_pattern[opponent_index % len(opponent_pattern)]
        entrants.append(opponent_spec(pattern_item, opponent_index))
        opponent_index += 1
    return entrants


def rank_target_by_score(scores: Dict[str, Dict[str, float]], target_colour: str) -> int:
    ordered = sorted(
        ((colour, float(score.get("final_score", 0.0))) for colour, score in scores.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    for index, (colour, _) in enumerate(ordered, start=1):
        if colour == target_colour:
            return index
    return len(ordered)


def run_game(
    *,
    entrants: List[EntrantSpec],
    num_players: int,
    seed: int,
    max_moves: int,
    device: str,
    model_cache: Dict[str, torch.nn.Module],
    hidden_dim: Optional[int],
    num_layers: Optional[int],
    measure_time: bool,
) -> Dict[str, Any]:
    set_seed(seed)

    seated = list(entrants)
    random.Random(seed).shuffle(seated)

    env = ChineseCheckersEnv(
        num_players=num_players,
        player_names=[entrant.name for entrant in seated],
        core_kwargs={"enable_real_time_limits": measure_time},
    )
    env.reset()

    policies_by_colour = {}
    for player in env.game.players:
        spec = next(entrant for entrant in seated if entrant.name == player.name)
        policies_by_colour[player.colour] = make_policy(
            spec,
            device=device,
            seed=seed,
            model_cache=model_cache,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    return env.run_policies(policies_by_colour, max_moves=max_moves)


def extract_score_row(
    *,
    result: Dict[str, Any],
    collection_name: str,
    num_players: int,
    seed: int,
) -> Dict[str, Any]:
    state = result["state"]
    target_player = next(player for player in state["players"] if player["name"] == "target")
    target_colour = target_player["colour"]
    score = result["scores"].get(target_colour, {})
    model_move_count = float(score.get("moves", target_player.get("move_count", 0.0)))

    target_rank = rank_target_by_score(result["scores"], target_colour)

    return {
        "collection": collection_name,
        "num_players": num_players,
        "seed": seed,
        "target_colour": target_colour,
        "target_status": target_player["status"],
        "won": target_player["status"] == "WIN",
        "drawn": target_player["status"] == "DRAW",
        "lost": target_player["status"] not in {"WIN", "DRAW"},
        "final_score_rank": target_rank,
        "final_score": float(score.get("final_score", 0.0)),
        "time_score": float(score.get("time_score", 0.0)),
        "move_score": float(score.get("move_score", 0.0)),
        "pin_score": float(score.get("pin_goal_score", 0.0)),
        "distance_score": float(score.get("distance_score", 0.0)),
        "model_move_count": model_move_count,
        "move_count": model_move_count,
        "game_total_moves": float(state.get("move_count", 0.0)),
        "pins_in_goal": float(score.get("pins_in_goal", 0.0)),
        "total_distance": float(score.get("total_distance", 0.0)),
        "time_taken_sec": float(score.get("time_taken_sec", 0.0)),
        "truncated": bool(result.get("truncated", False)),
        "adjudication_reason": state.get("adjudication_reason"),
        "illegal_attempts": int(result.get("illegal_attempts", 0)),
    }


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"games": 0}

    fields = [
        "final_score",
        "time_score",
        "move_score",
        "pin_score",
        "distance_score",
        "model_move_count",
        "move_count",
        "game_total_moves",
        "pins_in_goal",
        "total_distance",
        "time_taken_sec",
        "final_score_rank",
    ]

    summary: Dict[str, Any] = {
        "games": len(rows),
        "wins": sum(row["won"] for row in rows),
        "draws": sum(row["drawn"] for row in rows),
        "losses": sum(row["lost"] for row in rows),
        "win_rate": safe_mean(1.0 if row["won"] else 0.0 for row in rows),
        "draw_rate": safe_mean(1.0 if row["drawn"] else 0.0 for row in rows),
        "loss_rate": safe_mean(1.0 if row["lost"] else 0.0 for row in rows),
        "truncation_rate": safe_mean(1.0 if row["truncated"] else 0.0 for row in rows),
        "illegal_attempts": sum(row["illegal_attempts"] for row in rows),
    }

    for field in fields:
        values = [float(row[field]) for row in rows]
        summary[f"avg_{field}"] = safe_mean(values)
        summary[f"std_{field}"] = safe_std(values)
        summary[f"min_{field}"] = min(values)
        summary[f"max_{field}"] = max(values)

    reasons: Dict[str, int] = {}
    for row in rows:
        reason = row.get("adjudication_reason") or "NONE"
        reasons[reason] = reasons.get(reason, 0) + 1
    summary["adjudication_reasons"] = reasons

    return summary


def aggregate_group(rows: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return {group: summarize_rows(group_rows) for group, group_rows in sorted(grouped.items())}


def format_summary_markdown(report: Dict[str, Any]) -> str:
    overall = report["overall"]
    lines = [
        "# Score Component Evaluation",
        "",
        f"Target: `{report['target']}`",
        f"Target kind: `{report['target_kind']}`",
        f"Games: {overall['games']}",
        f"Device: `{report['device']}`",
        "",
        "## Overall Averages",
        "",
        "| Games | Total | Time | Move | Pin | Distance | Model Moves | Game Moves | Pins Goal | Total Dist | Time Sec | Rank | Win | Cap |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {overall['games']} | {overall['avg_final_score']:.2f} | "
            f"{overall['avg_time_score']:.2f} | {overall['avg_move_score']:.4f} | "
            f"{overall['avg_pin_score']:.2f} | {overall['avg_distance_score']:.2f} | "
            f"{overall['avg_model_move_count']:.2f} | {overall['avg_game_total_moves']:.2f} | "
            f"{overall['avg_pins_in_goal']:.2f} | "
            f"{overall['avg_total_distance']:.2f} | {overall['avg_time_taken_sec']:.4f} | "
            f"{overall['avg_final_score_rank']:.2f} | {100.0 * overall['win_rate']:.1f}% | "
            f"{100.0 * overall['truncation_rate']:.1f}% |"
        ),
        "",
        "## By Opponent Collection",
        "",
        "| Collection | Games | Total | Time | Move | Pin | Distance | Model Moves | Game Moves | Pins Goal | Total Dist | Time Sec | Rank | Win | Cap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for collection, data in report["by_collection"].items():
        lines.append(
            f"| `{collection}` | {data['games']} | {data['avg_final_score']:.2f} | "
            f"{data['avg_time_score']:.2f} | {data['avg_move_score']:.4f} | "
            f"{data['avg_pin_score']:.2f} | {data['avg_distance_score']:.2f} | "
            f"{data['avg_model_move_count']:.2f} | {data['avg_game_total_moves']:.2f} | "
            f"{data['avg_pins_in_goal']:.2f} | "
            f"{data['avg_total_distance']:.2f} | {data['avg_time_taken_sec']:.4f} | "
            f"{data['avg_final_score_rank']:.2f} | {100.0 * data['win_rate']:.1f}% | "
            f"{100.0 * data['truncation_rate']:.1f}% |"
        )

    lines.extend([
        "",
        "## By Player Count",
        "",
        "| Players | Games | Total | Time | Move | Pin | Distance | Model Moves | Game Moves | Pins Goal | Total Dist | Time Sec | Rank | Win | Cap |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for players, data in report["by_player_count"].items():
        lines.append(
            f"| {players} | {data['games']} | {data['avg_final_score']:.2f} | "
            f"{data['avg_time_score']:.2f} | {data['avg_move_score']:.4f} | "
            f"{data['avg_pin_score']:.2f} | {data['avg_distance_score']:.2f} | "
            f"{data['avg_model_move_count']:.2f} | {data['avg_game_total_moves']:.2f} | "
            f"{data['avg_pins_in_goal']:.2f} | "
            f"{data['avg_total_distance']:.2f} | {data['avg_time_taken_sec']:.4f} | "
            f"{data['avg_final_score_rank']:.2f} | {100.0 * data['win_rate']:.1f}% | "
            f"{100.0 * data['truncation_rate']:.1f}% |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    TARGETKIND = "checkpoint"  # "checkpoint", "heuristic", or "random"
    TARGETCHECKPOINT = "checkpoints/gnn_h128_l4/self_play/shared_model_final.pt"

    OPPONENTCOLLECTIONS = [
        {"name": "all_random", "pattern": ["random"]},
        {"name": "all_heuristic", "pattern": ["heuristic"]},
        {"name": "mixed_random_heuristic", "pattern": ["heuristic", "random"]},
        {"name": "bootstrap_checkpoint", "pattern": [
            {"kind": "checkpoint",
             "name": "bootstrap_final",
             "checkpoint": "checkpoints/gnn_h128_l4/bootstrap/shared_model_final.pt"},
            "heuristic",
        ]},
        {"name": "champion_checkpoint", "pattern": [
            {"kind": "checkpoint",
             "name": "champion",
             "checkpoint": "checkpoints/gnn_h128_l4/self_play/champion.pt"},
            "heuristic",
        ]},
    ]

    PLAYERS = [2, 3, 4, 5, 6]
    GAMES = 20
    MAXMOVES = 300
    SEED = 5000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    HIDDENDIM = None
    NUMLAYERS = None

    # If True, the core records wall-clock action-selection time per player,
    # making time_score meaningful in local evaluation.
    MEASURETIME = True

    SUMMARYOUT = "eval_reports/score_component_summary.md"
    SUMMARYJSONOUT = "eval_reports/score_component_summary.json"
    JSONLOUT = "eval_reports/score_component_games.jsonl"
    WRITEJSONL = True
    QUIET = False

    if TARGETKIND == "checkpoint" and not TARGETCHECKPOINT:
        raise ValueError("TARGETCHECKPOINT must be set for checkpoint targets")
    if TARGETKIND == "checkpoint" and not os.path.exists(TARGETCHECKPOINT):
        raise FileNotFoundError(f"Target checkpoint not found: {TARGETCHECKPOINT}")

    for collection in OPPONENTCOLLECTIONS:
        for item in collection["pattern"]:
            if isinstance(item, dict) and item.get("kind") == "checkpoint":
                checkpoint = item.get("checkpoint")
                if not checkpoint or not os.path.exists(checkpoint):
                    raise FileNotFoundError(f"Opponent checkpoint not found: {checkpoint}")

    target = build_target_spec(TARGETKIND, TARGETCHECKPOINT)
    target_display = target_label(TARGETKIND, TARGETCHECKPOINT)

    model_cache: Dict[str, torch.nn.Module] = {}
    rows: List[Dict[str, Any]] = []
    scenario_index = 0

    for num_players in PLAYERS:
        for collection in OPPONENTCOLLECTIONS:
            entrants = build_entrants_for_collection(
                target=target,
                opponent_pattern=collection["pattern"],
                num_players=num_players,
            )

            if not QUIET:
                print(f"\n[{collection['name']}] players={num_players} games={GAMES}")

            for game_index in range(GAMES):
                seed = SEED + scenario_index * 10000 + game_index
                result = run_game(
                    entrants=entrants,
                    num_players=num_players,
                    seed=seed,
                    max_moves=MAXMOVES,
                    device=DEVICE,
                    model_cache=model_cache,
                    hidden_dim=HIDDENDIM,
                    num_layers=NUMLAYERS,
                    measure_time=MEASURETIME,
                )
                row = extract_score_row(
                    result=result,
                    collection_name=collection["name"],
                    num_players=num_players,
                    seed=seed,
                )
                rows.append(row)

                if not QUIET:
                    print(
                        f"  game {game_index + 1:>3}/{GAMES}: "
                        f"total={row['final_score']:.2f} "
                        f"time={row['time_score']:.2f} "
                        f"move={row['move_score']:.4f} "
                        f"model_moves={row['model_move_count']:.0f} "
                        f"game_moves={row['game_total_moves']:.0f} "
                        f"pin={row['pin_score']:.1f} "
                        f"dist={row['distance_score']:.1f} "
                        f"rank={row['final_score_rank']}"
                    )

            scenario_index += 1

    report = {
        "target": target_display,
        "target_kind": TARGETKIND,
        "target_checkpoint": TARGETCHECKPOINT if TARGETKIND == "checkpoint" else None,
        "device": DEVICE,
        "players": PLAYERS,
        "games_per_collection_player_count": GAMES,
        "max_moves": MAXMOVES,
        "measure_time": MEASURETIME,
        "opponent_collections": OPPONENTCOLLECTIONS,
        "overall": summarize_rows(rows),
        "by_collection": aggregate_group(rows, "collection"),
        "by_player_count": aggregate_group(rows, "num_players"),
    }

    write_text(SUMMARYOUT, format_summary_markdown(report))
    write_json(SUMMARYJSONOUT, report)
    if WRITEJSONL:
        write_jsonl(JSONLOUT, rows)

    print("\nSCORE COMPONENT SUMMARY")
    print(f"Target: {target_display}")
    print(f"Games: {report['overall']['games']}")
    print(f"Average total score: {report['overall']['avg_final_score']:.2f}")
    print(f"Average time score: {report['overall']['avg_time_score']:.2f}")
    print(f"Average move score: {report['overall']['avg_move_score']:.4f}")
    print(f"Average model moves: {report['overall']['avg_model_move_count']:.2f}")
    print(f"Average total game moves: {report['overall']['avg_game_total_moves']:.2f}")
    print(f"Average pin score: {report['overall']['avg_pin_score']:.2f}")
    print(f"Average distance score: {report['overall']['avg_distance_score']:.2f}")
    print(f"Saved summary: {SUMMARYOUT}")
    print(f"Saved JSON summary: {SUMMARYJSONOUT}")
    if WRITEJSONL:
        print(f"Saved per-game rows: {JSONLOUT}")


if __name__ == "__main__":
    main()
