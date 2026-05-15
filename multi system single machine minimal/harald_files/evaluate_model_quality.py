from __future__ import annotations

import glob
import json
import os
import random
import sys
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

for path in (CURRENT_DIR, PARENT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from environment import ChineseCheckersEnv
from policies import RandomPolicy
from policy_template import HeuristicPolicy, MyPolicy, build_model


@dataclass(frozen=True)
class EntrantSpec:
    name: str
    kind: str
    checkpoint: Optional[str] = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_mean(values: Iterable[float]) -> float:
    xs = list(values)
    return mean(xs) if xs else 0.0


def safe_std(values: Iterable[float]) -> float:
    xs = list(values)
    return pstdev(xs) if len(xs) > 1 else 0.0


def load_model_auto(
    checkpoint: str,
    *,
    device: str,
    hidden_dim: Optional[int] = None,
    num_layers: Optional[int] = None,
) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location=device)

    if isinstance(payload, dict) and "model_state_dict" in payload:
        resolved_hidden = hidden_dim or int(payload.get("hidden_dim", 128))
        resolved_layers = num_layers or int(payload.get("num_layers", 4))
        state_dict = payload["model_state_dict"]
    else:
        resolved_hidden = hidden_dim or 128
        resolved_layers = num_layers or 4
        state_dict = payload

    model = build_model(device=device, hidden_dim=resolved_hidden, num_layers=resolved_layers)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def make_policy(spec: EntrantSpec,*,
                device: str,
                seed: int,
                model_cache: Dict[str, torch.nn.Module],
                hidden_dim: Optional[int],
                num_layers: Optional[int]):
    if spec.kind == "random":
        return RandomPolicy(seed=seed)

    if spec.kind == "heuristic":
        return HeuristicPolicy(epsilon=0.0)

    if spec.kind == "checkpoint":
        if spec.checkpoint is None:
            raise ValueError(f"Checkpoint entrant {spec.name} has no checkpoint path")
        if spec.checkpoint not in model_cache:
            model_cache[spec.checkpoint] = load_model_auto(
                spec.checkpoint,
                device=device,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )
        return MyPolicy(model=model_cache[spec.checkpoint], device=device)

    raise ValueError(f"Unknown entrant kind: {spec.kind}")


def expand_checkpoint_args(values: List[str]) -> List[str]:
    paths: List[str] = []
    for value in values:
        if os.path.isdir(value):
            paths.extend(glob.glob(os.path.join(value, "*.pt")))
        else:
            matched = glob.glob(value)
            paths.extend(matched if matched else [value])
    return sorted(dict.fromkeys(paths))


def build_scenario_entrants(*,
                            target_checkpoint: str,
                            scenario: str,
                            num_players: int,
                            opponent_checkpoint: Optional[str] = None,
                            fill: str = "heuristic") -> List[EntrantSpec]:
    target = EntrantSpec("target", "checkpoint", target_checkpoint)

    if scenario == "random":
        return [target] + [EntrantSpec(f"random_{i}", "random") for i in range(num_players - 1)]

    if scenario == "heuristic":
        return [target] + [EntrantSpec(f"heuristic_{i}", "heuristic") for i in range(num_players - 1)]

    if scenario == "mixed":
        entrants = [target]
        for i in range(num_players - 1):
            kind = "heuristic" if i % 2 == 0 else "random"
            entrants.append(EntrantSpec(f"{kind}_{i}", kind))
        return entrants

    if scenario == "checkpoint":
        if opponent_checkpoint is None:
            raise ValueError("checkpoint scenario requires opponent_checkpoint")
        entrants = [target, EntrantSpec("opponent", "checkpoint", opponent_checkpoint)]
        while len(entrants) < num_players:
            i = len(entrants)
            entrants.append(EntrantSpec(f"{fill}_{i}", fill))
        return entrants

    raise ValueError(f"Unknown scenario: {scenario}")


def player_metric(state: Dict[str, Any], colour: str, metric: str, default: float = 0.0) -> float:
    values = state.get(metric, {})
    if not isinstance(values, dict):
        return default
    try:
        return float(values.get(colour, default))
    except (TypeError, ValueError):
        return default


def rank_colour(values_by_colour: Dict[str, float], colour: str) -> int:
    ordered = sorted(values_by_colour.items(), key=lambda item: item[1], reverse=True)
    for index, (candidate_colour, _) in enumerate(ordered, start=1):
        if candidate_colour == colour:
            return index
    return len(ordered)


def placement_score(rank: int, num_players: int) -> float:
    if num_players <= 1:
        return 1.0
    return max(0.0, min(1.0, (num_players - rank) / (num_players - 1)))


def extract_game_metrics(*,
                         result: Dict[str, Any],
                         target_name: str,
                         seed: int,
                         scenario: str,
                         num_players: int) -> Dict[str, Any]:
    state = result["state"]
    target_player = next(player for player in state["players"] if player["name"] == target_name)
    colour = target_player["colour"]
    score = result["scores"].get(colour, {})
    progress_by_colour = {player["colour"]: player_metric(state, player["colour"], "training_progress")
                          for player in state["players"]}
    final_score_by_colour = {player["colour"]: float(result["scores"].get(player["colour"], {}).get("final_score", 0.0))
                             for player in state["players"]}
    progress_rank = rank_colour(progress_by_colour, colour)
    final_score_rank = rank_colour(final_score_by_colour, colour)
    progress_place = placement_score(progress_rank, num_players)
    final_score_place = placement_score(final_score_rank, num_players)
    performance_score = 0.70 * progress_place + 0.30 * final_score_place

    return {
        "seed": seed,
        "scenario": scenario,
        "num_players": num_players,
        "target_colour": colour,
        "target_status": target_player["status"],
        "won": target_player["status"] == "WIN",
        "drawn": target_player["status"] == "DRAW",
        "lost": target_player["status"] not in ("WIN", "DRAW"),
        "expected_random_win_rate": 1.0 / num_players,
        "chance_adjusted_win_value": float(num_players if target_player["status"] == "WIN" else 0.0),
        "chance_adjusted_loss_value": float(1.0 / (1.0 - (1.0 / num_players))
                                            if target_player["status"] not in ("WIN", "DRAW") else 0.0),
        "final_score": float(score.get("final_score", 0.0)),
        "pin_goal_score": float(score.get("pin_goal_score", 0.0)),
        "distance_score": float(score.get("distance_score", 0.0)),
        "move_score": float(score.get("move_score", 0.0)),
        "target_moves": float(score.get("moves", 0.0)),
        "pins_in_goal": float(score.get("pins_in_goal", 0.0)),
        "total_distance": float(score.get("total_distance", 0.0)),
        "training_progress": player_metric(state, colour, "training_progress"),
        "progress_rank": progress_rank,
        "progress_placement_score": progress_place,
        "final_score_rank": final_score_rank,
        "final_score_placement_score": final_score_place,
        "performance_score": performance_score,
        "top_half_by_progress": progress_rank <= ((num_players + 1) // 2),
        "top_two_by_progress": progress_rank <= min(2, num_players),
        "home_pieces": player_metric(state, colour, "home_pieces"),
        "stranded_home_pieces": player_metric(state, colour, "stranded_home_pieces"),
        "move_count": int(state.get("move_count", 0)),
        "finished": state.get("status") == "FINISHED",
        "truncated": bool(result.get("truncated", False)),
        "adjudication_reason": state.get("adjudication_reason"),
        "illegal_attempts": int(result.get("illegal_attempts", 0)),
    }


def summarize_games(games: List[Dict[str, Any]]) -> Dict[str, Any]:
    count = len(games)
    if count == 0:
        return {"games": 0}

    numeric_fields = [
        "expected_random_win_rate",
        "chance_adjusted_win_value",
        "chance_adjusted_loss_value",
        "final_score",
        "pin_goal_score",
        "distance_score",
        "move_score",
        "target_moves",
        "pins_in_goal",
        "total_distance",
        "training_progress",
        "progress_rank",
        "progress_placement_score",
        "final_score_rank",
        "final_score_placement_score",
        "performance_score",
        "home_pieces",
        "stranded_home_pieces",
        "move_count",
    ]

    summary: Dict[str, Any] = {
        "games": count,
        "wins": sum(game["won"] for game in games),
        "draws": sum(game["drawn"] for game in games),
        "losses": sum(game["lost"] for game in games),
        "win_rate": safe_mean(1.0 if game["won"] else 0.0 for game in games),
        "draw_rate": safe_mean(1.0 if game["drawn"] else 0.0 for game in games),
        "loss_rate": safe_mean(1.0 if game["lost"] else 0.0 for game in games),
        "top_half_rate": safe_mean(1.0 if game.get("top_half_by_progress") else 0.0 for game in games),
        "top_two_rate": safe_mean(1.0 if game.get("top_two_by_progress") else 0.0 for game in games),
        "truncation_rate": safe_mean(1.0 if game["truncated"] else 0.0 for game in games),
        "illegal_attempts": sum(int(game["illegal_attempts"]) for game in games),
    }

    for field in numeric_fields:
        values = [float(game.get(field, 0.0)) for game in games]
        summary[f"avg_{field}"] = safe_mean(values)
        summary[f"std_{field}"] = safe_std(values)

    expected_win = summary["avg_expected_random_win_rate"]
    expected_loss = max(1.0 - expected_win, 0.000001)
    summary["chance_adjusted_win_rate"] = summary["win_rate"] / expected_win if expected_win > 0 else 0.0
    summary["chance_adjusted_loss_rate"] = summary["loss_rate"] / expected_loss

    reasons: Dict[str, int] = {}
    for game in games:
        reason = game.get("adjudication_reason") or "NONE"
        reasons[reason] = reasons.get(reason, 0) + 1
    summary["adjudication_reasons"] = reasons

    return summary


def run_game(*,
             entrant_specs: List[EntrantSpec],
             target_name: str,
             num_players: int,
             seed: int,
             max_moves: int,
             device: str,
             model_cache: Dict[str, torch.nn.Module],
             hidden_dim: Optional[int],
             num_layers: Optional[int]) -> Dict[str, Any]:
    set_seed(seed)

    entrants = list(entrant_specs)
    random.Random(seed).shuffle(entrants)

    env = ChineseCheckersEnv(num_players=num_players, player_names=[entrant.name for entrant in entrants])
    env.reset()

    policies_by_colour = {}
    for player in env.game.players:
        spec = next(entrant for entrant in entrants if entrant.name == player.name)
        policies_by_colour[player.colour] = make_policy(spec,
                                                        device=device,
                                                        seed=seed,
                                                        model_cache=model_cache,
                                                        hidden_dim=hidden_dim,
                                                        num_layers=num_layers)

    return env.run_policies(policies_by_colour, max_moves=max_moves)


def evaluate_scenario(*,
                      scenario_label: str,
                      entrant_specs: List[EntrantSpec],
                      num_players: int,
                      games: int,
                      seed_base: int,
                      max_moves: int,
                      device: str,
                      model_cache: Dict[str, torch.nn.Module],
                      hidden_dim: Optional[int],
                      num_layers: Optional[int],
                      verbose: bool) -> Dict[str, Any]:
    game_metrics: List[Dict[str, Any]] = []

    if verbose:
        print(f"\n[{scenario_label}] players={num_players} games={games}")

    for game_idx in range(games):
        seed = seed_base + game_idx
        result = run_game(entrant_specs=entrant_specs,
                          target_name="target",
                          num_players=num_players,
                          seed=seed,
                          max_moves=max_moves,
                          device=device,
                          model_cache=model_cache,
                          hidden_dim=hidden_dim,
                          num_layers=num_layers)
        metrics = extract_game_metrics(result=result,
                                       target_name="target",
                                       seed=seed,
                                       scenario=scenario_label,
                                       num_players=num_players)
        game_metrics.append(metrics)

        if verbose:
            print(
                f"  game {game_idx + 1:>3}/{games}: "
                f"{metrics['target_status']:<7} score={metrics['final_score']:.1f} "
                f"progress={metrics['training_progress']:.1f} "
                f"moves={metrics['move_count']} cap={metrics['truncated']}"
            )

    return {
        "scenario": scenario_label,
        "num_players": num_players,
        "entrants": [entrant.__dict__ for entrant in entrant_specs],
        "summary": summarize_games(game_metrics),
        "games": game_metrics,
    }


def write_json(path: str, payload: Dict[str, Any]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_jsonl(path: str, scenarios: List[Dict[str, Any]]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for scenario in scenarios:
            for game in scenario["games"]:
                handle.write(json.dumps(game) + "\n")


def write_text(path: str, text: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def aggregate_summaries(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_games = sum(item["summary"]["games"] for item in items)
    if total_games == 0:
        return {"games": 0}

    def weighted_average(field: str) -> float:
        return sum(item["summary"].get(field, 0.0) * item["summary"]["games"] for item in items) / total_games

    return {
        "games": total_games,
        "win_rate": weighted_average("win_rate"),
        "draw_rate": weighted_average("draw_rate"),
        "loss_rate": weighted_average("loss_rate"),
        "chance_adjusted_win_rate": weighted_average("chance_adjusted_win_rate"),
        "chance_adjusted_loss_rate": weighted_average("chance_adjusted_loss_rate"),
        "top_half_rate": weighted_average("top_half_rate"),
        "top_two_rate": weighted_average("top_two_rate"),
        "truncation_rate": weighted_average("truncation_rate"),
        "avg_final_score": weighted_average("avg_final_score"),
        "avg_training_progress": weighted_average("avg_training_progress"),
        "avg_progress_rank": weighted_average("avg_progress_rank"),
        "avg_progress_placement_score": weighted_average("avg_progress_placement_score"),
        "avg_final_score_rank": weighted_average("avg_final_score_rank"),
        "avg_final_score_placement_score": weighted_average("avg_final_score_placement_score"),
        "avg_performance_score": weighted_average("avg_performance_score"),
        "avg_pins_in_goal": weighted_average("avg_pins_in_goal"),
        "avg_total_distance": weighted_average("avg_total_distance"),
        "avg_home_pieces": weighted_average("avg_home_pieces"),
        "avg_stranded_home_pieces": weighted_average("avg_stranded_home_pieces"),
        "illegal_attempts": sum(item["summary"]["illegal_attempts"] for item in items),
    }


def compact_row(item: Dict[str, Any]) -> Dict[str, Any]:
    summary = item["summary"]
    return {
        "scenario": item["scenario"],
        "players": item["num_players"],
        "games": summary["games"],
        "win_rate": round(summary["win_rate"], 3),
        "chance_adjusted_win_rate": round(summary.get("chance_adjusted_win_rate", 0.0), 3),
        "loss_rate": round(summary["loss_rate"], 3),
        "cap_rate": round(summary["truncation_rate"], 3),
        "placement": round(summary.get("avg_progress_placement_score", 0.0), 3),
        "performance_score": round(summary.get("avg_performance_score", 0.0), 3),
        "avg_rank": round(summary.get("avg_progress_rank", 0.0), 2),
        "top_half_rate": round(summary.get("top_half_rate", 0.0), 3),
        "avg_score": round(summary["avg_final_score"], 1),
        "avg_progress": round(summary["avg_training_progress"], 1),
        "avg_pins_goal": round(summary["avg_pins_in_goal"], 2),
        "avg_distance": round(summary["avg_total_distance"], 1),
    }


def quality_label(overall: Dict[str, Any], baseline: Dict[str, Any], checkpoint: Dict[str, Any]) -> str:
    baseline_adjusted_win = baseline.get("chance_adjusted_win_rate", 0.0)
    checkpoint_adjusted_win = checkpoint.get("chance_adjusted_win_rate", 0.0)
    performance = overall.get("avg_performance_score", 0.0)
    cap_rate = overall.get("truncation_rate", 1.0)
    illegal_attempts = overall.get("illegal_attempts", 0)

    if illegal_attempts > 0:
        return "Invalid: one or more illegal moves were attempted"
    if baseline_adjusted_win >= 1.50 and checkpoint_adjusted_win >= 1.05 and performance >= 0.65 and cap_rate <= 0.20:
        return "Strong"
    if baseline_adjusted_win >= 1.20 and checkpoint_adjusted_win >= 0.90 and performance >= 0.55 and cap_rate <= 0.35:
        return "Good"
    if baseline_adjusted_win >= 1.00 and performance >= 0.45:
        return "Promising but uneven"
    if baseline_adjusted_win >= 0.85 or performance >= 0.40:
        return "Baseline-capable, weak against trained opponents"
    return "Weak"


def build_compact_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    results = report["results"]
    baseline_items = [item for item in results if item["scenario"] in {"vs_random", "vs_heuristic", "vs_mixed"}]
    checkpoint_items = [item for item in results if item["scenario"].startswith("vs_checkpoint:")]

    overall = aggregate_summaries(results)
    baseline = aggregate_summaries(baseline_items)
    checkpoint = aggregate_summaries(checkpoint_items)

    rows = [compact_row(item) for item in results]
    best = max(rows, key=lambda row: (row["win_rate"], row["avg_progress"]), default=None)
    worst = min(rows, key=lambda row: (row["win_rate"], row["avg_progress"]), default=None)

    by_scenario = {}
    for scenario in sorted({item["scenario"] for item in results}):
        by_scenario[scenario] = aggregate_summaries([item for item in results if item["scenario"] == scenario])

    by_player_count = {}
    for players in sorted({item["num_players"] for item in results}):
        by_player_count[str(players)] = aggregate_summaries([item for item in results if item["num_players"] == players])

    return {
        "target_checkpoint": report["target_checkpoint"],
        "quality_label": quality_label(overall, baseline, checkpoint),
        "games_total": overall.get("games", 0),
        "overall": overall,
        "baseline_opponents": baseline,
        "checkpoint_opponents": checkpoint,
        "by_scenario": by_scenario,
        "by_player_count": by_player_count,
        "best_case": best,
        "worst_case": worst,
        "compact_rows": rows,
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def format_compact_markdown(summary: Dict[str, Any]) -> str:
    overall = summary["overall"]
    baseline = summary["baseline_opponents"]
    checkpoint = summary["checkpoint_opponents"]

    lines = [
        "# Model Quality Summary",
        "",
        f"Target: `{summary['target_checkpoint']}`",
        f"Overall quality: **{summary['quality_label']}**",
        f"Total games: {summary['games_total']}",
        "",
        "## Headline",
        "",
        f"- Overall win/draw/loss: {pct(overall.get('win_rate', 0.0))} / "
        f"{pct(overall.get('draw_rate', 0.0))} / {pct(overall.get('loss_rate', 0.0))}",
        f"- Chance-adjusted overall win rate: {overall.get('chance_adjusted_win_rate', 0.0):.2f}x random expectation",
        f"- Baseline win rate: {pct(baseline.get('win_rate', 0.0))}",
        f"- Chance-adjusted baseline win rate: {baseline.get('chance_adjusted_win_rate', 0.0):.2f}x random expectation",
        f"- Checkpoint-opponent win rate: {pct(checkpoint.get('win_rate', 0.0))}",
        f"- Chance-adjusted checkpoint win rate: {checkpoint.get('chance_adjusted_win_rate', 0.0):.2f}x random expectation",
        f"- Average progress rank: {overall.get('avg_progress_rank', 0.0):.2f}",
        f"- Progress placement score: {overall.get('avg_progress_placement_score', 0.0):.3f} "
        "(1.0 first place, 0.0 last place)",
        f"- Overall performance score: {overall.get('avg_performance_score', 0.0):.3f} "
        "(blends progress rank and final-score rank)",
        f"- Top-half by progress rate: {pct(overall.get('top_half_rate', 0.0))}",
        f"- Move-cap/adjudication rate: {pct(overall.get('truncation_rate', 0.0))}",
        f"- Average score: {overall.get('avg_final_score', 0.0):.1f}",
        f"- Average training progress: {overall.get('avg_training_progress', 0.0):.1f}",
        f"- Average pins in goal: {overall.get('avg_pins_in_goal', 0.0):.2f}",
        f"- Average remaining goal distance: {overall.get('avg_total_distance', 0.0):.1f}",
        f"- Illegal attempts: {overall.get('illegal_attempts', 0)}",
        "",
    ]

    if summary["best_case"] is not None:
        best = summary["best_case"]
        lines.extend([
            "## Best And Weakest Cases",
            "",
            f"- Best: `{best['scenario']}` with {best['players']} players, "
            f"win rate {pct(best['win_rate'])}, adjusted win {best['chance_adjusted_win_rate']:.2f}x, "
            f"performance {best['performance_score']:.3f}",
        ])

    if summary["worst_case"] is not None:
        worst = summary["worst_case"]
        lines.append(
            f"- Weakest: `{worst['scenario']}` with {worst['players']} players, "
            f"win rate {pct(worst['win_rate'])}, adjusted win {worst['chance_adjusted_win_rate']:.2f}x, "
            f"performance {worst['performance_score']:.3f}"
        )
        lines.append("")

    lines.extend([
        "## Scenario Averages",
        "",
        "| Scenario | Games | Win | Adj Win | Avg Rank | Place | Perf | Top Half | Loss | Cap | Score | Progress |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for scenario, data in summary["by_scenario"].items():
        lines.append(
            f"| `{scenario}` | {data.get('games', 0)} | {pct(data.get('win_rate', 0.0))} | "
            f"{data.get('chance_adjusted_win_rate', 0.0):.2f}x | "
            f"{data.get('avg_progress_rank', 0.0):.2f} | {data.get('avg_progress_placement_score', 0.0):.3f} | "
            f"{data.get('avg_performance_score', 0.0):.3f} | {pct(data.get('top_half_rate', 0.0))} | "
            f"{pct(data.get('loss_rate', 0.0))} | "
            f"{pct(data.get('truncation_rate', 0.0))} | {data.get('avg_final_score', 0.0):.1f} | "
            f"{data.get('avg_training_progress', 0.0):.1f} |"
        )

    lines.extend([
        "",
        "## Player Count Averages",
        "",
        "| Players | Games | Win | Adj Win | Avg Rank | Place | Perf | Top Half | Loss | Cap | Score | Progress |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for players, data in summary["by_player_count"].items():
        lines.append(
            f"| {players} | {data.get('games', 0)} | {pct(data.get('win_rate', 0.0))} | "
            f"{data.get('chance_adjusted_win_rate', 0.0):.2f}x | "
            f"{data.get('avg_progress_rank', 0.0):.2f} | {data.get('avg_progress_placement_score', 0.0):.3f} | "
            f"{data.get('avg_performance_score', 0.0):.3f} | {pct(data.get('top_half_rate', 0.0))} | "
            f"{pct(data.get('loss_rate', 0.0))} | "
            f"{pct(data.get('truncation_rate', 0.0))} | {data.get('avg_final_score', 0.0):.1f} | "
            f"{data.get('avg_training_progress', 0.0):.1f} |"
        )

    lines.append("")
    return "\n".join(lines)


def print_summary(report: Dict[str, Any]) -> None:
    print("\nMODEL QUALITY SUMMARY")
    print(f"Target: {report['target_checkpoint']}")
    print(f"Device: {report['device']}")
    print(f"Max moves: {report['max_moves']}")
    print()

    for item in report["results"]:
        summary = item["summary"]
        print(
            f"{item['scenario']:<42} "
            f"players={item['num_players']} "
            f"games={summary['games']:>3} "
            f"win={summary['win_rate']:.3f} "
            f"adj_win={summary.get('chance_adjusted_win_rate', 0.0):.2f}x "
            f"perf={summary.get('avg_performance_score', 0.0):.3f} "
            f"place={summary.get('avg_progress_placement_score', 0.0):.3f} "
            f"rank={summary.get('avg_progress_rank', 0.0):.2f} "
            f"draw={summary['draw_rate']:.3f} "
            f"loss={summary['loss_rate']:.3f} "
            f"score={summary['avg_final_score']:.1f}+/-{summary['std_final_score']:.1f} "
            f"progress={summary['avg_training_progress']:.1f} "
            f"pins_goal={summary['avg_pins_in_goal']:.2f} "
            f"dist={summary['avg_total_distance']:.1f} "
            f"cap={summary['truncation_rate']:.3f}"
        )


def main() -> None:
    TARGETCHECKPOINT = "checkpoints/gnn_h128_l4/self_play/shared_model_final.pt"
    OPPONENTCHECKPOINTS = ["checkpoints/gnn_h128_l4/bootstrap/shared_model_final.pt",
                           "checkpoints/gnn_h128_l4/self_play/champion.pt"]
    BASELINES = ["random", "heuristic", "mixed"]
    PLAYERS = [2, 3, 4, 5, 6]
    GAMES = 20
    MAXMOVES = 300
    SEED = 1000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    HIDDENDIM = None
    NUMLAYERS = None
    FILL = "heuristic"  # "heuristic" or "random"
    SUMMARYOUT = "eval_reports/model_quality_summary.md"
    SUMMARYJSONOUT = "eval_reports/model_quality_summary.json"

    WRITEDETAILEDREPORTS = False
    JSONOUT = "eval_reports/model_quality_report.json"
    JSONLOUT = "eval_reports/model_quality_games.jsonl"
    QUIET = False

    if not os.path.exists(TARGETCHECKPOINT):
        raise FileNotFoundError(f"Target checkpoint not found: {TARGETCHECKPOINT}")

    opponent_checkpoints = expand_checkpoint_args(OPPONENTCHECKPOINTS)
    missing = [path for path in opponent_checkpoints if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Opponent checkpoint(s) not found: {missing}")

    model_cache: Dict[str, torch.nn.Module] = {}
    results: List[Dict[str, Any]] = []
    scenario_index = 0

    for num_players in PLAYERS:
        if not 2 <= num_players <= 6:
            raise ValueError(f"Player count must be between 2 and 6, got {num_players}")

        for baseline in BASELINES:
            entrants = build_scenario_entrants(target_checkpoint=TARGETCHECKPOINT,
                                               scenario=baseline,
                                               num_players=num_players,
                                               fill=FILL)
            results.append(evaluate_scenario(scenario_label=f"vs_{baseline}",
                                             entrant_specs=entrants,
                                             num_players=num_players,
                                             games=GAMES,
                                             seed_base=SEED + scenario_index * 10000,
                                             max_moves=MAXMOVES,
                                             device=DEVICE,
                                             model_cache=model_cache,
                                             hidden_dim=HIDDENDIM,
                                             num_layers=NUMLAYERS,
                                             verbose=not QUIET))
            scenario_index += 1

        for opponent in opponent_checkpoints:
            label = f"vs_checkpoint:{os.path.basename(opponent)}"
            entrants = build_scenario_entrants(target_checkpoint=TARGETCHECKPOINT,
                                               scenario="checkpoint",
                                               num_players=num_players,
                                               opponent_checkpoint=opponent,
                                               fill=FILL)
            results.append(evaluate_scenario(scenario_label=label,
                                             entrant_specs=entrants,
                                             num_players=num_players,
                                             games=GAMES,
                                             seed_base=SEED + scenario_index * 10000,
                                             max_moves=MAXMOVES,
                                             device=DEVICE,
                                             model_cache=model_cache,
                                             hidden_dim=HIDDENDIM,
                                             num_layers=NUMLAYERS,
                                             verbose=not QUIET))
            scenario_index += 1

    report = {"target_checkpoint": TARGETCHECKPOINT,
              "opponent_checkpoints": opponent_checkpoints,
              "baselines": BASELINES,
              "players": PLAYERS,
              "games_per_scenario": GAMES,
              "max_moves": MAXMOVES,
              "seed": SEED,
              "device": DEVICE,
              "hidden_dim_override": HIDDENDIM,
              "num_layers_override": NUMLAYERS,
              "fill": FILL,
              "results": results}

    print_summary(report)
    compact_summary = build_compact_summary(report)
    compact_markdown = format_compact_markdown(compact_summary)

    write_text(SUMMARYOUT, compact_markdown)
    write_json(SUMMARYJSONOUT, compact_summary)
    print(f"\nSaved concise summary: {SUMMARYOUT}")
    print(f"Saved concise JSON summary: {SUMMARYJSONOUT}")

    if WRITEDETAILEDREPORTS:
        write_json(JSONOUT, report)
        write_jsonl(JSONLOUT, results)
        print(f"Saved detailed JSON report: {JSONOUT}")
        print(f"Saved per-game JSONL: {JSONLOUT}")


if __name__ == "__main__":
    main()
