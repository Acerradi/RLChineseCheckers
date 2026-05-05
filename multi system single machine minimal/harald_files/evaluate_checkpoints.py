from __future__ import annotations

import json
import sys
import os
import random
import re
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from environment import ChineseCheckersEnv
from policies import RandomPolicy
from policy_template import HeuristicPolicy, MyPolicy, load_model


# ============================================================
# MODE
# ============================================================
# "single" = evaluate one target checkpoint, optionally against one earlier checkpoint
# "ladder" = evaluate a whole checkpoint sequence over time
RUN_MODE = "single"   # "single" or "ladder"

# ============================================================
# SINGLE-EVALUATION CONFIG
# ============================================================
TARGET_CHECKPOINT = "checkpoints/gnn_h64_l3/self_play/shared_model_final.pt"
EARLIER_CHECKPOINT = "checkpoints/gnn_h64_l3/bootstrap/shared_model_final.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"      # "cpu" or "cuda"
REPEATS = 20
MAX_MOVES = 300
PLAYER_COUNTS = [2, 3, 4, 5, 6]

JSON_OUT = "eval_reports/single_eval_report.json"


# ============================================================
# LADDER CONFIG
# ============================================================
# Option A: explicit checkpoint list in chronological order
CHECKPOINT_LADDER = [
    "checkpoints/gnn_h64_l3/bootstrap/shared_model_400.pt",
    "checkpoints/gnn_h64_l3/self_play/shared_model_400.pt",
    "checkpoints/gnn_h64_l3/self_play/shared_model_final.pt"]

# Option B: auto-build ladder from a directory
AUTO_BUILD_LADDER = False
LADDER_DIR = "checkpoints/gnn_h64_l3/self_play"
LADDER_EVERY_NTH = 5        # take every Nth numbered checkpoint from sorted list
INCLUDE_FINAL_IF_PRESENT = True

LADDER_JSON_OUT = "eval_reports/ladder_eval_report.json"


# ============================================================
# PROGRESS / VERBOSITY
# ============================================================
PRINT_SCENARIO_START = True
PRINT_EACH_GAME = True
PRINT_GAME_RESULT = True
PRINT_REQUIREMENT_DETAILS = True


# ============================================================
# SUCCESS CRITERIA THRESHOLDS
# Tune these after your first real run if needed.
# ============================================================
REQ_RANDOM_MIN_WINRATE = 0.75

# "Competitive with heuristic" can be satisfied by either win rate or average score.
REQ_HEURISTIC_MIN_WINRATE = 0.35
REQ_HEURISTIC_MIN_AVG_SCORE = 250.0

REQ_EARLIER_MIN_WINRATE = 0.55

REQ_MAX_MOVE_CAP_RATE = 0.20
REQ_MAX_SCORE_STD = 250.0


# ============================================================
# UTILITIES
# ============================================================
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_mean(xs: List[float]) -> float:
    return mean(xs) if xs else 0.0


def safe_std(xs: List[float]) -> float:
    return pstdev(xs) if len(xs) >= 2 else 0.0


def checkpoint_sort_key(path: str):
    """
    Sort numbered checkpoints by numeric suffix, with final at the end.
    Examples:
      shared_model_100.pt
      shared_model_500.pt
      shared_model_final.pt
    """
    name = os.path.basename(path)
    if name == "shared_model_final.pt":
        return (1, float("inf"))

    m = re.match(r"shared_model_(\d+)\.pt$", name)
    if m:
        return (0, int(m.group(1)))

    return (2, name)


def build_checkpoint_ladder_from_dir(checkpoint_dir: str,
                                     every_nth: int = 5,
                                     include_final: bool = True) -> List[str]:
    
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    numbered = []
    final_path = None

    for name in os.listdir(checkpoint_dir):
        full = os.path.join(checkpoint_dir, name)
        if name == "shared_model_final.pt":
            final_path = full
            continue

        m = re.match(r"shared_model_(\d+)\.pt$", name)
        if m:
            numbered.append((int(m.group(1)), full))

    numbered.sort(key=lambda x: x[0])

    if every_nth <= 1:
        selected = [p for _, p in numbered]
    else:
        selected = [p for i, (_, p) in enumerate(numbered) if i % every_nth == 0]

        # always include the latest numbered checkpoint if it wasn't selected
        if numbered:
            latest_numbered = numbered[-1][1]
            if latest_numbered not in selected:
                selected.append(latest_numbered)

    selected.sort(key=checkpoint_sort_key)

    if include_final and final_path is not None:
        if final_path not in selected:
            selected.append(final_path)

    return selected


def infer_truncated(result: Dict[str, Any], state: Dict[str, Any], max_moves: int) -> bool:
    # works whether or not environment.py was patched to return "truncated"
    if "truncated" in result:
        return bool(result["truncated"])

    move_count = int(state.get("move_count", 0))
    finished = state.get("status") == "FINISHED"
    return (not finished) and move_count >= max_moves


@dataclass
class EntrantSpec:
    name: str
    kind: str   # "checkpoint" | "heuristic" | "random"
    checkpoint: Optional[str] = None


def make_policy(spec: EntrantSpec, device: str, seed: int):
    if spec.kind == "checkpoint":
        if not spec.checkpoint:
            raise ValueError(f"Checkpoint entrant {spec.name} missing checkpoint path")
        model = load_model(spec.checkpoint, device=device)
        return MyPolicy(model=model, device=device)

    if spec.kind == "heuristic":
        return HeuristicPolicy(epsilon=0.0)

    if spec.kind == "random":
        return RandomPolicy(seed=seed)

    raise ValueError(f"Unknown entrant kind: {spec.kind}")


# ============================================================
# SCENARIO BUILDERS
# ============================================================
def target_vs_random(target: EntrantSpec, num_players: int) -> List[EntrantSpec]:
    return [target] + [EntrantSpec(name=f"random_{i}", kind="random")
                       for i in range(num_players - 1)]


def target_vs_heuristic(target: EntrantSpec, num_players: int) -> List[EntrantSpec]:
    return [target] + [EntrantSpec(name=f"heuristic_{i}", kind="heuristic")
                       for i in range(num_players - 1)]


def target_vs_mixed_baselines(target: EntrantSpec, num_players: int) -> List[EntrantSpec]:
    entrants = [target]
    for i in range(num_players - 1):
        if i % 2 == 0:
            entrants.append(EntrantSpec(name=f"heuristic_{i}", kind="heuristic"))
        else:
            entrants.append(EntrantSpec(name=f"random_{i}", kind="random"))
    return entrants


def checkpoint_head_to_head(target: EntrantSpec, opponent: EntrantSpec, num_players: int) -> List[EntrantSpec]:
    entrants = [target, opponent]
    while len(entrants) < num_players:
        i = len(entrants)
        entrants.append(EntrantSpec(name=f"heuristic_fill_{i}", kind="heuristic"))
    return entrants


# ============================================================
# MATCH RUNNER
# ============================================================
def run_single_match(entrant_specs: List[EntrantSpec],
                     target_name: str,
                     num_players: int,
                     seed: int,
                     max_moves: int,
                     device: str,
                     shuffle_seats: bool = True) -> Dict[str, Any]:
    
    if len(entrant_specs) != num_players:
        raise ValueError(f"Need exactly {num_players} entrants, got {len(entrant_specs)}")

    set_global_seed(seed)

    entrants = list(entrant_specs)
    if shuffle_seats:
        rng = random.Random(seed)
        rng.shuffle(entrants)

    player_names = [e.name for e in entrants]

    env = ChineseCheckersEnv(num_players=num_players, player_names=player_names)
    env.reset()

    policies_by_colour = {}
    for player in env.game.players:
        spec = next(e for e in entrants if e.name == player.name)
        policies_by_colour[player.colour] = make_policy(spec, device=device, seed=seed)

    result = env.run_policies(policies_by_colour, max_moves=max_moves)
    state = result["state"]
    truncated = infer_truncated(result, state, max_moves=max_moves)

    target_player = next(p for p in state["players"] if p["name"] == target_name)
    target_colour = target_player["colour"]
    target_score = result["scores"].get(target_colour, {}).get("final_score", 0.0)

    return {"seed": seed,
            "num_players": num_players,
            "target_name": target_name,
            "target_colour": target_colour,
            "target_status": target_player["status"],
            "target_score": target_score,
            "move_count": state.get("move_count", 0),
            "truncated": truncated,
            "finished": state.get("status") == "FINISHED",
            "turn_order": state.get("turn_order", []),
            "full_state": state,
            "scores": result["scores"]}


# ============================================================
# AGGREGATION
# ============================================================
def summarize_match_batch(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    wins = sum(r["target_status"] == "WIN" for r in results)
    draws = sum(r["target_status"] == "DRAW" for r in results)
    losses = sum(r["target_status"] not in ("WIN", "DRAW") for r in results)

    scores = [float(r["target_score"]) for r in results]
    move_counts = [int(r["move_count"]) for r in results]
    truncs = sum(bool(r["truncated"]) for r in results)

    return {"games": len(results),
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": wins / len(results) if results else 0.0,
            "draw_rate": draws / len(results) if results else 0.0,
            "loss_rate": losses / len(results) if results else 0.0,
            "avg_score": safe_mean(scores),
            "std_score": safe_std(scores),
            "avg_moves": safe_mean(move_counts),
            "std_moves": safe_std(move_counts),
            "move_cap_rate": truncs / len(results) if results else 0.0}


# ============================================================
# SCENARIO EVALUATION
# ============================================================
def evaluate_scenario(scenario_name: str,
                      entrant_specs: List[EntrantSpec],
                      target_name: str,
                      num_players: int,
                      repeats: int,
                      max_moves: int,
                      device: str,
                      seed_offset: int = 0) -> Dict[str, Any]:
    
    if PRINT_SCENARIO_START:
        print("\n" + "=" * 90)
        print(f"STARTING SCENARIO: {scenario_name}")
        print(f"players={num_players} repeats={repeats} max_moves={max_moves}")
        print("entrants:", [e.name for e in entrant_specs])
        print("=" * 90)

    results = []

    for i in range(repeats):
        seed = seed_offset + i

        if PRINT_EACH_GAME:
            print(f"[{scenario_name} | players={num_players}] "
                  f"game {i+1}/{repeats} seed={seed} ...")

        out = run_single_match(entrant_specs=entrant_specs,
                               target_name=target_name,
                               num_players=num_players,
                               seed=seed,
                               max_moves=max_moves,
                               device=device,
                               shuffle_seats=True)
        results.append(out)

        if PRINT_GAME_RESULT:
            print(f"  -> status={out['target_status']:<7} "
                  f"score={out['target_score']:.2f} "
                  f"moves={out['move_count']} "
                  f"truncated={out['truncated']}")

    summary = summarize_match_batch(results)

    print(f"COMPLETED [{scenario_name} | players={num_players}] "
          f"win={summary['win_rate']:.3f} "
          f"draw={summary['draw_rate']:.3f} "
          f"loss={summary['loss_rate']:.3f} "
          f"score={summary['avg_score']:.2f}±{summary['std_score']:.2f} "
          f"moves={summary['avg_moves']:.1f}±{summary['std_moves']:.1f} "
          f"cap={summary['move_cap_rate']:.3f}")

    return {"scenario": scenario_name,
            "num_players": num_players,
            "summary": summary,
            "games": results}


# ============================================================
# CHECKPOINT SUITE
# ============================================================
def evaluate_checkpoint_suite(target_checkpoint: str,
                              earlier_checkpoint: Optional[str],
                              repeats: int,
                              max_moves: int,
                              device: str,
                              player_counts: List[int]) -> Dict[str, Any]:
    
    if not os.path.exists(target_checkpoint):
        raise FileNotFoundError(f"Target checkpoint not found: {target_checkpoint}")

    if earlier_checkpoint is not None and not os.path.exists(earlier_checkpoint):
        raise FileNotFoundError(f"Earlier checkpoint not found: {earlier_checkpoint}")

    target = EntrantSpec(name="target_model", kind="checkpoint", checkpoint=target_checkpoint)

    suite = []

    for num_players in player_counts:
        suite.append(evaluate_scenario(scenario_name="vs_random",
                                       entrant_specs=target_vs_random(target, num_players),
                                       target_name=target.name,
                                       num_players=num_players,
                                       repeats=repeats,
                                       max_moves=max_moves,
                                       device=device,
                                       seed_offset=1000 * num_players + 10))
        
        suite.append(evaluate_scenario(scenario_name="vs_heuristic",
                                       entrant_specs=target_vs_heuristic(target, num_players),
                                       target_name=target.name,
                                       num_players=num_players,
                                       repeats=repeats,
                                       max_moves=max_moves,
                                       device=device,
                                       seed_offset=1000 * num_players + 100))
        
        suite.append(evaluate_scenario(scenario_name="vs_mixed_baselines",
                                       entrant_specs=target_vs_mixed_baselines(target, num_players),
                                       target_name=target.name,
                                       num_players=num_players,
                                       repeats=repeats,
                                       max_moves=max_moves,
                                       device=device,
                                       seed_offset=1000 * num_players + 200))

        if earlier_checkpoint:
            earlier = EntrantSpec(name="earlier_model",
                                  kind="checkpoint",
                                  checkpoint=earlier_checkpoint)
            suite.append(evaluate_scenario(scenario_name="vs_earlier_checkpoint",
                                           entrant_specs=checkpoint_head_to_head(target, earlier, num_players),
                                           target_name=target.name,
                                           num_players=num_players,
                                           repeats=repeats,
                                           max_moves=max_moves,
                                           device=device,
                                           seed_offset=1000 * num_players + 300))

    return {"target_checkpoint": target_checkpoint,
            "earlier_checkpoint": earlier_checkpoint,
            "repeats": repeats,
            "max_moves": max_moves,
            "player_counts": player_counts,
            "results": suite}


# ============================================================
# REQUIREMENT REPORT
# ============================================================
def build_requirement_report(report: Dict[str, Any]) -> Dict[str, Any]:
    by_key = {(item["scenario"], item["num_players"]): item["summary"]
               for item in report["results"]}

    requirements = []

    # 1. Better than random across all player counts
    random_passes = []
    for n in report["player_counts"]:
        s = by_key.get(("vs_random", n))
        passed = s is not None and s["win_rate"] >= REQ_RANDOM_MIN_WINRATE
        random_passes.append(passed)
        requirements.append({"requirement": f"Better than random ({n} players)",
                             "passed": passed,
                             "details": s
                             })

    # 2. Competitive with heuristic across all player counts
    heuristic_passes = []
    for n in report["player_counts"]:
        s = by_key.get(("vs_heuristic", n))
        passed = (s is not None and (s["win_rate"] >= REQ_HEURISTIC_MIN_WINRATE or
                                     s["avg_score"] >= REQ_HEURISTIC_MIN_AVG_SCORE))
        heuristic_passes.append(passed)
        requirements.append({"requirement": f"Competitive with heuristic ({n} players)",
                             "passed": passed,
                             "details": s,
                            })

    # 3. Better than earlier checkpoint
    if report.get("earlier_checkpoint"):
        earlier_passes = []
        for n in report["player_counts"]:
            s = by_key.get(("vs_earlier_checkpoint", n))
            passed = s is not None and s["win_rate"] >= REQ_EARLIER_MIN_WINRATE
            earlier_passes.append(passed)
            requirements.append({"requirement": f"Better than earlier checkpoint ({n} players)",
                                 "passed": passed,
                                 "details": s,
                                })

    # 4. Stable across repeated batches
    stability_passes = []
    for item in report["results"]:
        s = item["summary"]
        passed = s["std_score"] <= REQ_MAX_SCORE_STD
        stability_passes.append(passed)
        requirements.append({"requirement": f"Stable scores ({item['scenario']}, {item['num_players']} players)",
                             "passed": passed,
                             "details": s,
                            })

    # 5. Not frequently hitting the move cap
    cap_passes = []
    for item in report["results"]:
        s = item["summary"]
        passed = s["move_cap_rate"] <= REQ_MAX_MOVE_CAP_RATE
        cap_passes.append(passed)
        requirements.append({"requirement": f"Low move-cap rate ({item['scenario']}, {item['num_players']} players)",
                             "passed": passed,
                             "details": s,
                        })

    # 6. Useful behavior across all player counts 2-6
    overall_multiplayer_pass = all(random_passes) and all(heuristic_passes)
    requirements.append({"requirement": "Shows useful behavior across all player counts 2-6",
                         "passed": overall_multiplayer_pass,
                         "details": {"random_passes": random_passes,
                                     "heuristic_passes": heuristic_passes},
                         })

    overall_pass = all(r["passed"] for r in requirements)

    return {"overall_pass": overall_pass, "requirements": requirements,}


# ============================================================
# PRINTING
# ============================================================
def print_human_summary(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print("CHECKPOINT EVALUATION SUMMARY")
    print("=" * 100)
    print("Target :", report["target_checkpoint"])
    print("Earlier:", report["earlier_checkpoint"])
    print("Repeats:", report["repeats"])
    print("Max moves:", report["max_moves"])
    print("Player counts:", report["player_counts"])
    print()

    for item in report["results"]:
        s = item["summary"]
        print(f"[{item['scenario']:>20}] "
              f"players={item['num_players']}  "
              f"win={s['win_rate']:.3f}  "
              f"draw={s['draw_rate']:.3f}  "
              f"loss={s['loss_rate']:.3f}  "
              f"score={s['avg_score']:.2f}±{s['std_score']:.2f}  "
              f"moves={s['avg_moves']:.1f}±{s['std_moves']:.1f}  "
              f"cap={s['move_cap_rate']:.3f}")


def print_requirement_report(req_report: Dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print("REQUIREMENT CHECK")
    print("=" * 100)

    for item in req_report["requirements"]:
        tag = "PASS" if item["passed"] else "FAIL"
        print(f"[{tag}] {item['requirement']}")
        if PRINT_REQUIREMENT_DETAILS:
            details = item.get("details")
            if isinstance(details, dict):
                print(f"       details: {json.dumps(details, ensure_ascii=False)}")

    print("-" * 100)
    print("OVERALL:", "PASS" if req_report["overall_pass"] else "FAIL")


# ============================================================
# LADDER EVALUATION
# ============================================================
def evaluate_checkpoint_ladder(checkpoints: List[str],
                               repeats: int,
                               max_moves: int,
                               device: str,
                               player_counts: List[int]) -> List[Dict[str, Any]]:
    if len(checkpoints) < 2:
        raise ValueError("Need at least 2 checkpoints in ladder mode")

    reports = []

    for i in range(1, len(checkpoints)):
        earlier = checkpoints[i - 1]
        target = checkpoints[i]

        print("\n" + "#" * 100)
        print(f"LADDER COMPARISON {i}/{len(checkpoints) - 1}")
        print(f"Earlier: {earlier}")
        print(f"Target : {target}")
        print("#" * 100)

        report = evaluate_checkpoint_suite(target_checkpoint=target,
                                           earlier_checkpoint=earlier,
                                           repeats=repeats,
                                           max_moves=max_moves,
                                           device=device,
                                           player_counts=player_counts)
        req_report = build_requirement_report(report)

        print_human_summary(report)
        print_requirement_report(req_report)

        reports.append({"evaluation_report": report, "requirement_report": req_report})

    return reports


# ============================================================
# SAVE
# ============================================================
def save_json(payload: Dict[str, Any], path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved report to: {path}")


# ============================================================
# MAIN
# ============================================================
def main():
    if RUN_MODE == "single":
        report = evaluate_checkpoint_suite(target_checkpoint=TARGET_CHECKPOINT,
                                           earlier_checkpoint=EARLIER_CHECKPOINT,
                                           repeats=REPEATS,
                                           max_moves=MAX_MOVES,
                                           device=DEVICE,
                                           player_counts=PLAYER_COUNTS)

        req_report = build_requirement_report(report)

        print_human_summary(report)
        print_requirement_report(req_report)

        payload = {"mode": "single", "evaluation_report": report, "requirement_report": req_report}
        save_json(payload, JSON_OUT)

    elif RUN_MODE == "ladder":
        checkpoints = list(CHECKPOINT_LADDER)

        if AUTO_BUILD_LADDER:
            checkpoints = build_checkpoint_ladder_from_dir(checkpoint_dir=LADDER_DIR,
                                                           every_nth=LADDER_EVERY_NTH,
                                                           include_final=INCLUDE_FINAL_IF_PRESENT)

        checkpoints = [p for p in checkpoints if p]
        checkpoints = sorted(checkpoints, key=checkpoint_sort_key)

        print("\n" + "=" * 100)
        print("CHECKPOINT LADDER")
        print("=" * 100)
        for i, ckpt in enumerate(checkpoints):
            print(f"{i:02d}: {ckpt}")

        ladder_reports = evaluate_checkpoint_ladder(checkpoints=checkpoints,
                                                    repeats=REPEATS,
                                                    max_moves=MAX_MOVES,
                                                    device=DEVICE,
                                                    player_counts=PLAYER_COUNTS)

        payload = {"mode": "ladder",
                   "checkpoints": checkpoints,
                   "ladder_reports": ladder_reports}
        save_json(payload, LADDER_JSON_OUT)

    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")


if __name__ == "__main__":
    main()