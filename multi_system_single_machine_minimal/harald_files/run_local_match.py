from __future__ import annotations

import argparse
import json

from .environment import ChineseCheckersEnv
from .policies import RandomPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local in-process Chinese Checkers match")
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--max-moves", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = ChineseCheckersEnv(num_players=args.players)
    env.reset()
    policies = {
        colour: RandomPolicy(seed=args.seed + i)
        for i, colour in enumerate(env.turn_order)
    }
    result = env.run_policies(policies, max_moves=args.max_moves)
    print(json.dumps(result["scores"], indent=2))
    print(json.dumps(result["state"], indent=2))


if __name__ == "__main__":
    main()
