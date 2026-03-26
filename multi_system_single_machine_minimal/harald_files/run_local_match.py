from __future__ import annotations

import argparse
import json

from .environment import ChineseCheckersEnv
from .policy_template import MyPolicy, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local in-process Chinese Checkers match")
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--max-moves", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    env = ChineseCheckersEnv(num_players=args.players)
    env.reset()
    policies = {}

    for colour in env.turn_order:
        if args.checkpoint:
            model = load_model(args.checkpoint, device=args.device)
            policies[colour] = MyPolicy(model=model, device=args.device)
        else:
            policies[colour] = MyPolicy(device=args.device)
    result = env.run_policies(policies, max_moves=args.max_moves)
    print(json.dumps(result["scores"], indent=2))
    print(json.dumps(result["state"], indent=2))


if __name__ == "__main__":
    main()
