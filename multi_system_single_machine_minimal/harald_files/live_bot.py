from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, Optional

from policies import RandomPolicy
from socket_adapter import SocketRPCClient


class LiveSocketBot:
    """Deploy a policy through the original socket/server setup.

    The important part is that the policy sees the same action/observation shape
    as in training: {"colour", "state", "legal_moves"} and returns
    (pin_id, to_index).
    """

    def __init__(
        self,
        policy,
        *,
        name: str,
        host: str = "127.0.0.1",
        port: int = 50555,
        poll_interval_sec: float = 0.2,
        auto_start: bool = True,
        verbose: bool = True,
    ):
        self.policy = policy
        self.name = name
        self.client = SocketRPCClient(host=host, port=port)
        self.poll_interval_sec = poll_interval_sec
        self.auto_start = auto_start
        self.verbose = verbose

        self.game_id: Optional[str] = None
        self.player_id: Optional[str] = None
        self.colour: Optional[str] = None
        self._start_sent = False
        self._last_seen_move = -1

    def log(self, *args):
        if self.verbose:
            print(*args)

    def join(self) -> None:
        reply = self.client.rpc({"op": "join", "player_name": self.name})
        if not reply.get("ok"):
            raise RuntimeError(f"join failed: {reply.get('error')}")
        self.game_id = reply["game_id"]
        self.player_id = reply["player_id"]
        self.colour = reply["colour"]
        self.log(f"Joined game {self.game_id} as {self.colour}")

    def build_observation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        legal_reply = self.client.rpc({
            "op": "get_legal_moves",
            "game_id": self.game_id,
            "player_id": self.player_id,
        })
        if not legal_reply.get("ok"):
            raise RuntimeError(f"get_legal_moves failed: {legal_reply.get('error')}")
        return {
            "colour": self.colour,
            "state": state,
            "legal_moves": legal_reply.get("legal_moves", {}),
        }

    def maybe_send_start(self, state: Dict[str, Any]) -> None:
        if self._start_sent or not self.auto_start:
            return
        if state.get("status") in ("READY_TO_START", "PLAYING"):
            reply = self.client.rpc({"op": "start", "game_id": self.game_id, "player_id": self.player_id})
            if reply.get("ok"):
                self._start_sent = True
                self.log("Sent START")

    def run(self) -> Dict[str, Any]:
        if self.game_id is None:
            self.join()

        while True:
            st = self.client.rpc({"op": "get_state", "game_id": self.game_id})
            if not st.get("ok"):
                raise RuntimeError(f"get_state failed: {st.get('error')}")

            state = st["state"]
            self.maybe_send_start(state)

            if state.get("move_count", 0) > self._last_seen_move:
                mv = state.get("last_move")
                if mv is not None:
                    self.log(f"MOVE: {mv['by']} ({mv['colour']}) {mv['from']}->{mv['to']} [{mv['move_ms']:.1f}ms]")
                self._last_seen_move = state.get("move_count", 0)

            if state["status"] == "FINISHED":
                self.log("Game finished")
                return state

            if state.get("current_turn_colour") == self.colour and state["status"] == "PLAYING":
                obs = self.build_observation(state)
                pin_id, to_index = self.policy.select_action(obs)
                mv = self.client.rpc({
                    "op": "move",
                    "game_id": self.game_id,
                    "player_id": self.player_id,
                    "pin_id": pin_id,
                    "to_index": to_index,
                })
                if not mv.get("ok"):
                    self.log("Move rejected:", mv.get("error"))
                    time.sleep(self.poll_interval_sec)
                    continue
                if mv.get("status") == "WIN":
                    self.log(mv.get("msg"))

            time.sleep(self.poll_interval_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a policy through the original socket server")
    parser.add_argument("--name", default="bot")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50555)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--policy", default="random", choices=["random"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.policy == "random":
        policy = RandomPolicy(seed=args.seed)
    else:
        raise ValueError(f"Unknown policy {args.policy}")

    bot = LiveSocketBot(policy, name=args.name, host=args.host, port=args.port, verbose=not args.quiet)
    final_state = bot.run()
    if not args.quiet:
        print(json.dumps(final_state, indent=2))


if __name__ == "__main__":
    main()
