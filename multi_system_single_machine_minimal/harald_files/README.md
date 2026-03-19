# Chinese Checkers training scaffold

This package gives you a bridge between a **fast local training loop** and the **original socket/server tournament client**.

## Design goal

Train policies in-process, but keep the policy interface identical to the live client:

- observation: `{ "colour", "state", "legal_moves" }`
- action: `(pin_id, to_index)`

That way, the same policy object can later be wrapped by a socket bot and used against other people's models.

## Files

- `core.py` — shared rules engine extracted from your server logic
- `env.py` — RL-friendly wrapper with `reset()`, `observe()`, `legal_moves()`, `step()`
- `policies.py` — policy interface plus a random baseline
- `socket_adapter.py` — thin JSON RPC client for the original server
- `live_bot.py` — production-style bot that joins the socket server and plays automatically
- `run_local_match.py` — quick local test harness
- `league.py` — small helper for running repeated local self-play matches

## Expected project layout

This code expects the original board implementation to be importable:

- `checkers_board.py`
- `checkers_pins.py`

Place `cc_training/` beside those files, or otherwise ensure they are on `PYTHONPATH`.

## Quick test: local match

Run an in-process 6-player random match:

```bash
python -m cc_training.run_local_match --players 6
```

## Quick test: live socket bot

Start the original server in one terminal. Then start one or more automated bots:

```bash
python -m cc_training.live_bot --name bot_1
python -m cc_training.live_bot --name bot_2
python -m cc_training.live_bot --name bot_3
```

You can replace `RandomPolicy` with your eventual learned policy without changing the server protocol.

## How to plug in a future model

Your future policy just needs a method like:

```python
class MyPolicy(BasePolicy):
    def select_action(self, observation):
        # observation["state"] matches the server's public JSON state
        # observation["legal_moves"] matches get_legal_moves()
        return pin_id, to_index
```

Use that policy in:

- `ChineseCheckersEnv.run_policies(...)` for training/evaluation
- `LiveSocketBot(policy=...)` for tournament-style deployment

## Important note

This scaffold mirrors the **public interface and rules flow** of your current system, but it does not duplicate the human-only parts of the original client, such as prompting for keyboard input or manual start confirmation.

That is intentional: the policy should learn the game, not the terminal workflow.
