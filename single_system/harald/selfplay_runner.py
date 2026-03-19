
import sys
import os

# Go two directories up
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..')
    )
)

from single_system.harald.checkers_env import ChineseCheckersEnv
from single_system.harald.agents import RandomAgent


def run_single_game(num_players=2, render=False, max_turns=500):
    env = ChineseCheckersEnv(use_gui=False)
    env.reset(num_players=num_players, randomize_start_player=True)

    agents = {
        colour: RandomAgent(name=f"{colour}_agent")
        for colour in env.assigned_colours
    }

    if render:
        print("Assigned players:", env.assigned_colours)
        print("Starting player:", env.current_player)
        env.render_ascii()

    while not env.done and env.num_turns < max_turns:
        current_colour = env.current_player
        obs = env.get_observation(current_colour)

        legal_moves = obs.get("legal_moves", [])
        if not legal_moves:
            if render:
                print(f"\nTurn {env.num_turns}: {current_colour} has no legal moves. Skipping turn.")
            env.current_player_idx = (env.current_player_idx + 1) % len(env.assigned_colours)
            env.num_turns += 1
            continue

        try:
            action = agents[current_colour].choose_action(obs)
            next_obs, reward, done, info = env.step(action)
        except Exception as e:
            if render:
                print(f"\nCrash on turn {env.num_turns} for player {current_colour}: {e}")
                print("Observation:", obs)
            raise

        if render:
            print(f"\nTurn {env.num_turns}: {current_colour} chose {action}")
            env.render_ascii()

    if not env.done:
        if render:
            print(f"Reached max_turns={max_turns} without a winner.")
        return "draw"

    if render:
        print("Winner:", env.winner)

    return env.winner


def run_selfplay(num_games=100, num_players=2, render=False, max_turns=500):
    results = {}

    for _ in range(num_games):
        winner = run_single_game(
            num_players=num_players,
            render=render,
            max_turns=max_turns,
        )
        results[winner] = results.get(winner, 0) + 1

    return results


if __name__ == "__main__":
    summary = run_selfplay(num_games=1, num_players=2, render=True, max_turns=500)
    print(summary)