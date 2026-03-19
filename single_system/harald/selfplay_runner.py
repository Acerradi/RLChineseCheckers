
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


def run_single_game(num_players=2, render=False):
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

    while not env.done:
        current_colour = env.current_player
        obs = env.get_observation(current_colour)
        action = agents[current_colour].choose_action(obs)

        next_obs, reward, done, info = env.step(action)

        if render:
            print(f"\nTurn {env.num_turns}: {current_colour} chose {action}")
            env.render_ascii()

    print("Winner:", env.winner)
    return env.winner


def run_selfplay(num_games=100, num_players=2, render=False):
    results = {}

    for _ in range(num_games):
        winner = run_single_game(num_players=num_players, render=False)
        results[winner] = results.get(winner, 0) + 1

    return results


if __name__ == "__main__":
    summary = run_selfplay(num_games=1, num_players=2, render=True)
    print(summary)