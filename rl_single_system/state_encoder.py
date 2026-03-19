import numpy as np


def _extract_occupant(cell):
    if cell is None:
        return None

    if isinstance(cell, str):
        return cell

    if isinstance(cell, dict):
        for key in ("occupant", "player", "colour", "color"):
            if key in cell:
                return cell[key]
        return None

    return None


def encode_observation(obs, current_colour, colour_to_idx=None):
    board_state = obs["board_state"]
    goal_indices = set(obs.get("goal_indices", []))
    num_players = obs.get("num_players", 2)
    turn_number = obs.get("turn_number", 0)

    # board_state is a list indexed by actual board id
    if isinstance(board_state, list):
        board_indices = list(range(len(board_state)))
        board_values = board_state
    elif isinstance(board_state, dict):
        board_indices = sorted(board_state.keys())
        board_values = [board_state[i] for i in board_indices]
    else:
        raise TypeError(f"Unsupported board_state type: {type(board_state)}")

    n_cells = len(board_indices)

    my_piece = np.zeros(n_cells, dtype=np.float32)
    opp_piece = np.zeros(n_cells, dtype=np.float32)
    empty = np.zeros(n_cells, dtype=np.float32)
    my_goal = np.zeros(n_cells, dtype=np.float32)

    for i, (idx, cell) in enumerate(zip(board_indices, board_values)):
        occupant = _extract_occupant(cell)

        if occupant is None:
            empty[i] = 1.0
        elif occupant == current_colour:
            my_piece[i] = 1.0
        else:
            opp_piece[i] = 1.0

        if idx in goal_indices:
            my_goal[i] = 1.0

    global_feats = np.array([
        float(turn_number) / 500.0,
        float(num_players) / 6.0,
    ], dtype=np.float32)

    x = np.concatenate([my_piece, opp_piece, empty, my_goal, global_feats], axis=0)
    return x, board_indices