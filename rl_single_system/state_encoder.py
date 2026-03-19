import numpy as np

def encode_observation(obs, current_colour, colour_to_idx):
    """
    Convert repo observation dict into a flat float32 vector.

    Expected obs keys from the wrapper:
      - board_state
      - own_pieces
      - legal_moves
      - goal_indices
      - last_move
      - turn_number
      - num_players
    """

    board_state = obs["board_state"]  # usually mapping axial index -> occupant / info
    goal_indices = set(obs.get("goal_indices", []))
    num_players = obs.get("num_players", 2)
    turn_number = obs.get("turn_number", 0)

    # Sort board indices so encoding is stable
    board_indices = sorted(board_state.keys())
    n_cells = len(board_indices)

    # Channels:
    # 0 = my piece
    # 1 = opponent piece
    # 2 = empty
    # 3 = my goal cell
    my_piece = np.zeros(n_cells, dtype=np.float32)
    opp_piece = np.zeros(n_cells, dtype=np.float32)
    empty = np.zeros(n_cells, dtype=np.float32)
    my_goal = np.zeros(n_cells, dtype=np.float32)

    for i, idx in enumerate(board_indices):
        cell = board_state[idx]

        occupant = None
        if isinstance(cell, dict):
            occupant = cell.get("occupant")
        else:
            occupant = cell

        if occupant is None:
            empty[i] = 1.0
        elif occupant == current_colour:
            my_piece[i] = 1.0
        else:
            opp_piece[i] = 1.0

        if idx in goal_indices:
            my_goal[i] = 1.0

    # small global features
    global_feats = np.array([
        float(turn_number) / 500.0,
        float(num_players) / 6.0,
    ], dtype=np.float32)

    x = np.concatenate([my_piece, opp_piece, empty, my_goal, global_feats], axis=0)
    return x, board_indices