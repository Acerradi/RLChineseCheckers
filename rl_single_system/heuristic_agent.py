def choose_heuristic_move(obs):
    legal_moves = obs["legal_moves"]
    goal_indices = set(obs.get("goal_indices", []))

    best_move = None
    best_score = -1e9

    for move in legal_moves:
        pin_id = move["pin_id"]
        dest = move["to"]

        score = 0.0

        if dest in goal_indices:
            score += 10.0

        # crude forward-progress approximation
        nearest_goal_dist = min(abs(dest - g) for g in goal_indices) if goal_indices else 0
        score += -nearest_goal_dist

        if score > best_score:
            best_score = score
            best_move = move

    return best_move