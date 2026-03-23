def compute_progress_reward(prev_info, next_info, won=False, lost=False, draw=False):
    if won:
        return 1000.0
    if lost:
        return -1000.0
    if draw:
        return -200.0

    dist_delta = prev_info["total_distance"] - next_info["total_distance"]

    in_goal_delta = next_info["pieces_in_goal"] - prev_info["pieces_in_goal"]
    within_1_delta = next_info["pieces_within_1"] - prev_info["pieces_within_1"]
    within_2_delta = next_info["pieces_within_2"] - prev_info["pieces_within_2"]
    within_3_delta = next_info["pieces_within_3"] - prev_info["pieces_within_3"]

    max_dist_prev = max(prev_info["distances"])
    max_dist_next = max(next_info["distances"])
    tail_delta = max_dist_prev - max_dist_next

    reward = (
        0.25 * dist_delta
        + 20.0 * in_goal_delta
        + 8.0 * within_1_delta
        + 4.0 * within_2_delta
        + 2.0 * within_3_delta
        + 1.5 * tail_delta
        - 0.1
    )

    return reward