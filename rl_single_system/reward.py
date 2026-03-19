def compute_progress_reward(prev_info, next_info, won=False, lost=False):
    if won:
        return 1000.0
    if lost:
        return -1000.0

    dist_delta = prev_info["total_distance"] - next_info["total_distance"]
    goal_delta = next_info["pieces_in_goal"] - prev_info["pieces_in_goal"]

    reward = 1.0 * dist_delta + 10.0 * goal_delta - 0.1
    return reward