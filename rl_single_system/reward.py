def compute_progress_reward(prev_info, next_info, won=False, lost=False):
    """
    prev_info and next_info are your own summary dicts, for example:
      {
        "total_distance": ...,
        "pieces_in_goal": ...
      }
    """

    if won:
        return 100.0
    if lost:
        return -100.0

    dist_delta = prev_info["total_distance"] - next_info["total_distance"]
    goal_delta = next_info["pieces_in_goal"] - prev_info["pieces_in_goal"]

    reward = 0.5 * dist_delta + 2.0 * goal_delta - 0.01
    return reward