from collections import deque
import math


def get_index_to_xy(board):
    if not hasattr(board, "cartesian"):
        raise RuntimeError("board.cartesian not found")

    cart = board.cartesian

    # dict[int -> (x,y)]
    if isinstance(cart, dict) and cart:
        keys = list(cart.keys())
        if all(isinstance(k, int) for k in keys):
            out = {}
            for idx, xy in cart.items():
                if isinstance(xy, (tuple, list)) and len(xy) >= 2:
                    out[idx] = (float(xy[0]), float(xy[1]))
            if out:
                return out

    # dict[pos_obj -> (x,y)] with board.index_of
    if isinstance(cart, dict) and hasattr(board, "index_of"):
        index_of = board.index_of
        if isinstance(index_of, dict):
            out = {}
            for pos, xy in cart.items():
                if pos in index_of and isinstance(xy, (tuple, list)) and len(xy) >= 2:
                    out[index_of[pos]] = (float(xy[0]), float(xy[1]))
            if out:
                return out

    # list aligned with ids
    if isinstance(cart, list) and cart:
        out = {}
        for idx, xy in enumerate(cart):
            if isinstance(xy, (tuple, list)) and len(xy) >= 2:
                out[idx] = (float(xy[0]), float(xy[1]))
        if out:
            return out

    raise RuntimeError("Could not extract indexed cartesian coordinates")


def infer_neighbor_distance(idx_to_xy):
    """
    Infer the smallest non-zero pairwise distance; on a lattice this should
    be the edge length between adjacent cells.
    """
    items = list(idx_to_xy.items())
    dmin = float("inf")

    for i, (_, (x1, y1)) in enumerate(items):
        for _, (x2, y2) in items[i + 1:]:
            d = math.hypot(x1 - x2, y1 - y2)
            if d > 1e-9 and d < dmin:
                dmin = d

    if not math.isfinite(dmin):
        raise RuntimeError("Could not infer neighbor distance from coordinates")

    return dmin


def build_adjacency_from_cartesian(board):
    idx_to_xy = get_index_to_xy(board)
    neighbor_dist = infer_neighbor_distance(idx_to_xy)

    # tolerance around inferred nearest-neighbor distance
    tol = neighbor_dist * 0.10

    adjacency = {idx: [] for idx in idx_to_xy.keys()}
    items = list(idx_to_xy.items())

    for i, (idx_a, (x1, y1)) in enumerate(items):
        for idx_b, (x2, y2) in items[i + 1:]:
            d = math.hypot(x1 - x2, y1 - y2)
            if abs(d - neighbor_dist) <= tol:
                adjacency[idx_a].append(idx_b)
                adjacency[idx_b].append(idx_a)

    for idx in adjacency:
        adjacency[idx] = sorted(set(adjacency[idx]))

    return adjacency, neighbor_dist


def bfs_distances_from(start_idx, adjacency):
    dist = {start_idx: 0}
    queue = deque([start_idx])

    while queue:
        cur = queue.popleft()
        for nbr in adjacency[cur]:
            if nbr not in dist:
                dist[nbr] = dist[cur] + 1
                queue.append(nbr)

    return dist


def precompute_all_pairs_shortest_paths(board):
    adjacency, neighbor_dist = build_adjacency_from_cartesian(board)
    all_distances = {}

    for idx in adjacency.keys():
        all_distances[idx] = bfs_distances_from(idx, adjacency)

    return all_distances, adjacency, neighbor_dist