import numpy as np


class ActionMapper:
    def __init__(self, board_indices, max_pin_id=10):
        self.board_indices = sorted(board_indices)
        self.dest_to_offset = {d: i for i, d in enumerate(self.board_indices)}
        self.offset_to_dest = {i: d for i, d in enumerate(self.board_indices)}
        self.max_pin_id = max_pin_id
        self.action_size = max_pin_id * len(self.board_indices)

    def encode_action(self, pin_id, dest_idx):
        return pin_id * len(self.board_indices) + self.dest_to_offset[dest_idx]

    def decode_action(self, action_id):
        pin_id = action_id // len(self.board_indices)
        dest_offset = action_id % len(self.board_indices)
        dest_idx = self.offset_to_dest[dest_offset]
        return {"pin_id": pin_id, "to": dest_idx}

    def _extract_pin_id(self, move):
        if not isinstance(move, dict):
            return None
        for key in ("pin_id", "piece_id", "id"):
            if key in move:
                return move[key]
        return None

    def _extract_dest(self, move):
        if not isinstance(move, dict):
            return None
        for key in ("to", "to_axial", "destination", "target", "move_to"):
            if key in move:
                return move[key]
        return None

    def legal_action_mask(self, legal_moves):
        mask = np.zeros(self.action_size, dtype=np.float32)

        for move in legal_moves:
            pin_id = self._extract_pin_id(move)
            dest_idx = self._extract_dest(move)

            if pin_id is None or dest_idx is None:
                continue

            if not isinstance(pin_id, int):
                continue

            if dest_idx not in self.dest_to_offset:
                continue

            if not (0 <= pin_id < self.max_pin_id):
                continue

            a = self.encode_action(pin_id, dest_idx)
            mask[a] = 1.0

        return mask