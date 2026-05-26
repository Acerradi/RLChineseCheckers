# =============================================================
# player.py — simplified client that relies ONLY on JSON state
# =============================================================

import os
import json
import random
import socket
import time
import sys
from typing import Dict, Any, List

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Add harald_files to path so we can import policy_template
HARALD_FILES_DIR = os.path.join(CURRENT_DIR, "harald_files")
if HARALD_FILES_DIR not in sys.path:
    sys.path.insert(0, HARALD_FILES_DIR)

from policy_template import load_model, MyPolicy
from checkers_board import HexBoard
from checkers_pins import Pin
from checkers_gui import BoardGUI

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "gnn_h128_l4",
    "bootstrap",
    "shared_model_final.pt",
)

device = "cuda" if torch.cuda.is_available() else "cpu"

HOST = "10.245.30.144"
PORT = 50555
DEBUG_NET = os.getenv("DEBUG_NET", "0") not in ("0", "", "false", "False")


def debug(*args):
    if DEBUG_NET:
        print("[NET]", *args)


def rpc(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send JSON to server and receive JSON reply."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(10.0)
    try:
        s.connect((HOST, PORT))
    except Exception as e:
        return {"ok": False, "error": f"connect-failed: {e}"}

    s.sendall(json.dumps(payload).encode("utf-8"))
    chunks = []
    while True:
        chunk = s.recv(65_536)
        if not chunk:
            break
        chunks.append(chunk)
    data = b''.join(chunks)
    s.close()

    if not data:
        return {"ok": False, "error": "no-response"}

    try:
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        return {"ok": False, "error": f"bad-json: {e}"}


# =============================================================
# Simple renderer for the server's JSON board (optional)
# =============================================================
def render_json_board(state):
    """
    Rudimentary visualization using only JSON information.
    Does NOT require HexBoard or Pin.
    """
    pins = state.get("pins", {})
    print("=== BOARD STATE ===")
    for colour, indices in pins.items():
        print(f"{colour}: {indices}")
    print("===================")


# =============================================================
# Visual board display
# =============================================================
def create_pins_from_state(board: HexBoard, state: Dict[str, Any]) -> List[Pin]:
    """
    Create Pin objects from the JSON state.
    Returns a list of Pin objects for all pieces on the board.
    """
    pins = state.get("pins", {})
    colour_map = {
        "red": "red",
        "blue": "blue",
        "yellow": "yellow",
        "lawn green": "lawn green",
        "purple": "purple",
        "gray0": "gray0"
    }
    
    pin_list = []
    pin_id = 0
    
    for colour_name, indices in pins.items():
        colour_display = colour_map.get(colour_name, colour_name)
        for index in indices:
            pin = Pin(board, index, pin_id, color=colour_display)
            pin_list.append(pin)
            pin_id += 1
    
    return pin_list


# =============================================================
# Main client loop
# =============================================================
def main():
    timeoutnotice_move = -1
    print("==== Player ====")
    name = input("Enter name: ").strip()
    if not name:
        return

    # JOIN GAME
    r = rpc({"op": "join", "player_name": name})
    if not r.get("ok"):
        print("JOIN ERROR:", r.get("error"))
        return

    game_id = r["game_id"]
    player_id = r["player_id"]
    colour = r["colour"]

    model = load_model(CHECKPOINT_PATH, device=device)
    policy = MyPolicy(model=model, device=device)

    print(f"Joined game {game_id} as {colour}")

    # Initialize visual board display
    board = HexBoard()
    gui = None

    # Wait until game ready
    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if st.get("state", {}).get("status") in ("READY_TO_START", "PLAYING"):
            break
        print("Waiting for players...")
        time.sleep(0.5)

    #input("Press ENTER to send START...")
    rpc({"op": "start", "game_id": game_id, "player_id": player_id})
    print("Sent START")

    # Wait until PLAYING
    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if st.get("state", {}).get("status") == "PLAYING":
            break
        time.sleep(0.5)

    print("=== GAME STARTED ===\n")

    last_move_seen = 0

    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if not st.get("ok"):
            print("Error:", st.get("error"))
            return

        state = st["state"]

        # Initialize GUI on first state
        if gui is None:
            pin_list = create_pins_from_state(board, state)
            gui = BoardGUI(board, pin_list)
            gui.root.update_idletasks()
            gui.root.update()
        else:
            # Update GUI with current state
            pin_list = create_pins_from_state(board, state)
            gui.refresh(pin_list)

        # Timeout messages
        if state.get("turn_timeout_notice") and timeoutnotice_move< state.get("move_count"):
            print("⚠ TIMEOUT:", state["turn_timeout_notice"])
            timeoutnotice_move =  state.get("move_count")


        # Finished?
        if state["status"] == "FINISHED":
            print("\n=== GAME FINISHED ===")
            print("FINAL SCORES:")
            for pl in state["players"]:
                sc = pl.get("score")
                if sc:
                    print(
                        f"{pl['name']} ({pl['colour']}): "
                        f"{sc['final_score']:.1f} "
                        f"[time={sc['time_score']:.1f}, "
                        f"moves({sc['moves']})={sc['move_score']:.1f}, "
                        f"pins={sc['pin_goal_score']:.1f}, "
                        f"dist={sc['distance_score']:.1f}]"
                    )
            print("======================")
            break

        # Render board from JSON-only
        

        # Show last move
        if state["move_count"] > last_move_seen:
            mv = state.get("last_move")
            if mv:
                print(
                    f"MOVE: {mv['by']} ({mv['colour']}) "
                    f"{mv['from']}→{mv['to']}  [{mv['move_ms']:.1f}ms]"
                )
            last_move_seen = state["move_count"]

        
        # If it's our turn, choose a random move
        if state.get("current_turn_colour") == colour and state["status"] == "PLAYING":
            print("\nMy turn")
            '''------------PLAYING LOGIC-----------'''
            # Request legal moves for each pin from server
            legal_req = rpc({
                "op": "get_legal_moves",
                "game_id": game_id,
                "player_id": player_id
            })

            if not legal_req.get("ok"):
                print("Error requesting legal moves:", legal_req.get("error"))
                time.sleep(0.5)
                continue

            legal_moves = legal_req.get("legal_moves", {})

            # legal_moves example structure:
            # { pin_id: [to_index1, to_index2, ...], ... }

            movable = [(pid, moves) for pid, moves in legal_moves.items() if moves]
            if not movable:
                print("No legal moves available.")
                time.sleep(0.5)
                continue

            observation = {"colour": colour,
                           "state": state,
                           "legal_moves": legal_moves}
            
            try:
                pid, to_index = policy.select_action(observation)
            except Exception as e:
                print("Policy error, falling back to random move: ", e)
                pid, moves = random.choice(movable)
                to_index = random.choice(moves)
            '''-----------------PLAYING LOGIC----------------'''

            mv = rpc({
                "op": "move",
                "game_id": game_id,
                "player_id": player_id,
                "pin_id": pid,
                "to_index": to_index
            })
            render_json_board(state)
            if not mv.get("ok"):
                print("Move rejected:", mv.get("error"))
            else:
                if mv.get("status") == "WIN":
                    print("YOU WIN!")
                    print(mv.get("msg"))
                elif mv.get("status") == "DRAW":
                    print("DRAW")
                    print(mv.get("msg"))

        time.sleep(0.5)


if __name__ == "__main__":
    main()