# RLChineseCheckers
For IKT 460 (RL) base Chinese Checkers


# Rules
- each player starts with their colored pieces on one of the six points or corners of the star and attempts to race them all home into the opposite corner
- Players take turns moving a single piece,
  - either by moving one step in any direction to an adjacent empty space,
  - or by jumping in one or any number of available consecutive hops over other single pieces.
- A player may not combine hopping with a single-step move :: a move consists of one or the other.
- A pin maynot go into a 'coloured triangle' unless it is either its source or its destination
- There is no capturing

## single system
Designed to be played on a single system (no separate teams joining from separate systems)

Run : python checkers_main.py

You'll be prompted to type 'assign' to assign a random colour (without the quotes)
```
Type 'exit' to exit. Type 'assign' and press enter for first player:
```
Again for the next player
```
Type 'assign' and press enter for second player:
```

You'll see the updated ascii board on the commandline screen everytime there is a change.

Next prompt:
```
Type 'assign' and press enter for more players, else type 'start game':
```

Once you enter 'start_game', you'll see
  - Tkinter version in a separate window
  - *colour* 's current positions : format (pin_id, pin_axialindex) *list of (id, position) of that colour*
  - If 'help_mode = True' under checkers_main.py, you'll be prompted:
    - ```
      Helpmode:Which pin will you move?:
      ```
    - Enter a pin number, e.g: 5
    - ```
      Possible moves: *list of possible axial positions that the given pin can go to*
      ```
    - ```
      Need more help? Yes/No
      ```
    - If 'Yes'
      - Helpmode continues
    - If 'No', 
      - ```
        Enter *colour* 's Move: (pin_number, dest_axial):
        ```
      - enter the move **with** '(' ,')', eg: (2,66)
      - If it was a valid move, see update on Ascii/Tkinter, and control moves to the next player.
      - Else, current player's turn continues
     
  # Current Limitations/Expected Changes
  - There is no logic implemented yet for checking if a player has won - Expect changes
  - Expect changes in Starting a Game/Assigning colours.
  - There is no logic implemented yet for storing moves made - Expect changes
  - There is no logic implemented yet to indicate 'Pass' on a turn - Expect changes
  - There is no logic implemented yet to timeout a players turn - Expect changes
  - There is no logic implemented yet for scoring - Expect changes (will be based on time taken, total number of moves, number of pins successfully moved to opposite colour)
  - There is no logic implemented yet for maximum allowable time per game - Expect changes
