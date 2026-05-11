#Given a list of names ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'] create a list of games involving the players. 
#Each game can have between 2 and 6 players. 
#Organize the list in terms of Rounds. Each round should have a list of games, and each game should be a list of player names. 
#Ensure that no player plays against the same opponent more than once across all rounds.
#Each player should play only once in each round, and the total number of games for each player should be limited to number of rounds.
from itertools import combinations
import random
player_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
number_of_rounds = 5

def generate_games(players):
    rounds = []
    all_combinations = list(combinations(players, 2))  # Generate all possible pairs of players
    random.shuffle(all_combinations)  # Shuffle to randomize the order of games
    for round_num in range(number_of_rounds):
        round_games = []
        used_players = set()  # Track players used in this round
        for game in all_combinations:
            if game[0] not in used_players and game[1] not in used_players:
                round_games.append(list(game))
                used_players.update(game)  # Mark these players as used
            if len(round_games) >= len(players) // 2:  # Limit number of games per round
                break
        rounds.append(round_games)
    return rounds

r = generate_games(player_list)

#pretty print the round
rnum=1
for ground in r:
    gnum=1
    for game in ground:
        print(f"Round {rnum} Game {gnum}",game)
        gnum+=1
    rnum+=1
#print("Rounds of games:", r)