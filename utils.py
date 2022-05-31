import numpy as np
from tic_env import TictactoeEnv, InvalidMoveError, OptimalPlayer
from tqdm import tqdm
import random

def play(player1, player2, episodes=5, debug=False, first_player="alternate", disable_tqdm=False, seed=None):
    env = TictactoeEnv()
    Turns = np.array(['X','O'])
    player1_stats = {'wins': 0, 'losses': 0, 'M': 0}
    player2_stats = {'wins': 0, 'losses': 0, 'M': 0}
    
    if seed is not None: 
        # Set a seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)

    for i in tqdm(range(episodes), disable=disable_tqdm):
        env.reset()
        grid, _, __ = env.observe()
        invalid_player = None

        if first_player == "alternate":
            Turns = np.flip(Turns)
        else:
            Turns = Turns[np.random.permutation(2)]

        player1.set_player(Turns[0])
        player2.set_player(Turns[1])

        if debug:
            print('-------------------------------------------')
            print(f"Game {i}, Player 1 = {Turns[0]}, Player 2 = {Turns[1]}")

        for j in range(9):
            if env.current_player == player1.player:
                move = player1.act(grid)
            else:
                move = player2.act(grid)

            try:
                grid, end, winner = env.step(move, print_grid=False)
            except InvalidMoveError:
                # If wrong move is played, penalize the player
                end = True
                invalid_player = player1.player if env.current_player == player1.player else player2.player
                winner = None

            if end:
                if hasattr(player1, 'end'):
                    player1.end(grid, winner, invalid_move=(invalid_player==player1))
                
                if hasattr(player2, 'end'):
                    player2.end(grid, winner, invalid_move=(invalid_player==player2))

                if winner == player1.player:
                    player1_stats['wins'] += 1
                    player2_stats['losses'] += 1
                elif winner == player2.player:
                    player1_stats['losses'] += 1
                    player2_stats['wins'] += 1
                
                if debug:
                    print('-------------------------------------------')
                    print('Game end, winner is player ' + str(winner))
                    print('Player 1 = ' +  Turns[0])
                    print('Player 2 = ' +  Turns[1])
                    env.render()
                    print('-------------------------------------------')
                
                break
    
    player1_stats['M'] = (player1_stats['wins'] - player1_stats['losses']) / episodes
    player2_stats['M'] = (player2_stats['wins'] - player2_stats['losses']) / episodes
    return player1_stats, player2_stats


def print_qstate(qstate):
    env = TictactoeEnv()
    env.grid = qstate.grid
    env.render()
    print(f"Next action: {qstate.action}")


def calculate_m_opt(q_player, episodes=500):
    if hasattr(q_player, 'eval'): q_player.eval()
    optimal_player = OptimalPlayer(epsilon=0.0)
    player1_stats, _ = play(q_player, optimal_player, episodes=episodes, debug=False, first_player='alternate', disable_tqdm=True)
    if hasattr(q_player, 'train'): q_player.train()
    return player1_stats['M']

def calculate_m_rand(q_player, episodes=500):
    if hasattr(q_player, 'eval'): q_player.eval()
    random_player = OptimalPlayer(epsilon=1.0)
    player1_stats, _ = play(q_player, random_player, episodes=episodes, debug=False, first_player='alternate', disable_tqdm=True)
    if hasattr(q_player, 'train'): q_player.train()
    return player1_stats['M']