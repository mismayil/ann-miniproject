import numpy as np
from tic_env import TictactoeEnv

def play(player1, player2, episodes=5, debug=False):
    env = TictactoeEnv()
    Turns = np.array(['X','O'])
    player1_stats = {'wins': 0, 'losses': 0}
    player2_stats = {'wins': 0, 'losses': 0}

    for i in range(episodes):
        env.reset()
        grid, _, __ = env.observe()
        Turns = Turns[np.random.permutation(2)]
        player1.set_player(Turns[0])
        player2.set_player(Turns[1])

        for j in range(9):
            if env.current_player == player1.player:
                move = player1.act(grid)
            else:
                move = player2.act(grid)

            grid, end, winner = env.step(move, print_grid=False)

            if end:
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
                
                break
    
    return player1_stats, player2_stats