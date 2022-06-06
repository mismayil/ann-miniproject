import numpy as np
from tic_env import TictactoeEnv, OptimalPlayer
from tqdm import tqdm
import random


def play(player1, player2, episodes=5, debug=False, first_player="alternate", disable_tqdm=False, seed=None):
    """Play Tic Tact Toe between two players

    Args:
        player1: Player 1
        player2: Player 2
        episodes (int, optional): Number of episodes. Defaults to 5.
        debug (bool, optional): Whether to print debug messages. Defaults to False.
        first_player (str, optional): Strategy to determine first player. Defaults to "alternate".
            "alternate" means alternate between player1 and player2 every game. Otherwise, randomly determined.
        disable_tqdm (bool, optional): Whether to disable progress bar. Defaults to False.
        seed (_type_, optional): Random seed. Defaults to None.

    Returns:
        (dict, dict): Player 1 stats and Player 2 stats (M-values)
    """
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
            except ValueError:
                # If wrong move is played, penalize the player
                end = True
                invalid_player = player1.player if env.current_player == player1.player else player2.player
                winner = None

            if end:
                if hasattr(player1, 'end'):
                    player1.end(grid, winner, invalid_move=(invalid_player==player1.player))
                
                if hasattr(player2, 'end'):
                    player2.end(grid, winner, invalid_move=(invalid_player==player2.player))

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

    if hasattr(player1, 'finish_run'):
        player1.finish_run()

    if hasattr(player2, 'finish_run'):
        player2.finish_run()

    return player1_stats, player2_stats


def calculate_m_opt(q_player, episodes=500):
    """Calculate M_opt for a given player

    Args:
        q_player: Player
        episodes (int, optional): Number of episodes. Defaults to 500.

    Returns:
        float: M_opt value
    """
    # Put player in evaluation mode to avoid training and logging
    if hasattr(q_player, 'eval'): 
        q_player.eval()

    optimal_player = OptimalPlayer(epsilon=0.0)
    player1_stats, _ = play(q_player, optimal_player, episodes=episodes, debug=False, first_player='alternate', disable_tqdm=True)

    # Put player back in training mode
    if hasattr(q_player, 'train'): 
        q_player.train()

    return player1_stats['M']

def calculate_m_rand(q_player, episodes=500):
    """Calculate M_rand for a given player

    Args:
        q_player: Player
        episodes (int, optional): Number of episodes. Defaults to 500.

    Returns:
        float: M_rand value
    """
    # Put player in evaluation mode to avoid training and logging
    if hasattr(q_player, 'eval'): 
        q_player.eval()

    random_player = OptimalPlayer(epsilon=1.0)
    player1_stats, _ = play(q_player, random_player, episodes=episodes, debug=False, first_player='alternate', disable_tqdm=True)

    # Put player back in training mode
    if hasattr(q_player, 'train'): 
        q_player.train()

    return player1_stats['M']

def read_stats(path):
    with open(path, 'rb') as npy:
        return np.load(npy, allow_pickle=True)

def save_stats(players, path):
    """ Save player stats to a file"""
    player_stats = []
    
    for player in players:
        
        stat = {
            'loss': player.avg_losses if hasattr(player, 'avg_losses') else None,
            'reward': player.avg_rewards,
            'm_opt': player.m_values['m_opt'],
            'm_rand': player.m_values['m_rand']
        }
        player_stats.append(stat)
        
    with open(path, 'wb') as npy:
        np.save(npy, player_stats)


def fetch_from_wandb(id_list):
    """ Fetch player stats from wandb"""
    import wandb
    api = wandb.Api()

    player_stats = []

    for run_id in id_list:
        run = api.run(f"mismayil/ann-project/{run_id}")
        history = run.scan_history()

        avg_reward, avg_loss, m_opt, m_rand = [], [], [], []
        for row in history:
            reward = row.get('avg_reward')
            if reward is not None: avg_reward.append(reward)
                
            loss = row.get('avg_loss')
            if loss is not None: avg_loss.append(loss)
                
            opt = row.get('m_opt')
            if opt is not None: m_opt.append(opt)

            rand = row.get('m_rand')
            if rand is not None: m_rand.append(rand)

        player_stats.append({
            'reward': avg_reward,
            'loss': avg_loss,
            'm_opt': m_opt,
            'm_rand': m_rand
        })
        
    return player_stats

def highest_m_values(stats):
    highest_m_opts = []
    highest_m_rands = []

    for stat in stats:
        highest_m_opts.append(max(stat['m_opt']))
        highest_m_rands.append(max(stat['m_rand']))
    
    return {'m_opt': max(highest_m_opts), 'm_rand': max(highest_m_rands)}
