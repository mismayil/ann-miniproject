import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

from qlearner import QStateAction

def plot_average_rewards(stats_path, labels, log_every=250, save_path=None):
    try:
        with open(stats_path, 'rb') as npy:
            player_stats = np.load(npy, allow_pickle=True)
    except:
        print('File not found!')
        raise
        
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    game_ids = list(range(0, len(player_stats[0]['reward'])*log_every, log_every))
    for i, player in enumerate(player_stats):
        label = labels[i]
        ax.plot(game_ids, player['reward'], label=label)
    
    ax.set_title(f'Average reward per {log_every} games', fontsize=20, fontweight='bold')
    ax.set_xlabel('Game', fontsize=16)
    ax.set_ylabel('Reward', fontsize=16)
    ax.set_xlim([0, len(game_ids)*log_every])
    ax.legend(loc='lower right')
    ax.grid()

    plt.show()
    
    if save_path is not None: fig.savefig(save_path, format='pdf')

def plot_m_values(stats_path, labels, test_every=250, save_path=None):
    try:
        with open(stats_path, 'rb') as npy:
            player_stats = np.load(npy, allow_pickle=True)
    except:
        print('File not found!')
        raise
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    game_ids = list(range(0, len(player_stats[0]['reward'])*test_every, test_every))
    for i, player in enumerate(player_stats):
        label = labels[i]
        axes[0].plot(game_ids, player["m_opt"], label=label)
        axes[1].plot(game_ids, player["m_rand"], label=label)

        
    axes[0].set_title(f'M_opt per {test_every} games', fontsize=20, fontweight='bold')
    axes[0].set_xlabel('Game', fontsize=16)
    axes[0].set_ylabel('M_opt', fontsize=16)
    axes[0].set_xlim([0, len(game_ids)*test_every])
    axes[0].legend()
    axes[0].grid()
    
    axes[1].set_title(f'M_rand per {test_every} games', fontsize=20, fontweight='bold')
    axes[1].set_xlabel('Game', fontsize=16)
    axes[1].set_ylabel('M_rand', fontsize=16)
    axes[1].set_xlim([0, len(game_ids)*test_every])
    axes[1].legend()
    axes[1].grid()

    plt.show()
    
    if save_path is not None: fig.savefig(save_path, format='pdf')

def read_grid(grid):
    empty = list(zip(*np.where(grid==0)))
    filled = list(zip(*np.where(grid!=0)))
    return empty, filled

def plot_heatmaps(states_list, qvalues_list, titles, cmap=None, save_path=None):
    fig = plt.figure(figsize=(15, 5))
    axes = AxesGrid(fig, 111,
                nrows_ncols=(1, 3),
                axes_pad=0.2,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.1
                )
    
    if cmap is None: cmap = plt.cm.get_cmap('Blues', 10)
    
    for ax, state, qvalues, title in zip(axes, states_list, qvalues_list, titles):
        
        empty, filled = read_grid(state) # Read the configuration of the grid
        qvalue_grid = -1*np.ones((3,3))

        for action in empty:
            qstate = QStateAction(state, action)
            qvalue_grid[action] = qvalues.get(qstate, 0)
        

        img = ax.imshow(qvalue_grid, cmap=cmap, vmin=-1, vmax=1)
        ax.set_axis_off()

        for i, j in empty:
            qval = qvalue_grid[i, j]
            text = ax.text(j, i, f'{qval:.3f}', ha='center', va='center', color='k', fontsize=14)

        for i, j in filled:
            player = 'X' if state[i, j] == 1 else 'O'
            text = ax.text(j, i, f'{player}', ha='center', va='center', color='k', fontsize=16, fontweight='bold')
            
        
        ax.set_title(title, fontsize=14, fontweight='bold')
            
            
    cbar = ax.cax.colorbar(img)
    cbar = axes.cbar_axes[0].colorbar(img)


    if save_path is not None: fig.savefig(save_path, format='pdf')