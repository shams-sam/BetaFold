import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
import numpy as np

def plot_protein_io(X, Y, n, file_name):
    c = 5
    r = n//c + bool(n%c)
    fig = figure(num=None, figsize=(c*5, r*4),
                 facecolor='w', frameon=True, edgecolor='k')
    print('')
    print(f'Generating seaborn plots. Saving: {file_name}_{n}.png')
    assert n <= X.shape[0]+1
    for i in range(n-1):
        plt.subplot(r, c, i + 1)
        sns.heatmap(X[i, :, :], cmap='RdYlBu')
        plt.title('Channel ' + str(i))
    plt.subplot(r, c, n)
    plt.grid(None)
    y = np.copy(Y)
    y[y > 25.0] = 25.0
    sns.heatmap(y, cmap='Spectral')
    plt.title('True Distances')
    plt.savefig(f'{file_name}_{n}.png', bbox_inches='tight', dpi=100)
    plt.savefig(f'{file_name}_{n}.pdf', bbox_inches='tight', dpi=300)


def plot_protein_yy(Y1s, Y2s, plt_out):
    c = 2
    r = len(Y1s)
    assert len(Y1s) == len(Y2s)

    file_name = f'{plt_out}/distogram_{r}'
    fig = figure(num=None, figsize=(c*5, r*4),
                 facecolor='w', frameon=True, edgecolor='k')
    print('')
    print(f'Generating seaborn plots. Saving: {file_name}.png')
    for i in range(r):
        Y1 = np.copy(Y1s[i])
        Y1[Y1 > 25.0] = 25.0
        Y2 = np.copy(Y2s[i])
        Y2[Y2 > 25.0] = 25.0
        plt.subplot(r, c, 2*i + 1)
        sns.heatmap(Y1[0, :, :], vmin=0, vmax=16, cmap='Spectral')
        if i==0:
            plt.title('True Distances')
        plt.subplot(r, c, 2*i + 2)
        sns.heatmap(Y2[0, :, :], vmin=0, vmax=16, cmap='Spectral')
        if i==0:
            plt.title('Predicted Distances')
    plt.savefig(f'{file_name}.png', bbox_inches='tight', dpi=100)
    plt.savefig(f'{file_name}.pdf', bbox_inches='tight', dpi=300)


def plot_protein_scatter(Y1s, Y2s, plt_out):
    c = 2
    r = len(Y1s) // c + bool(len(Y1s) % c)
    assert len(Y1s) == len(Y2s)

    file_name = f'{plt_out}/scatter_{r}'
    fig = figure(num=None, figsize=(c*5, r*4),
                 facecolor='w', frameon=True, edgecolor='k')
    print('')
    print(f'Generating seaborn plots. Saving: {file_name}.png')
    for i in range(len(Y1s)):
        Y1 = np.copy(Y1s[i]).flatten()
        Y1[Y1 > 25.0] = 25.0
        Y2 = np.copy(Y2s[i]).flatten()
        Y2[Y2 > 25.0] = 25.0
        plt.subplot(r, c, i+1)
        plt.scatter(Y1, Y2, color=
'b', alpha=0.5)
        plt.xlabel('actual distances')
        plt.ylabel('pred. distances')
    plt.savefig(f'{file_name}.png', bbox_inches='tight', dpi=100)
    plt.savefig(f'{file_name}.pdf', bbox_inches='tight', dpi=300)

def plot_learning_curves(history, plt_out):
    print('')
    print('Plotting learning curves..')
    c, r = 2, 1
    fig = figure(num=None, figsize=(c*5, r*4),
                 facecolor='w', frameon=True, edgecolor='k')
    if 'mae' in history['train']:
        ax = plt.subplot(r, c, 1)
        ax.plot(history['train']['mae'], 'g', label='train')
        ax.plot(history['valid']['mae'], 'b', label='valid')
        ax.set_xlabel('epochs')
        ax.set_ylabel('MAE')
    if 'acc' in history['train']:
        ax = plt.subplot(r, c, 1)
        ax.plot(history['train']['acc'], 'g', label='train')
        ax.plot(history['valid']['acc'], 'b', label='valid')
        ax.set_xlabel('epochs')
        ax.set_ylabel('accuracy')

    ax = plt.subplot(r, c, 2)
    ax.plot(history['train']['loss'], 'g', label='train')
    ax.plot(history['valid']['loss'], 'b', label='valid')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')

    ax.legend()
    plt.savefig(f'{plt_out}/training.png', bbox_inches='tight', dpi=100)
    plt.savefig(f'{plt_out}/training.pdf', bbox_inches='tight', dpi=300)

def plot_four_pair_maps(T, P, pdb_list, length_dict):
    figure(num=None, figsize=(24, 10), dpi=60, facecolor='w', frameon=True, edgecolor='k')
    I = 1
    for k in range(4):
        L = length_dict[pdb_list[k]]
        plt.subplot(2, 4, I)
        I += 1
        sns.heatmap(T[k, 0:L, 0:L, 0], cmap='Spectral')
        plt.title('True - ' + pdb_list[k])
    for k in range(4):
        L = length_dict[pdb_list[k]]
        plt.subplot(2, 4, I)
        I += 1
        sns.heatmap(P[k, 0:L, 0:L, 0], cmap='Spectral')
        plt.title('Prediction - ' + pdb_list[k])
    plt.show()

def plot_channel_histograms(X):
    for i in range(len(x[0, 0, :])):
        print ('Input feature', i)
        plt.hist(x[:, :, i].flatten())
        plt.show()
    print('Output labels')
    plt.hist(y.flatten())
    plt.show()
