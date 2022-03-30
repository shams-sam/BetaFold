import os
import sys
from collections import defaultdict
import datetime
import numpy as np
from timeit import default_timer as timer
import torch
import torch.nn as nn
from torchinfo import summary
from tqdm import tqdm

from common import get_args
from dataset import get_loader
from dataset.io import load_list, summarize_channels
# from metrics import *
from models import deepcon_rdd_distances
from models.losses import log_cosh
from viz.plots import plot_learning_curves, plot_protein_io

start = timer()

args = get_args()
dev_size = args.dev_size
training_window = args.training_window
training_epochs = args.training_epochs
lr = args.lr
arch_depth = args.arch_depth
filters_per_layer = args.filters_per_layer
dir_dataset = args.dir_dataset
dir_out = args.dir_out
plt_out = f'{dir_out}/plots'
ckpt_out = f'{dir_out}/ckpt'
file_weights = f'{dir_out}/{args.file_weights}'
flag_eval_only = False
if args.flag_eval_only == 1:
    flag_eval_only = True
pad_size = 10
batch_size = args.batch_size
n_channels = 57
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print('*'*80)
print('Start ' + str(datetime.datetime.now()))
print('is cuda available?:', torch.cuda.is_available())
print('*'*80)

print('')
print('Parameters:')
print('dev_size', dev_size)
print('file_weights', file_weights)
print('training_window', training_window)
print('training_epochs', training_epochs)
print('arch_depth', arch_depth)
print('filters_per_layer', filters_per_layer)
print('pad_size', pad_size)
print('batch_size', batch_size)
print('dir_dataset', dir_dataset)
print('dir_out', dir_out)
print('')
print('*'*80)

os.system(f'mkdir -p {dir_out}')
os.system(f'mkdir -p {plt_out}')
os.system(f'mkdir -p {ckpt_out}')

all_feat_paths = [
    f'{dir_dataset}/deepcov/features/',
    f'{dir_dataset}/psicov/features/',
    f'{dir_dataset}/cameo/features/'
]
all_dist_paths = [
    f'{dir_dataset}/deepcov/distance/',
    f'{dir_dataset}/psicov/distance/',
    f'{dir_dataset}/cameo/distance/'
]

deepcov_list = load_list(f'{dir_dataset}/deepcov.lst', dev_size)

length_dict = {}
for pdb in deepcov_list:
    # ly: sequence length
    # seqy: amino acid sequence
    # pairwise carbon CB distance
    (ly, seqy, cb_map) = np.load(
        f'{dir_dataset}/deepcov/distance/{pdb}-cb.npy', allow_pickle = True)
    length_dict[pdb] = ly

print('')
print('Split into training and validation set..')
valid_pdbs = deepcov_list[:int(0.3 * len(deepcov_list))]
train_pdbs = deepcov_list[int(0.3 * len(deepcov_list)):]
if len(deepcov_list) > 200:
    valid_pdbs = deepcov_list[:100]
    train_pdbs = deepcov_list[100:]

print('Total validation proteins : ', len(valid_pdbs))
print('Total training proteins   : ', len(train_pdbs))

# print('')
# print('Validation proteins: ', valid_pdbs)

train_loader = get_loader(name='dist', batch_size=batch_size, shuffle=True,
                          pdb_id_list=train_pdbs,
                          features_path=all_feat_paths,
                          distmap_path=all_dist_paths,
                          dim=training_window,
                          pad_size=pad_size,
                          n_channels=n_channels,
                          label_engineering='16.0')
valid_loader = get_loader(name='dist', batch_size=batch_size, shuffle=True,
                          pdb_id_list=valid_pdbs,
                          features_path=all_feat_paths,
                          distmap_path=all_dist_paths,
                          dim=training_window,
                          pad_size=pad_size,
                          n_channels=n_channels,
                          label_engineering='16.0')

print('')
print(f'len(train_dataset) : {len(train_loader.dataset)}')
print(f'len(valid_dataset) : {len(valid_loader.dataset)}')

X, Y = next(iter(train_loader))
print('Actual shape of X    : ' + str(X.shape))
print('Actual shape of Y    : ' + str(Y.shape))

# print('')
# print('Channel summaries:')
# summarize_channels(X[0, :, :, :], Y[0])


# larger the size of n, the longer it would take
# 1 <= n <= 58 (58 = num_input_channels + output)
plot_io = f'{plt_out}/io'
plot_protein_io(X[0, :, :, :], Y[0, 0, :, :], 5, plot_io)

print('')
print('Build a model..')
model = deepcon_rdd_distances(training_window, arch_depth,
                              filters_per_layer, n_channels)
model = model.to(device)

# do training
if flag_eval_only == 0:
    if os.path.exists(file_weights):
        print('')
        print('Loading existing weights..')
        model.load_state_dict(torch.load(file_weights))

    print('')
    print('Train..')
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    l1_loss = nn.L1Loss()
    history = {
        'epoch': [],
        'train': defaultdict(list),
        'valid': defaultdict(list),
    }
    metric = 'mae'
    best_metric = 100.0
    best_epoch = 0 
    for epoch in range(training_epochs):

        # model.train()
        for x, y in tqdm(train_loader, total=len(train_loader), leave=False):
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            train_loss, train_mae = model.loss_fn(y, y_hat)
            optim.zero_grad()
            train_loss.backward()
            optim.step()

        # model.eval()
        val_loss, val_mae = 0, 0
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss, mae = model.loss_fn(y, y_hat)
            val_loss += loss.item()
            val_mae += mae.item()
        val_loss /= len(valid_loader)
        val_mae /= len(valid_loader)

        history['epoch'].append(epoch)
        history['train']['loss'].append(train_loss.item())
        history['train']['mae'].append(train_mae.item())
        history['valid']['loss'].append(val_loss)
        history['valid']['mae'].append(val_mae)
        print("epoch {}/{}: \t train - loss:{:.4f}, mae:{:.4f} "
              "\t valid - loss:{:.4f}, mae:{:.4f}".format(
                  epoch, training_epochs,
                  train_loss.item(), train_mae.item(),
                  val_loss, val_mae
        ))

        if best_metric > history['valid'][metric][-1]:
            best_metric = history['valid'][metric][-1]
            best_epoch = epoch
            torch.save(model.state_dict(), f'{ckpt_out}/model.pt')

    print(f'best model @ epoch: {best_epoch} saved at: {ckpt_out}/model.pt')
    plot_learning_curves(history, plt_out)

end = timer()
delta = datetime.timedelta(seconds=end-start)
print('')
print('*'*80)
print(f'time to train: {delta}')
print('*'*80)
exit()


psicov_list = load_list(dir_dataset + 'psicov.lst')
psicov_length_dict = {}
for pdb in psicov_list:
    (ly, seqy, cb_map) = np.load(dir_dataset + '/psicov/distance/' + pdb + '-cb.npy', allow_pickle = True)
    psicov_length_dict[pdb] = ly

cameo_list = load_list(dir_dataset + 'cameo-hard.lst')
cameo_length_dict = {}
for pdb in cameo_list:
    (ly, seqy, cb_map) = np.load(dir_dataset + '/cameo/distance/' + pdb + '-cb.npy', allow_pickle = True)
    cameo_length_dict[pdb] = ly

evalsets = {}
#evalsets['validation'] = {'LMAX': 512,  'list': valid_pdbs, 'lendict': length_dict}
evalsets['psicov'] = {'LMAX': 512,  'list': psicov_list, 'lendict': psicov_length_dict}
evalsets['cameo']  = {'LMAX': 1300, 'list': cameo_list,  'lendict': cameo_length_dict}

for my_eval_set in evalsets:
    print('')
    print(f'Evaluate on the {my_eval_set} set..')
    my_list = evalsets[my_eval_set]['list']
    LMAX = evalsets[my_eval_set]['LMAX']
    length_dict = evalsets[my_eval_set]['lendict']
    print('L', len(my_list))
    print(my_list)

    model = deepcon_rdd_distances(LMAX, arch_depth, filters_per_layer, n_channels)
    model.load_weights(file_weights)
    my_generator = DistGenerator(my_list, all_feat_paths, all_dist_paths, LMAX, pad_size, 1, n_channels, label_engineering = None)

    # Padded but full inputs/outputs
    P = model.predict_generator(my_generator, max_queue_size=10, verbose=1)
    Y = np.full((len(my_generator), LMAX, LMAX, 1), np.nan)
    for i, xy in enumerate(my_generator):
        Y[i, :, :, 0] = xy[1][0, :, :, 0]
    # Average the predictions from both triangles
    for j in range(0, len(P[0, :, 0, 0])):
        for k in range(j, len(P[0, :, 0, 0])):
            P[ :, j, k, :] = (P[ :, k, j, :] + P[ :, j, k, :]) / 2.0
    P[ P < 0.01 ] = 0.01
    # Remove padding, i.e. shift up and left by int(pad_size/2)
    P[:, :LMAX-pad_size, :LMAX-pad_size, :] = P[:, int(pad_size/2) : LMAX-int(pad_size/2), int(pad_size/2) : LMAX-int(pad_size/2), :]
    Y[:, :LMAX-pad_size, :LMAX-pad_size, :] = Y[:, int(pad_size/2) : LMAX-int(pad_size/2), int(pad_size/2) : LMAX-int(pad_size/2), :]
    # Recover the distance translations
    #P = 100.0 / (P + epsilon)
    print('')
    print('Evaluating distances..')
    results_list = evaluate_distances(P, Y, my_list, length_dict)
    print('')
    numcols = len(results_list[0].split())
    print(f'Averages for {my_eval_set}', end = ' ')
    for i in range(2, numcols):
        x = results_list[0].split()[i].strip()
        if x == 'count' or results_list[0].split()[i-1].strip() == 'count':
            continue
        avg_this_col = False
        if x == 'nan':
            avg_this_col = True
        try:
            float(x)
            avg_this_col = True
        except ValueError:
            None
        if not avg_this_col:
            print(x, end=' ')
            continue
        avg = 0.0
        count = 0
        for mrow in results_list:
            a = mrow.split()
            if len(a) != numcols:
                continue
            x = a[i]
            if x == 'nan':
                continue
            try:
                avg += float(x)
                count += 1
            except ValueError:
                print(f'ERROR!! float value expected!! {x}')
        print(f'AVG: {avg/count:.4f} items={count}', end = ' ')
    print('')

    if flag_plots:
        plot_four_pair_maps(Y, P, my_list, my_length_dict)

    print('')
    print('Save predictions..')
    for i in range(len(my_list)):
        L = length_dict[my_list[i]]
        np.save(dir_out + '/' + my_list[i] + '.npy', P[i, :L, :L, 0])

print('')
print ('Everything done! ' + str(datetime.datetime.now()) )
