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
from models import get_model
from models.losses import log_cosh
from viz.plots import plot_learning_curves, plot_protein_io, \
    plot_protein_scatter, plot_protein_yy

start = timer()

args = get_args()
dist_type = args.type
dev_size = args.dev_size
training_window = args.training_window
training_epochs = args.training_epochs
lr = args.lr
inv_loss = args.inv_loss
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

train_loader = get_loader(name=dist_type, batch_size=batch_size, shuffle=True,
                          pdb_id_list=train_pdbs,
                          features_path=all_feat_paths,
                          distmap_path=all_dist_paths,
                          dim=training_window,
                          pad_size=pad_size,
                          n_channels=n_channels,
                          label_engineering='100/d')
valid_loader = get_loader(name=dist_type, batch_size=batch_size, shuffle=True,
                          pdb_id_list=valid_pdbs,
                          features_path=all_feat_paths,
                          distmap_path=all_dist_paths,
                          dim=training_window,
                          pad_size=pad_size,
                          n_channels=n_channels,
                          label_engineering='100/d')

print('')
print(f'len(train_dataset) : {len(train_loader.dataset)}')
print(f'len(valid_dataset) : {len(valid_loader.dataset)}')

X, Y = next(iter(train_loader))
print('Actual shape of X    : ' + str(X.shape))
print('Actual shape of Y    : ' + str(Y.shape))
n_bins = Y.shape[1]

# print('')
# print('Channel summaries:')
# summarize_channels(X[0, :, :, :], Y[0])


# larger the size of n, the longer it would take
# 1 <= n <= 58 (58 = num_input_channels + output)
if flag_eval_only == 0:
    plot_io = f'{plt_out}/io'
    plot_protein_io(X[0, :, :, :], Y[0, 0, :, :], 5, plot_io)

print('')
print('Build a model..')
model = get_model(dist_type, L=training_window, num_blocks=arch_depth,
                  width=filters_per_layer, n_bins=n_bins, n_channels=n_channels)
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
    metric = model.get_metric()
    if metric == 'mae':
        best_metric = 100.0
    else:
        best_metric = 0.0
    best_epoch = 0 
    for epoch in range(training_epochs):

        # model.train()
        for x, y in tqdm(train_loader, total=len(train_loader), leave=False):
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            train_loss, train_metric = model.loss_fn(y, y_hat, inv_loss)
            optim.zero_grad()
            train_loss.backward()
            optim.step()

        # model.eval()
        val_loss, val_metric = 0, 0
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            iter_loss, iter_metric = model.loss_fn(y, y_hat, inv_loss)
            val_loss += iter_loss.item()
            val_metric += iter_metric.item()
        val_loss /= len(valid_loader)
        val_metric /= len(valid_loader)

        history['epoch'].append(epoch)
        history['train']['loss'].append(train_loss.item())
        history['train'][metric].append(train_metric.item())
        history['valid']['loss'].append(val_loss)
        history['valid'][metric].append(val_metric)
        print("epoch {}/{}: \t train - loss:{:.4f}, {}:{:.4f} "
              "\t valid - loss:{:.4f}, {}:{:.4f}".format(
                  epoch, training_epochs,
                  train_loss.item(), metric, train_metric.item(),
                  val_loss, metric, val_metric
        ))

        if (metric == 'mae' and best_metric > history['valid'][metric][-1]) or \
           (metric == 'acc' and best_metric < history['valid'][metric][-1]):
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

else:
    print('*'*80)
    print('Skipping training')
    print('*'*80)


if os.path.exists(file_weights):
    print('')
    print('Loading existing weights..')
    model.load_state_dict(torch.load(file_weights))
else:
    print('')
    print('Model not found. Quiting..')
    exit()

n = 1

y1s, y2s = [], []
l1_loss = nn.L1Loss()

mae = 0.0
for x, y in tqdm(valid_loader, total=len(valid_loader)):
    x, y = x.to(device), y.to(device)
    y_hat = model(x)
    y = 100.0/y
    y_hat = 100.0/y
    mae += l1_loss(y, y_hat).item()
    for i in range(y_hat.size(0)):
        if len(y1s) >= n:
            break
        y1s.append(y[i].cpu())
        y2s.append(y_hat[i].detach().cpu())
print('valid: ', mae/len(train_loader))


plot_protein_yy(y1s, y2s, plt_out)

print('')
print ('Everything done! ' + str(datetime.datetime.now()) )
