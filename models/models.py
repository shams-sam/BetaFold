import torch
from torch import nn
import torch.nn.functional as F

from .losses import log_cosh, inv_log_cosh


# Architecture DEEPCON (original)
class DeepConRDD(nn.Module):
    def __init__(self, L, num_blocks, width, n_channels):
        super().__init__()
        self.L = L
        self.num_blocks = num_blocks
        self.width = width
        self.n_channels = n_channels

        self.input_block = nn.Sequential(
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, width, 1,  padding=0)
        )

        blocks = []
        n_channels = width
        d_rate = 1
        for i in range(num_blocks):
            blocks += [
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_channels, n_channels, 3, padding=1),
                nn.Dropout(0.3),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_channels, n_channels, 3,
                          dilation=d_rate, padding=d_rate)
            ]
            if d_rate == 1:
                d_rate = 2
            elif d_rate == 2:
                d_rate = 4
            else:
                d_rate = 1

        self.mid_block = nn.Sequential(*blocks)

        self.output_block = nn.Sequential(
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, 1, 3, padding=1),
        )

    def forward(self, x):
        x = self.input_block(x)
        x = self.mid_block(x)

        return self.output_block(x)

    def loss_fn(self, y, y_hat, inv=False):
        bce_loss = nn.BCEWithLogitsLoss()
        loss = bce_loss(y_hat, y)
        y_pred = (torch.sigmoid(y_hat) > 0.5).float()
        correct = (y == y_pred).float().sum()
        acc = correct/torch.numel(y)

        return loss, acc

    def get_metric(self):
        return 'acc'


def deepcon_rdd(L, num_blocks, width, n_channels, **kwargs):
    print('')
    print('Model params:')
    print('L', L)
    print('num_blocks', num_blocks)
    print('width', width)
    print('n_channels', n_channels)
    print('')

    return DeepConRDD(L, num_blocks, width, n_channels)


# Architecture DEEPCON (distances)
class DeepConDistances(nn.Module):
    def __init__(self, L, num_blocks, width, n_channels):
        super().__init__()
        self.L = L
        self.num_blocks = num_blocks
        self.width = width
        self.n_channels = n_channels

        self.input_block = nn.Sequential(
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, width, 1,  padding=0)
        )

        blocks = []
        n_channels = width
        d_rate = 1
        for i in range(num_blocks):
            blocks += [
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_channels, n_channels, 3, padding=1),
                nn.Dropout(0.3),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_channels, n_channels, 3,
                          dilation=d_rate, padding=d_rate)
            ]
            if d_rate == 1:
                d_rate = 2
            elif d_rate == 2:
                d_rate = 4
            else:
                d_rate = 1

        self.mid_block = nn.Sequential(*blocks)

        self.output_block = nn.Sequential(
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, 1, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.input_block(x)
        x = self.mid_block(x)

        return self.output_block(x)

    def loss_fn(self, y, y_hat, inv=False):
        l1_loss = nn.L1Loss()
        if inv:
            loss = inv_log_cosh(y, y_hat)
        else:
            loss = log_cosh(y, y_hat)
        mae = l1_loss(y, y_hat)

        return loss, mae

    def get_metric(self):
        return 'mae'


def deepcon_rdd_distances(L, num_blocks, width, n_channels, **kwargs):
    print('')
    print('Model params:')
    print('L', L)
    print('num_blocks', num_blocks)
    print('width', width)
    print('n_channels', n_channels)
    print('')

    return DeepConDistances(L, num_blocks, width, n_channels)

# Architecture DEEPCON (binner)
class DeepConRDDBinned(nn.Module):
    def __init__(self, L, num_blocks, width, bins, n_channels):
        super().__init__()
        self.L = L
        self.num_blocks = num_blocks
        self.width = width
        self.bins = bins
        self.n_channels = n_channels

        self.input_block = nn.Sequential(
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, width, 1,  padding=0)
        )

        blocks = []
        n_channels = width
        d_rate = 1
        for i in range(num_blocks):
            blocks += [
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_channels, n_channels, 3, padding=1),
                nn.Dropout(0.3),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_channels, n_channels, 3,
                          dilation=d_rate, padding=d_rate)
            ]
            if d_rate == 1:
                d_rate = 2
            elif d_rate == 2:
                d_rate = 4
            else:
                d_rate = 1

        self.mid_block = nn.Sequential(*blocks)

        self.output_block = nn.Sequential(
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, bins, 3, padding=1),
        )

    def forward(self, x):
        x = self.input_block(x)
        x = self.mid_block(x)

        return self.output_block(x)

    def loss_fn(self, y, y_hat, inv=False):
        ce_loss = nn.CrossEntropyLoss()
        y = y.argmax(dim=1)
        loss = ce_loss(y_hat.view(torch.numel(y),-1), y.flatten())
        output = y_hat.argmax(dim=1).float()
        correct = (output==y).float().sum()
        acc = correct/torch.numel(y)

        return loss, acc

    def get_metric(self):
        return 'acc'


def deepcon_rdd_binned(L, num_blocks, width, n_bins, n_channels, **kwargs):
    print('')
    print('Model params:')
    print('L', L)
    print('num_blocks', num_blocks)
    print('width', width)
    print('n_bins', n_bins)
    print('n_channels', n_channels)
    print('')

    return DeepConRDDBinned(L, num_blocks, width, n_bins, n_channels)
