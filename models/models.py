import torch
from torch import nn
import torch.nn.functional as F

from .losses import log_cosh, inv_log_cosh

# A basic fully convolutional network
# def basic_fcn(L, num_blocks, width, n_channels):
#     input_shape = (L, L, n_channels)
#     img_input = layers.Input(shape = input_shape)
#     x = img_input
#     for i in range(num_blocks):
#         x = layers.Conv2D(width, (3, 3), padding = 'same')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#     x = layers.Conv2D(1, (3, 3), padding = 'same', kernel_initializer = 'one')(x)
#     x = layers.Activation('relu')(x)
#     inputs = img_input
#     model = tf.keras.models.Model(inputs, x, name = 'fcn')
#     return model

# Architecture DEEPCON (original)
# def deepcon_rdd(L, num_blocks, width, n_channels):
#     print('')
#     print('Model params:')
#     print('L', L)
#     print('num_blocks', num_blocks)
#     print('width', width)
#     print('n_channels', n_channels)
#     print('')
#     dropout_value = 0.3
#     my_input = Input(shape = (L, L, n_channels))
#     tower = BatchNormalization()(my_input)
#     tower = Activation('relu')(tower)
#     tower = Convolution2D(width, 1, padding = 'same')(tower)
#     n_channels = width
#     d_rate = 1
#     for i in range(num_blocks):
#         block = BatchNormalization()(tower)
#         block = Activation('relu')(block)
#         block = Convolution2D(n_channels, kernel_size = (3, 3), padding = 'same')(block)
#         block = Dropout(dropout_value)(block)
#         block = Activation('relu')(block)
#         block = Convolution2D(n_channels, kernel_size = (3, 3), dilation_rate=(d_rate, d_rate), padding = 'same')(block)
#         tower = add([block, tower])
#         if d_rate == 1:
#             d_rate = 2
#         elif d_rate == 2:
#             d_rate = 4
#         else:
#             d_rate = 1
#     tower = BatchNormalization()(tower)
#     tower = Activation('relu')(tower)
#     tower = Convolution2D(1, 3, padding = 'same')(tower)
#     tower = Activation('sigmoid')(tower)
#     model = Model(my_input, tower)
#     return model

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
        p_rate = 1
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


def deepcon_rdd_distances(L, num_blocks, width, n_channels):
    print('')
    print('Model params:')
    print('L', L)
    print('num_blocks', num_blocks)
    print('width', width)
    print('n_channels', n_channels)
    print('')

    return DeepConDistances(L, num_blocks, width, n_channels)

# Architecture DEEPCON (binned)
# def deepcon_rdd_binned(L, num_blocks, width, bins, n_channels):
#     print('')
#     print('Model params:')
#     print('L', L)
#     print('num_blocks', num_blocks)
#     print('width', width)
#     print('n_channels', n_channels)
#     print('')
#     dropout_value = 0.3
#     my_input = Input(shape = (L, L, n_channels))
#     tower = BatchNormalization()(my_input)
#     tower = Activation('relu')(tower)
#     tower = Convolution2D(width, 1, padding = 'same')(tower)
#     n_channels = width
#     d_rate = 1
#     for i in range(num_blocks):
#         block = BatchNormalization()(tower)
#         block = Activation('relu')(block)
#         block = Convolution2D(n_channels, kernel_size = (3, 3), padding = 'same')(block)
#         block = Dropout(dropout_value)(block)
#         block = Activation('relu')(block)
#         block = Convolution2D(n_channels, kernel_size = (3, 3), dilation_rate=(d_rate, d_rate), padding = 'same')(block)
#         tower = add([block, tower])
#         if d_rate == 1:
#             d_rate = 2
#         elif d_rate == 2:
#             d_rate = 4
#         else:
#             d_rate = 1
#     tower = BatchNormalization()(tower)
#     tower = Activation('relu')(tower)
#     tower = Convolution2D(bins, 3, padding = 'same')(tower)
#     tower = Activation('softmax')(tower)
#     model = Model(my_input, tower)
#     return model
