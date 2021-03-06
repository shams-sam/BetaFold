import numpy as np
from torch.utils.data import Dataset, DataLoader
from .io import get_input_output_bins, get_input_output_dist

class ContactDataset(Dataset):
    def __init__(self, pdb_id_list, features_path, distmap_path,
                 dim, pad_size, n_channels, **kwargs):
        self.pdb_id_list = pdb_id_list
        self.features_path = features_path
        self.dim = dim
        self.pad_size = pad_size
        self.distmap_path = distmap_path
        self.n_channels = n_channels

    def __len__(self):
        return len(self.pdb_id_list)

    def __getitem__(self, index):
        X, Y = get_input_output_dist(self.pdb_id_list[index: index+1],
                                     self.features_path,
                                     self.distmap_path, self.pad_size,
                                     self.dim, self.n_channels)
        Y[Y < 8.0] = 1.0
        Y[Y >= 8.0] = 0.0

        X = np.moveaxis(X, -1, 0)
        Y = np.moveaxis(Y, -1, 0)

        return X, Y

class BinnedDistDataset(Dataset):
    def __init__(self, pdb_id_list, features_path, distmap_path,
                 dim, pad_size, n_channels, **kwargs):
        self.pdb_id_list = pdb_id_list
        self.features_path = features_path
        self.dim = dim
        self.pad_size = pad_size
        self.distmap_path = distmap_path
        self.bins = self._get_bins()
        self.n_channels = n_channels

    def __len__(self):
        return len(self.pdb_id_list)

    def __getitem__(self, index):
        X, Y = get_input_output_bins(self.pdb_id_list[index: index+1],
                                     self.features_path,
                                     self.distmap_path, self.pad_size,
                                     self.dim, self.n_channels,
                                     self.bins)
        X = np.moveaxis(X, -1, 0)
        Y = np.moveaxis(Y, -1, 0)

        return X, Y

    def _get_bins(self):
        bins = {}
        bins[0] = '0.0 4.0'
        b = 1
        range_min = float(bins[0].split()[1])
        interval = 0.2
        while(range_min < 8.0):
            bins[b] = str(round(range_min, 2)) + ' ' + str(round(range_min + interval, 2))
            range_min += interval
            b += 1
        while(range_min <= 26):
            interval += 0.2
            bins[b] = str(round(range_min, 2)) + ' ' + str(round(range_min + interval, 2))
            b += 1
            range_min += interval
        bins[b] = '26.0 1000.0'
        print('')
        print('Number of bins', len(bins))
        print('Actual bins:', bins)

        return bins

class DistDataset(Dataset):
    def __init__(self, pdb_id_list, features_path, distmap_path,
                 dim, pad_size, n_channels,
                 label_engineering = None):
        self.pdb_id_list = pdb_id_list
        self.features_path = features_path
        self.distmap_path = distmap_path
        self.dim = dim
        self.pad_size = pad_size
        self.n_channels = n_channels
        self.label_engineering = label_engineering

    def __len__(self):
        return len(self.pdb_id_list)

    def __getitem__(self, index):
        
        X, Y = get_input_output_dist(self.pdb_id_list[index: index+1],
                                     self.features_path,
                                     self.distmap_path, self.pad_size,
                                     self.dim, self.n_channels)
        X = np.moveaxis(X, -1, 0)
        Y = np.moveaxis(Y, -1, 0)
        if self.label_engineering is None:
            return X, Y
        if self.label_engineering == '100/d':
            return X, 100.0 / Y
        try:
            t = float(self.label_engineering)
            Y[Y > t] = t
        except ValueError:
            print('ERROR!! Unknown label_engineering parameter!!')
            return
        return X, Y


def get_loader(name, batch_size, shuffle, **kwargs):
    if name == 'contact':
        dataset = ContactDataset(**kwargs)
    elif name == 'binned':
        dataset = BinnedDistDataset(**kwargs)
    elif name == 'dist':
        dataset = DistDataset(**kwargs)

    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=0)
