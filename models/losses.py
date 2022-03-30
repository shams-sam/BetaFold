import numpy as np
import torch
import torch.nn.functional as F

epsilon = torch.finfo(torch.float).eps

def _logcosh(x):
    return x + F.softplus(-2. * x) - np.log(2.)

def inv_log_cosh(y_true, y_pred):
    y_true = y_true.type(y_pred.dtype)

    return torch.mean(_logcosh(
            100.0 / (y_pred + epsilon) - 100.0 / (y_true + epsilon)
        ))

def log_cosh(y_true, y_pred):
    y_true = y_true.type(y_pred.dtype)

    return torch.mean(_logcosh(y_pred - y_true))
