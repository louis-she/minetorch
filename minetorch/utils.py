import torch
import numpy as np


def seed_everything(seed):
    torch.manual_seed(seed)
    # may lead to bad performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
