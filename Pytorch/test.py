import enum
import os
import torch
from torch import tensor
from torch import nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader , random_split
from torchvision import transforms, utils
from models import gammaModel1
from cDataLoaders import*

x   = tensor([0.3405, 0.3765, 0.3286, 0.9052, 0.2257])
y   = tensor([1.8090, 1.4250, 1.3270, 0.5340, 1.7840])
res = tensor(1.3638)

z = (abs(x-y) <= 0.5).sum().item()
print(z)