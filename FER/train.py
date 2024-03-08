import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from models import resnet50

import os


transforms = {
    'train' : transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test' : transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = './dataset'

train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                            transforms['train'])
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 8, shuffle=True)

test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'),
                                transforms['test'])
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 8, shuffle=False)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available!')

