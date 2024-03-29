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
# from Custom_CNN import cnn 

import os
from tqdm import tqdm


# Data struct
transforms = {
    'train' : transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = 'C:/Users/SJ/Documents/FER-on-Exynos-device/FER/dataset'

train_set = datasets.ImageFolder(data_dir + '/train',
                            transforms['train'])
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 8, shuffle=True)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available!')


# load model & set param
model = resnet50().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

loss_ = []
batch_num = len(train_loader)


# training
print('start training!!!')

for epoch in range(20):
    running_loss = 0.0

    for i, data in enumerate(tqdm(train_loader)):
        input, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(input)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    loss_.append(running_loss / batch_num)
    print('[%d] loss: %.3f' %(epoch + 1, running_loss / len(train_loader)))


weights_path = './FER/resnet50.pt'
# Model_PATH = './FER/cnn.pt'

torch.save(model.state_dict(), weights_path)