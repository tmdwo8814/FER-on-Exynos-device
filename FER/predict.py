import torch
from torchvision import datasets, transforms

from models import resnet50

from tqdm import tqdm

# Data handling
transforms = {
  'test' : transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = 'C:/Users/SJ/Documents/FER-on-Exynos-device/FER/dataset'

test_set = datasets.ImageFolder(data_dir + '/test',
                                transforms['test'])
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 8, shuffle=False)

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available!')

# load model & weights
weights_path = './FER/resnet50.pt'
model = resnet50(pretrained=False).to(device)
model.load_state_dict(torch.load(weights_path))

correct = 0
total = 0

print('start predict!!!')

with torch.no_grad(): 
  for data in tqdm(test_loader):
    images, labels = data[0].to(device), data[1].to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1) 
    total += labels.size(0) 
    correct += (predicted == labels).sum().item() 

print(f'accuracy of 10000 test images: {100*correct/total}%')