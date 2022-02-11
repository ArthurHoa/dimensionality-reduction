"""
Train a Residual neural network with a train/test dataset.

Author : Arthur Hoarau
Date : 2022/02/11
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models

from cub_tools.train import train_model
from cub_tools.transforms import makeDefaultTransforms
import os

# Model
model_name = 'resnet34'
model_func = models.resnet34
working_dir = os.path.join('models/classification', model_name)
batch_size = 16
num_workers = 4
num_epochs = 25

# Dataset location
root_dir = 'data'

os.makedirs(working_dir, exist_ok=True)

# Setup data loaders with augmentation transforms
data_transforms = makeDefaultTransforms()
image_datasets = {x: datasets.ImageFolder(os.path.join(root_dir, x), data_transforms[x])
                  for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)
              for x in ['train', 'test']}


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

print('Number of data')
print('========================================')
for dataset in dataset_sizes.keys():
    print(dataset,' size:: ', dataset_sizes[dataset],' images')

print('')
print('Number of classes:: ', len(class_names))
print('========================================')
for i_class, class_name in enumerate(class_names):
    print(i_class,':: ',class_name)



# Setup the device to run the computations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device::', device)


# Setup the model and optimiser
model_ft = models.resnet34(pretrained=True)

num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, len(class_names))
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model=model_ft, criterion=criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler, 
                       device=device, dataloaders=dataloaders, dataset_sizes=dataset_sizes, num_epochs=num_epochs,
                       working_dir=working_dir)

# Save model
torch.save(model_ft, "models/model.save")

