"""
Extract images reduced features from a trained model.

Author : Arthur Hoarau
Date : 2022/02/11
"""

from __future__ import print_function, division

import torch
import os
import csv

from cub_tools.transforms import makeDefaultTransforms
from cub_tools.data import ImageFolderWithPaths
from cub_tools.layer_feature_maps import extract_feature_maps

# Directory paths
model_ = 'resnet34'
model_root_dir = "models"

# Max number of images to reduce
maxIm = 1000

# Paths setup
data_dir = "images"
output_dir = os.path.join(model_root_dir,'classification/{}'.format(model_))
model_history = os.path.join(output_dir,'model_history.pkl')
model_file = os.path.join(output_dir, 'caltech_birds_{}_full.pth'.format(model_))


# Get data transforms
data_transforms = makeDefaultTransforms(img_crop_size=224, img_resize=256)

# Setup data loaders with augmentation transforms
image_datasets = ImageFolderWithPaths(data_dir, data_transforms['test'])
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=1, shuffle=True, num_workers=4)

dataset_sizes = len(image_datasets)
class_names = image_datasets.classes


# Setup the device to run the computations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device::', device)

# Load the model saved during previous step
model_ = torch.load(os.path.join(model_root_dir,"model.save"))

fc_feature_extractions = []
def hook(module, input, output):
    fc_feature_extractions.append(output)

model_.avgpool.register_forward_hook(hook)

feature_extractions_dict = extract_feature_maps(model=model_, dataloader=dataloaders, fc_feature_extractions=fc_feature_extractions,device=device, batch_limit=maxIm)
print('\n')

# Precision of the model
prec = 0

# Final predictions
final = []
for i in range(len(feature_extractions_dict['feature extractions'])):
    buffer = []
    buffer.append(feature_extractions_dict['image paths'][i])
    buffer.append(feature_extractions_dict['labels truth'][i])
    buffer.append(feature_extractions_dict['labels pred'][i])

    for j in range(len(feature_extractions_dict['feature extractions'][i].flatten())):
        buffer.append(feature_extractions_dict['feature extractions'][i].flatten()[j])

    final.append(buffer)

    if(feature_extractions_dict['labels truth'][i] == feature_extractions_dict['labels pred'][i]):
        prec += 1
    else:
        print("Error of prediction on :", feature_extractions_dict['image paths'][i], "  predicted label :",
            feature_extractions_dict['labels pred'][i], "  true label :", feature_extractions_dict['labels truth'][i])

# Save data with new dimensions as csv
fields = ['File', 'Real', 'Predicted'] 
with open('features.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(final)
    f.close()

# Print accuracy
print("\nAccuracy :", prec / len(feature_extractions_dict['feature extractions']))
