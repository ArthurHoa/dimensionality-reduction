"""
Load labelled images into a train/test dataset.

Author : Arthur Hoarau
Date : 2022/02/10
"""

import os
import pandas as pd
import shutil
import random

def __main__():

    # Origin folder containing images
    orig_images_folder = 'images'
    # Destination folder for the train/test dataset
    root_dir = 'data'

    image_fnames = list_all_images(orig_images_folder)

    os.makedirs(os.path.join(root_dir,'train'), exist_ok=True)
    os.makedirs(os.path.join(root_dir,'test'), exist_ok=True)

    # Copy images in the appropriate train\test directory

    for i_image, image_fname in enumerate(image_fnames['file path']):
        if image_fnames['is training image?'].iloc[i_image]:
            new_dir = os.path.join(root_dir,'train',image_fname.split('/')[0])
            os.makedirs(new_dir, exist_ok=True)
            shutil.copy(src=os.path.join(orig_images_folder,image_fname), dst=os.path.join(new_dir, image_fname.split('/')[1]))
            print(i_image, ':: Image is in training set. [', bool(image_fnames['is training image?'].iloc[i_image]),']')
            print('Image:: ', image_fname)
            print('Destination:: ', new_dir)
        else:
            new_dir = os.path.join(root_dir,'test',image_fname.split('/')[0])
            os.makedirs(new_dir, exist_ok=True)
            shutil.copy(src=os.path.join(orig_images_folder,image_fname), dst=os.path.join(new_dir, image_fname.split('/')[1]))
            print(i_image, ':: Image is in testing set. [', bool(image_fnames['is training image?'].iloc[i_image]),']')
            print('Source Image:: ', image_fname)
            print('Destination:: ', new_dir)

def list_all_images(dirPath):
    """
    List all images at the specified path and split into train/test sets.
    80 % of the dataset will be used as training data and
    20 % as testing data.
    Remember to use at least 5 images per class.

    Parameters
    -----
    dirPath : str
        Path where to find the images

    Returns
    -----
    self : DataFrame
        The dataframe containing train/test samples.
    """

    i = 1
    images = []

    # List files for each class
    for dir in os.listdir(dirPath):
        fileList = os.listdir(os.path.join(dirPath,dir))

        # Randomly select test samples
        random_index = random.sample(fileList, int(len(fileList) / 5))
        for file in fileList:
            isTraining = 0 if file in random_index else 1
            images.append([i, os.path.join(dir,file), isTraining])
            i += 1

    return pd.DataFrame(images, columns=['Img ID', 'file path', 'is training image?'])

# Execute main code
__main__()