"""
Execute the load/train/reduce pipeline.

Author : Arthur Hoarau
Date : 2022/02/11
"""

import os

# Each script can be run separately for verbose

# Load images from ./images into ./data/train and ./data/test
os.system("python3 load_images.py")

# Train a residual neural network with the previous generated dataset
os.system("python3 train_model.py")

# Extract reduced features from the trained model into features.csv
os.system("python3 reduce_data.py")

