# Images dimensionality reduction

This work was inspired by : [https://github.com/ecm200/caltech_birds/blob/master/example_notebooks/002_Train_pytorch_resnet_Caltech_birds.ipynb](https://github.com/ecm200/caltech_birds/blob/master/example_notebooks/002_Train_pytorch_resnet_Caltech_birds.ipynb)

## Introduction

Images will be reduced from a **length \* width \* 3** into a **512** dimensionality space.

### Example with a colored 1920*1280 picture

1920 \* 1280 \* 3 = 7 372 800 dimensions

| | Input Dimensions | Output dimensions |
| - | - | - |
| Image of bird | 7 372 800 | 512 |

## Process

### Three steps

#### Step 1 : load images into train/test

Load images from the *images/* folder and randomly split *20%* of each classes into a test set.

#### Step 2 : Train a model

Train a resnet34 residual neural network to recognize each class.

#### Step 3 : Reduce dimension

For each images in *images/* stores the 512 features output in a *feature.csv* file.

## Loading images

Input images must be inserted in the images folder and follow this architecture :  

- images/
  - class1/
    - img1.jpg
    - img2.jpg
    - img3.jpg
  - class2/
    - img4.jpg
    - img5.jpg
    - img6.jpg

*Don't forget to remove the images/README.md file.*

## Run scripts

Either run scripts by hand or run ***main.py***.
