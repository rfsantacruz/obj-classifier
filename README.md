# obj-classifier
This project was developed for the CNN workshop of the Robotic Vision Summer School 2017 (see http://roboticvision.org/events/rvss-summer-school/). This workshop taught how to train a CNN classifier for user-defined classes using keras and tensorflow (See the presentation).

## Pre-requistes
1. Python 3
2. Tensorflow 0.12
3. keras 1.2.2

## Data directory
Create a data directory with train and validation folder. In each folder, add subfolders for each class and name it with the class label. Then, pass the data directory to the train and validation script. Note that you can always train with subsets of classes by passing -classes option.

## Train and Evaluation
Script to run train and evaluation. For instructions,

$ python ./src/objcls_train_val.py --help

## Prediction
Script to peform prediction on test data. For instructions,

$ python ./src/objcls_predict.py --help
