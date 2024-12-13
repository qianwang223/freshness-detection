# Fruit and Vegetable Feshness Detection using CNN

## Description

This project aims to experiment with multiple convolutional neural network models for freshness detection in fruits and vegetables, and compare their performances.

## Dataset

https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification
Please download the dataset from the link above, unzip it, and place the 'dataset' folder in the root directory of this project.

## AlexNet

Run "python model_alexnet_fresh.py" and "python model_alexnet_type.py" to train models for each task.
Run "python alexnet_inference.py" to test the performance of the model and some samples will be saved in "alexnet_testing" folder.

## ConvNeXt

Run "python model_convnext.py" to train the model.
Run "python convnext_inference.py" to test the performance of the model and some samples will be saved in "convnext_testing" folder.
