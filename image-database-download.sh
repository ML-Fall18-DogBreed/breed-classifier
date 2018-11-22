#!/bin/bash
# Script for downloading the image files and segmenting them
# into the training and validation sets according to the lists
# from the Stanford Dogs Dataset

wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar 
tar -xvf images.tar
rm -f images.tar
python cnn_classifier/data_segmenter.py
