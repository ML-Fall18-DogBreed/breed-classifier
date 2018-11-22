"""
This file copies all of the images in Images/ to subdirectories for
training and validation data. The dataset provides lists of which
images should be used for which, but does not segment them, so here
we have to do it manually.
"""
from scipy.io import loadmat
from shutil import copyfile, copytree
from tqdm import tqdm
import os

dataset_path = os.path.join(os.path.dirname(__file__), '..', 'Images')
train_path = os.path.join(os.path.dirname(__file__), '..', 'data/train')
validation_path = os.path.join(os.path.dirname(__file__), '..', 'data/validation')

# Helper function to filter out all the files in the tree of images
def ig_f(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

# Copies the entire directory tree over to the data/ directory minus images
copytree(dataset_path, train_path, ignore=ig_f)
copytree(dataset_path, validation_path, ignore=ig_f)

# Copies the select images from the training list and the validation list to
# the correctly labeled directories within data/

validation_list = os.path.join(os.path.dirname(__file__), '..', 'lists/test_list.mat')
train_list = os.path.join(os.path.dirname(__file__), '..', 'lists/train_list.mat')

matrix = loadmat(validation_list)
print("Transferring validation data. Progress: ")
for file in tqdm(matrix['file_list'], unit='files'):
    copyfile(os.path.join(dataset_path, '%s' % file[0][0]),
             os.path.join(validation_path, '%s' % file[0][0]))

matrix = loadmat(train_list)
print("Transferring training data. Progress: ")
for file in tqdm(matrix['file_list'], unit='files'):
    copyfile(os.path.join(dataset_path, '%s' % file[0][0]),
             os.path.join(train_path, '%s' % file[0][0]))