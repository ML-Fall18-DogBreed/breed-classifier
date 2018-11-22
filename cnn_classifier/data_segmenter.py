from scipy.io import loadmat
from shutil import copyfile, copytree
import os

inputpath = '../Images'
outputpath1 = '../data/train'
outputpath2 = '../data/validation'
def ig_f(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

copytree(inputpath, outputpath1, ignore=ig_f)
copytree(inputpath, outputpath2, ignore=ig_f)

# Add simple status indicator
x = loadmat('../lists/test_list.mat')
for y in x['file_list']:
    copyfile('../Images/%s' % y[0][0], '../data/validation/%s' % y[0][0])

x = loadmat('../lists/train_list.mat')
for y in x['file_list']:
    copyfile('../Images/%s' % y[0][0], '../data/train/%s' % y[0][0])
