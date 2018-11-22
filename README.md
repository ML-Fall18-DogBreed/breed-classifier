# Dog Breed Classification Algorithms
Final project for CS 539, Machine Learning

## Setting up

1. (Recommended) Create a virtual environment for this project.
[Virtual environments are documented extensively here](https://virtualenv.pypa.io/en/latest/userguide/). Do all the 
 following steps with that virtual environment activated.

2. Install dependencies with `pip install -r requirements.txt`.

3. For Linux and Mac, download the Stanford Dogs dataset of images with
`./image-database-download.sh`. 

    - You may need to set it to executable first with 
`chmod +x image-database-download.sh`.  
    - This command will automatically call `data_segmenter.py`, 
    which creates a `data` directory where the training and validation 
    sets of images are stored.

4. For Windows, download the 
data from [http://vision.stanford.edu/aditya86/StanfordDogs/](http://vision.stanford.edu/aditya86/StanfordDogs/) and use
a program like 7-Zip to extract the files. 
     - Then, run the data segmenter by hand with `python data_segmenter.py` 
     to create the `data` directory.

5. That's it! You should now have the following directories.
```bash
| breed_classifier
|--- cnn_classifier (algorithms and helpers)
|--- data           (segmented training/validation images)
|--- Images         (unsegmented images, in case we need them)
|--- lists          (lists of the training/validation files)
```

We can keep the Images directory for now, even though it is not
used by the algorithm. We may want to segment them differently later.

> Note that any and all image files ending with .jpg are ignored
in this repository, so nothing in Images/ or data/ will ever be
pushed. (This is probably faster than having them all in the repo anyways.)