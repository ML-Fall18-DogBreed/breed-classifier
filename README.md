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

## Running the code
Run `classifier.py` to train the network. This will save the model weights
to `out/model_weights.h5`. These can then be loaded in to test the 
model without having to train it again. An example of this is shown
in `predict.py`.

To run on the Ace cluster, change directory to `cnn_classifier` and
run `$ sbatch cnn-job.sh`. This will launch a job running the classifier script 
that should be significantly faster and more efficient than running on a local
machine.

# Neural Network Trial Results
For documentation purposes.

## Network structure
The network is built in the `create_model.py` file, where each Keras layer is added to the model one by one.
 * First, the convlutions (Conv2D) layers are added to pre-process the images and extract information about them. These layers are 
 listed in the table by how many convolutions are passed into the `Conv2D(filters, kernel_size=(3,3))` call. Each one is followed
 by a Batch Normalization, relu activation, and MaxPooling2D layer.
 * Then, the model is flattened from 2D data to 1D using a `GlobalAveragePooling2D()` layer.
 * Densely connected layers are added next. I have found that 512 hidden units per layer makes for better performance than larger values
 like 1024 and smaller values like 256, so all of the fully connected layers have 512 hidden units. Also, dropout was added to most of 
 these layers (which means that a certain percentage of the hidden units will be dropped from the network on every iteration in order to
 ensure that the model can generalize and nodes are not becoming specific to images). Each dense layer has a relu activation layer.
 * Finally, a densely connected layer of 120 units with softmax activation serves as the output.
 
All in all, this looks something like the code below:
```python
    model = Sequential()
    
    model.add(Conv2D(16, (3, 3), input_shape=(200, 200, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    ...
    model.add(GlobalAveragePooling2D()) 
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    ...
    model.add(Dense(120))
    model.add(Activation('softmax'))
```

## Results
The table below has the results for each of the trials so far based on the parameters used. 

| Batch size | Shear/ Zoom | Layer 1     | Layer 2     | Layer 3     | Layer 4      | Layer 5      | Layer 6                 | Layer 7                 | Layer 8                 | Layer 9             | Accuracy |
|------------|-------------|-------------|-------------|-------------|--------------|--------------|-------------------------|-------------------------|-------------------------|---------------------|----------|
| 64         | 0/0         | Conv2D - 16 | Conv2D - 32 | Conv2D - 64 | Conv2D - 128 | Dense - 512  | Dense - 120             |                         |                         |                     | 18%      |
| 64         | 0.2/0.2     | Conv2D - 16 | Conv2D - 32 | Conv2D - 64 | Conv2D - 128 | Dense - 512  | Dense - 512             | Dense - 120 Softmax     |                         |                     | 34%      |
| 64         | 0.2/0.2     | Conv2D - 16 | Conv2D - 32 | Conv2D - 64 | Conv2D - 128 | Conv2D - 256 | Dense - 512 50% dropout | Dense - 512 50% dropout | Dense - 512             | Dense - 120 Softmax | 25%      |
| 64         | 0.2/0.2     | Conv2D - 16 | Conv2D - 32 | Conv2D - 64 | Conv2D - 128 | Conv2D - 256 | Dense - 512 50% dropout | Dense - 512 50% dropout | Dense - 120 Softmax     |                     | 34%      |
| 32         | 0.2/0.2     | Conv2D - 16 | Conv2D - 32 | Conv2D - 64 | Conv2D - 128 | Conv2D - 256 | Dense - 512 50% dropout | Dense - 512 50% dropout | Dense - 120 Softmax     |                     | 12.5%    |
| 64         | 0.3/0.3     | Conv2D - 16 | Conv2D - 32 | Conv2D - 64 | Conv2D - 128 | Conv2D - 256 | Dense - 512 50% dropout | Dense - 512 50% dropout | Dense - 120 Softmax     |                     | 39%      |
| 128        | 0.3/0.3     | Conv2D - 16 | Conv2D - 32 | Conv2D - 64 | Conv2D - 128 | Conv2D - 256 | Dense - 512 50% dropout | Dense - 512 50% dropout | Dense - 120 Softmax     |                     | 30%      |
| 64         | 0.35/0.35   | Conv2D - 16 | Conv2D - 32 | Conv2D - 64 | Conv2D - 128 | Conv2D - 256 | Dense - 512 50% dropout | Dense - 512 50% dropout | Dense - 120 Softmax     |                     | 34%      |
| 128        | 0.4/0.4     | Conv2D - 16 | Conv2D - 32 | Conv2D - 64 | Conv2D - 128 | Conv2D - 256 | Dense - 512 50% dropout | Dense - 512 50% dropout | Dense - 120 Softmax     |                     | 26%      |
| 64         | 0.4/0.4     | Conv2D - 16 | Conv2D - 32 | Conv2D - 64 | Conv2D - 128 | Conv2D - 256 | Dense - 512 50% dropout | Dense - 512 50% dropout | Dense - 120 Softmax     |                     | 18%      |
| 64         | 0.3/0.3     | Conv2D - 16 | Conv2D - 32 | Conv2D - 64 | Conv2D - 128 | Conv2D - 256 | Conv2D - 512            | Dense - 512 50% dropout | Dense - 512 50% dropout | Dense - 120 Softmax | 25%      |
