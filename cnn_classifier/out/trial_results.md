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
| 64         | 0.3/0.3     | Conv2D - 16 | Conv2D - 32 | Conv2D - 64 | Conv2D - 128 | Conv2D - 256 | Conv2D - 512            | Dense - 512 50% dropout | Dense - 512 50% dropout | Dense - 120 Softmax | TBD      |
