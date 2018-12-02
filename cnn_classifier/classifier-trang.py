from create_model_trang import create_cnn_model_trang
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint  
import pickle

model = create_cnn_model_trang()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 32

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        '../data/train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size)  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        '../data/validation',
        target_size=(150, 150),
        batch_size=batch_size)

labels = (train_generator.class_indices)
label_map = dict((v,k) for k,v in labels.items())
with open('out/labels.pkl', 'wb') as f:
    pickle.dump(label_map, f, pickle.HIGHEST_PROTOCOL)
    
checkpointer = ModelCheckpoint(filepath='out_trang/weights.bestaugmented.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit_generator(
        train_generator,
        steps_per_epoch=2000// batch_size,
        epochs=150,
        validation_data=validation_generator,
        validation_steps=800 // batch_size,
        callbacks=[checkpointer])

#model.save_weights('out_trang/model_weights.h5')  # always save your weights after training or during training
