from create_model import create_cnn_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import pickle
import datetime # For naming files


model = create_cnn_model()

model.compile(loss='categorical_crossentropy',
              optimizer='RMSprop',
	      metrics=['accuracy'])

batch_size = 64

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        # shear_range=0.2,
        # zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        '../data/train',  # this is the target directory
        target_size=(200, 200),  # all images will be resized to 150x150
        batch_size=batch_size)  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        '../data/validation',
        target_size=(200, 200),
        batch_size=batch_size)

labels = (train_generator.class_indices)
label_map = dict((v,k) for k,v in labels.items())
with open('out/labels.pkl', 'wb') as f:
    pickle.dump(label_map, f, pickle.HIGHEST_PROTOCOL)


model.fit_generator(
        train_generator,
        steps_per_epoch=12000 // batch_size,
        epochs=25,
        validation_data=validation_generator,
        validation_steps=8000 // batch_size,
	verbose=2)

model.save_weights('out/model_weights%s.h5' % datetime.datetime.today().strftime('%b-%d-%H%M'))
model.save('out/model.h5')
img1 = cv2.imread('../Images/n02086910-papillon/n02086910_54.jpg')
img1 = cv2.resize(img1, (200,200))

img1 = np.array(img1).reshape((1, 200, 200, 3))
prediction = model.predict_classes(img1)

with open('out/labels.pkl', 'rb') as f:
    label_map = pickle.load(f)

print(label_map[prediction[0]])

score = model.evaluate_generator(validation_generator, steps=1, verbose=1)
print(score)
print(model.metrics_names)
