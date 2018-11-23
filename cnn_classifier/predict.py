from cnn_classifier.create_model import create_cnn_model
import numpy as np
import cv2
import pickle

model = create_cnn_model()

model.load_weights('first_try.h5')

img1 = cv2.imread('../Images/n02085620-Chihuahua/n02085620_7.jpg')
img1 = cv2.resize(img1, (150,150))

img1 = np.array(img1).reshape((1, 150, 150, 3))
prediction = model.predict_classes(img1)

with open('labels.pkl', 'rb') as f:
    label_map = pickle.load(f)

print(label_map[prediction])
