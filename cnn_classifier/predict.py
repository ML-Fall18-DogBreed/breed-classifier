from create_model import create_cnn_model
import numpy as np
import cv2
import pickle

model = create_cnn_model()

model.load_weights('out/model_weights.h5')

img1 = cv2.imread('../Images/n02086910-papillon/n02086910_26.jpg')
img1 = cv2.resize(img1, (150,200))

img1 = np.array(img1).reshape((1, 150, 200, 3))
prediction = model.predict_classes(img1)

with open('out/labels.pkl', 'rb') as f:
    label_map = pickle.load(f)

print(label_map[prediction[0]])