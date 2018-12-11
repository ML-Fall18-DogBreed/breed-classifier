
import numpy as np
import cv2
import pickle
from keras.models import load_model
import matplotlib.pyplot as plt
import re
history = load_model('out/final_model.h5')

# model.load_weights('out/model_weightsNov-28-1935.h5')
#
# img1 = cv2.imread('../Images/n02086910-papillon/n02086910_399.jpg')
# img1 = cv2.resize(img1, (200,200))
#
# img1 = np.array(img1).reshape((1, 200, 200, 3))
#
# prediction = model.predict_classes(img1)
#
# with open('out/loss.pkl', 'r') as f:
#     loss = pickle.load(f)
#
# print(label_map[prediction[0]])

# Get training and test loss histories
# training_loss = loss.history['loss']
# test_loss = loss.history['val_loss']

with open('out/final_model_output.out', 'r') as f:
    lines = f.readlines()

output = "\n".join(lines)

training_loss = []
test_loss = []
for loss, val_loss in re.findall('loss: (\d\.\d{4}) .* val_loss: (\d\.\d{4})', output):
    training_loss.append(float(loss))
    test_loss.append(float(val_loss))

print(training_loss)

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();