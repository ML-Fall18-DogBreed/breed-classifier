from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization

def create_cnn_model_trang():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(224, 224, 3)))
    model.add(Conv2D(filters=16, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=32, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=64, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=128, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=256, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())

    model.add(GlobalAveragePooling2D())
    #model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    #model.add(Dense(512))
    #model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(120))
    model.add(Activation('softmax'))

    #model.add(Dense(120, activation='softmax'))

    return model
