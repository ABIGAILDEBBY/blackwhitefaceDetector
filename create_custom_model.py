from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define image dimensions (height and width) for data preprocessing
IMG_HEIGHT = 86
IMG_WIDTH = 86


def black_white_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_filters=[16, 32, 64], dense_units=512):
    model = Sequential()

    for num_filter in num_filters:
        model.add(Conv2D(num_filter, 3, padding='same', activation='relu',
                         input_shape=input_shape))
        model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))

    return model
