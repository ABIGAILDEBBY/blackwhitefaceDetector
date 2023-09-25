import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

# Define image dimensions (height and width) for data preprocessing
IMG_HEIGHT = 86
IMG_WIDTH = 86


def black_white_model(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_filters=[16, 32, 64], dense_units=512
):
    model = Sequential()

    for num_filter in num_filters:
        model.add(
            Conv2D(
                num_filter,
                3,
                padding="same",
                activation="relu",
                input_shape=input_shape,
            )
        )
        model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(dense_units, activation="relu"))
    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model


def train_model(model, epochs, train_data_gen, batch_size, val_data_gen):
    PATH = os.getcwd()
    categories = ["white_faces", "black_faces"]
    subsets = ["train", "val", "test"]
    output_folder = f"{PATH}/../datasets_directory/splitted_dataset"
    data_counts = {"train": {}, "val": {}, "test": {}}

    for subset in subsets:
        subset_dir = os.path.join(output_folder, subset)
        for category in categories:
            category_dir = os.path.join(subset_dir, category)
            num_images = len(os.listdir(category_dir))
            data_counts[subset][category] = num_images

        # Calculate totals
        total_train = sum(data_counts["train"].values())
        total_val = sum(data_counts["val"].values())

    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=total_val // batch_size,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{PATH}/../Epochs_steps/blackWhite-{epochs:02d}.h5", verbose=1
            )
        ],
    )
    model.save(
        f"{PATH}/Models/blackVswhite_" + str(epochs) + "_" + str(batch_size) + ".h5"
    )
    return model, history


def plot_training_history(history, save_path=None):
    mpl.rcParams["lines.linewidth"] = 2

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", marker="o")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", marker="o")
    plt.legend(loc="best")
    plt.grid(True)
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", marker="o")
    plt.plot(epochs_range, val_loss, label="Validation Loss", marker="o")
    plt.legend(loc="best")
    plt.grid(True)
    plt.title("Training and Validation Loss")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
