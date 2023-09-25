# This function will plot images in the form of a grid with 1 row and 5
# columns where images are placed in each column.
import math
import matplotlib
import matplotlib.pyplot as plt


def plot_images(images_arr):
    num_cols = 5
    num_rows = math.ceil(len(images_arr) / num_cols)
    fig, axes = matplotlib.pyplot.subplots(num_rows, num_cols, figsize=(10, 10))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
