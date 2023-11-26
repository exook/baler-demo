import numpy as np
import matplotlib.pyplot as plt

def plot2D(original, reconstructed):

    diff = original - reconstructed

    max_value = np.amax([np.amax(original), np.amax(reconstructed)])
    min_value = np.amin([np.amin(original), np.amin(reconstructed)])

    fig, axs = plt.subplots(
        1, 3, figsize=(29.7 * (1 / 2.54), 10 * (1 / 2.54)), sharey=True
    )
    axs[0].set_title("Original", fontsize=11)
    im1 = axs[0].imshow(original, vmax=max_value, vmin=min_value)
    axs[1].set_title("Reconstructed", fontsize=11)
    im2 = axs[1].imshow(reconstructed, vmax=max_value, vmin=min_value)
    axs[2].set_title("Difference", fontsize=11)
    im3 = axs[2].imshow(diff, vmax=max_value, vmin=min_value)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.815, 0.2, 0.02, 0.59])
    cb2 = fig.colorbar(im3, cax=cbar_ax, aspect=10)
    plt.show()
