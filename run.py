import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


def plot_2D(original_path, decompressed_path, plot_path):
    data = np.load(original_path)["data"]
    data_decompressed = np.load(decompressed_path)["data"]
    print(data_decompressed.shape)

    data_decompressed = data_decompressed.reshape(
        data.shape[0], data.shape[1], data.shape[2]
    )

    if data.shape[0] > 1:
        num_tiles = data.shape[0]
    else:
        num_tiles = 1

    print("=== Plotting ===")
    for ind in trange(num_tiles):
        tile_data = data[ind]
        tile_data_decompressed = data_decompressed[ind]

        diff = tile_data - tile_data_decompressed

        max_value = np.amax([np.amax(tile_data), np.amax(tile_data_decompressed)])
        min_value = np.amin([np.amin(tile_data), np.amin(tile_data_decompressed)])

        fig, axs = plt.subplots(
            1, 3, figsize=(29.7 * (1 / 2.54), 10 * (1 / 2.54)), sharey=True
        )
        axs[0].set_title("Original", fontsize=11)
        im1 = axs[0].imshow(tile_data, vmax=max_value, vmin=min_value)
        axs[1].set_title("Reconstructed", fontsize=11)
        im2 = axs[1].imshow(tile_data_decompressed, vmax=max_value, vmin=min_value)
        axs[2].set_title("Difference", fontsize=11)
        im3 = axs[2].imshow(diff, vmax=max_value, vmin=min_value)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.815, 0.2, 0.02, 0.59])
        cb2 = fig.colorbar(im3, cax=cbar_ax, aspect=10)

        fig.savefig(plot_path, bbox_inches="tight")
        break


def main():
    plot_2D(
        "./input/exafel1.npz",
        "./output/decompressed_output/decompressed.npz",
        "./output/plot_output/test.png",
    )


if __name__ == "__main__":
    main()
