from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

colors = np.array([[158/256, 1/256, 66/256, 1],
                  [213/256, 62/256, 79/256, 1],
                  [244/256, 109/256, 67/256, 1],
                  [253/256, 174/256, 97/256, 1],
                  [254/256, 224/256, 139/256, 1],
                  [256/256, 256/256, 256/256, 1],
                  [230/256, 245/256, 152/256, 1],
                  [171/256, 221/256, 164/256, 1],
                  [102/256, 194/256, 165/256, 1],
                  [50/256, 136/256, 189/256, 1],
                  [94/256, 79/256, 162/256, 1]])

cmap1 = LinearSegmentedColormap.from_list("mycmap", colors, N=256)


def plot_examples(colormaps):
    """
    Helper function to plot data with associated colormap.
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)
    n = len(colormaps)
    fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 3),
                            constrained_layout=True, squeeze=False)
    for [ax, cmap] in zip(axs.flat, colormaps):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()


plot_examples([cmap1])
