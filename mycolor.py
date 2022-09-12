# -------------------------------------------------------------------------------
# Custom discrete colormap
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import cmcrameri.cm as cmc
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator

def custom_div_cmap(numcolors, colormap):
    """ Create a custom diverging colormap with three colors

    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """

    from matplotlib.colors import LinearSegmentedColormap

    colors_red = colormap(np.linspace(0, 0.5, 20))
    colors_blue = colormap(np.linspace(0.5, 1, 20))
    colors_white = np.array([1, 1, 1, 1])
    colors = np.vstack((colors_red, colors_white, colors_blue))

    cmap = LinearSegmentedColormap.from_list(name=colormap, colors=colors, N=numcolors)

    return cmap

def lapaz_cmap(numcolors, colormap):
    """ Create a custom diverging colormap with three colors

    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """

    from matplotlib.colors import LinearSegmentedColormap

    colors_lapaz = colormap(np.linspace(0, 1, 200))
    colors_white = np.array([1, 1, 1, 1])
    colors = np.vstack((colors_white, colors_lapaz))

    cmap = LinearSegmentedColormap.from_list(name=colormap, colors=colors, N=numcolors)

    return cmap



