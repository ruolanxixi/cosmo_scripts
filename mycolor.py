# -------------------------------------------------------------------------------
# Custom discrete colormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
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


def cbr_drywet(numcolors):
    colvals = [[84, 48, 5, 255],
               [140, 81, 10, 255],
               [191, 129, 45, 255],
               [223, 194, 125, 255],
               [246, 232, 195, 255],
               [245, 245, 245, 255],
               [199, 234, 229, 255],
               [128, 205, 193, 255],
               [53, 151, 143, 255],
               [1, 102, 95],
               [0, 60, 48]]

    rgb = []
    for i in range(len(colvals)):
        z = [x / 255 for x in colvals[i]]
        rgb.append(z)

    cmap = LinearSegmentedColormap.from_list('', rgb, numcolors)

    return cmap


def cbr_wet(numcolors):
    colvals = [[255, 255, 255, 255],
               [247, 252, 240, 255],
               [224, 243, 219, 255],
               [204, 235, 197, 255],
               [168, 221, 181, 255],
               [123, 204, 196, 255],
               [78, 179, 211, 255],
               [43, 140, 190, 255],
               [8, 104, 172, 255],
               [8, 64, 129, 255],
               [0, 32, 62, 255]]

    rgb = []
    for i in range(len(colvals)):
        z = [x / 255 for x in colvals[i]]
        rgb.append(z)

    cmap = LinearSegmentedColormap.from_list('', rgb, numcolors)

    return cmap


def prcp(numcolors):
    colvals = [[255, 255, 255, 255],
               [237, 250, 194, 255],
               [205, 255, 205, 255],
               [153, 240, 178, 255],
               [83, 189, 159, 255],
               [50, 166, 150, 255],
               [50, 150, 180, 255],
               [5, 112, 176, 255],
               [5, 80, 140, 255],
               [10, 31, 150, 255],
               [44, 2, 70, 255],
               [106, 44, 90, 255]]
    rgb = []
    for i in range(len(colvals)):
        z = [x / 255 for x in colvals[i]]
        rgb.append(z)

    cmap = LinearSegmentedColormap.from_list('', rgb, numcolors)

    return cmap


def cbr_coldhot(numcolors):
    colvals = [[5, 48, 97, 255],
               [33, 102, 172, 255],
               [67, 147, 195, 255],
               [146, 197, 222, 255],
               [209, 229, 240, 255],
               [247, 247, 247, 255],
               [254, 219, 199, 255],
               [244, 165, 130, 255],
               [214, 96, 77, 255],
               [178, 24, 43, 255],
               [103, 0, 31, 255]]

    rgb = []
    for i in range(len(colvals)):
        z = [x / 255 for x in colvals[i]]
        rgb.append(z)

    cmap = LinearSegmentedColormap.from_list('', rgb, numcolors)

    return cmap


def prdiff(numcolors):
    colvals = [[182, 106, 40, 255],
               [205, 133, 63, 255],
               [225, 165, 100, 255],
               [245, 205, 132, 255],
               [245, 224, 158, 255],
               [255, 245, 186, 255],
               [255, 255, 255, 255],
               [205, 255, 205, 255],
               [153, 240, 178, 255],
               [83, 189, 159, 255],
               [110, 170, 200, 255],
               [5, 112, 176, 255],
               [2, 56, 88, 255]]

    rgb = []
    for i in range(len(colvals)):
        z = [x / 255 for x in colvals[i]]
        rgb.append(z)

    cmap = LinearSegmentedColormap.from_list('', rgb, numcolors)

    return cmap


def hotcold(numcolors):
    colvals = [[24, 24, 112, 255],
               [16, 78, 139, 255],
               [23, 116, 205, 255],
               [72, 118, 255, 255],
               [91, 172, 237, 255],
               [173, 215, 230, 255],
               [209, 237, 237, 255],
               [229, 239, 249, 255],
               [242, 255, 255, 255],
               [255, 255, 255, 255],
               [253, 245, 230, 255],
               [255, 228, 180, 255],
               [243, 164, 96, 255],
               [237, 118, 0, 255],
               [205, 102, 29, 255],
               [224, 49, 15, 255],
               [237, 0, 0, 255],
               [205, 0, 0, 255],
               [139, 0, 0, 255]]

    rgb = []
    for i in range(len(colvals)):
        z = [x / 255 for x in colvals[i]]
        rgb.append(z)

    cmap = LinearSegmentedColormap.from_list('', rgb, numcolors)

    return cmap


def custom_div_cmap(numcolors, colormap):
    """ Create a custom diverging colormap with three colors

    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """

    from matplotlib.colors import LinearSegmentedColormap

    colors_red = colormap(np.linspace(0, 0.49, 20))
    colors_blue = colormap(np.linspace(0.51, 1, 20))
    colors_white = np.array([1, 1, 1, 1])
    colors = np.vstack((colors_red, colors_white, colors_blue))

    cmap = LinearSegmentedColormap.from_list(name=colormap, colors=colors, N=numcolors)

    return cmap


def custom_seq_cmap(numcolors, colormap, white, negative):
    """ Create a custom diverging colormap with three colors

    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """

    from matplotlib.colors import LinearSegmentedColormap

    colors_blue = colormap(np.linspace(0.5, 1, 20))
    colors_white = np.array([1, 1, 1, 1])
    colors = np.vstack((colors_white, colors_blue))

    if white == 1:
        colors = np.vstack((colors_white, colors_blue))
    else:
        colors = colors_blue

    if negative == 1:
        colors = colors[::-1]

    cmap = LinearSegmentedColormap.from_list(name=colormap, colors=colors, N=numcolors)

    return cmap


def custom_white_cmap(numcolors, colormap):
    """ Create a custom diverging colormap with three colors

    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """

    from matplotlib.colors import LinearSegmentedColormap

    colors_blue = colormap(np.linspace(0, 1, 100))
    colors_white = np.array([1, 1, 1, 1])
    colors = np.vstack((colors_white, colors_blue))

    cmap = LinearSegmentedColormap.from_list(name=colormap, colors=colors, N=numcolors)

    return cmap


def custom_white_cmap_(numcolors, colormap):
    """ Create a custom diverging colormap with three colors

    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """

    from matplotlib.colors import LinearSegmentedColormap

    colors_blue = colormap(np.linspace(0, 1, 20))
    colors_white = np.array([1, 1, 1, 1])
    colors = np.vstack((colors_blue, colors_white))

    cmap = LinearSegmentedColormap.from_list(name=colormap, colors=colors, N=numcolors)

    return cmap


def custom_seq_cmap_(numcolors, colormap):
    """ Create a custom diverging colormap with three colors

    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """

    from matplotlib.colors import LinearSegmentedColormap

    colors_blue = colormap(np.linspace(0.5, 1, 20))
    colors_white = np.array([1, 1, 1, 1])
    colors = np.vstack((colors_blue, colors_white))

    cmap = LinearSegmentedColormap.from_list(name=colormap, colors=colors, N=numcolors)

    return cmap


def drywet(numcolors, colormap):
    """ Create a custom diverging colormap with three colors

    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """

    from matplotlib.colors import LinearSegmentedColormap

    colors_blue = colormap(np.linspace(0.5, 1, 5))
    colors_white = np.array([1, 1, 1, 1])
    colors_brown = [[84, 48, 5, 255],
                    [140, 81, 10, 255],
                    [191, 129, 45, 255],
                    [223, 194, 125, 255],
                    [246, 232, 195, 255]]
    rgb = []
    for i in range(len(colors_brown)):
        z = [x / 255 for x in colors_brown[i]]
        rgb.append(z)
    colors = np.vstack((rgb, colors_white, colors_blue))

    cmap = LinearSegmentedColormap.from_list(name=colormap, colors=colors, N=numcolors)

    return cmap

def tmpmap(numcolors, colormap, nnum, pnum):
    """ Create a custom diverging colormap with three colors

    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """

    from matplotlib.colors import LinearSegmentedColormap

    colors_red = colormap(np.linspace(0.4, 0.5, nnum))
    colors_blue = colormap(np.linspace(0.5, 1, pnum))
    colors_white = np.array([1, 1, 1, 1])
    colors = np.vstack((colors_red, colors_blue))

    cmap = LinearSegmentedColormap.from_list(name=colormap, colors=colors, N=numcolors)

    return cmap

def tmpdiff(numcolors, colormap, nnum, pnum):
    """ Create a custom diverging colormap with three colors

    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """

    from matplotlib.colors import LinearSegmentedColormap

    colors_red = colormap(np.linspace(0.25, 0.45, nnum))
    colors_blue = colormap(np.linspace(0.5, 1, pnum))
    colors_white = np.array([1, 1, 1, 1])
    colors = np.vstack((colors_red, colors_white, colors_blue))

    cmap = LinearSegmentedColormap.from_list(name=colormap, colors=colors, N=numcolors)

    return cmap

def wind(numcolors, colormap):
    """ Create a custom diverging colormap with three colors

    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """

    from matplotlib.colors import LinearSegmentedColormap

    colors = colormap(np.linspace(0, 0.6, 20))

    cmap = LinearSegmentedColormap.from_list(name=colormap, colors=colors, N=numcolors)

    return cmap


def conv(numcolors):
    colvals = [[263, 77, 50, 255],
               [253, 56, 59, 255],
               [255, 33, 75, 255],
               [209, 73, 76, 255],
               [205, 52, 91, 255],
               [195, 16, 96, 255],
               [255, 255, 255, 255],
               [54, 34, 99, 255],
               [40, 78, 99, 255],
               [22, 97, 93, 255],
               [231, 22, 0, 255],
               [356, 98, 65, 255],
               [0, 0, 0, 255]]

    rgb = []
    for i in range(len(colvals)):
        z = [x / 255 for x in colvals[i]]
        rgb.append(z)

    cmap = LinearSegmentedColormap.from_list('', rgb, numcolors)

    return cmap


def prcp(numcolors):
    colvals = [[254, 217, 118, 255], # [254, 178, 76, 255],
               [255, 237, 160, 255],
               [237, 250, 194, 255],
               [205, 255, 205, 255],
               [153, 240, 178, 255],
               [83, 189, 159, 255],
               [50, 166, 150, 255],
               [50, 150, 180, 255],
               [5, 112, 176, 255],
               [5, 80, 140, 255],
               [10, 31, 150, 255],
               [44, 2, 70, 255],
               [106, 44, 90, 255]]
               # [168, 65, 91, 255]]
    rgb = []
    for i in range(len(colvals)):
        z = [x / 255 for x in colvals[i]]
        rgb.append(z)

    cmap = LinearSegmentedColormap.from_list('', rgb, numcolors)

    return cmap


def cape(numcolors):
    """ Create a custom diverging colormap with three colors

    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """

    from matplotlib.colors import LinearSegmentedColormap

    # colors_red = mpl.colormaps['hot_r'](np.linspace(0, 1, 20))
    colors_red = mpl.colormaps['plasma_r'](np.linspace(0, 1, 20))
    colors_blue = cmc.vik(np.linspace(0, 0.5, 20))
    colors_white = np.array([1, 1, 1, 1])
    colors = np.vstack((colors_blue, colors_white, colors_red))

    cmap = LinearSegmentedColormap.from_list('', colors=colors, N=numcolors)

    return cmap

