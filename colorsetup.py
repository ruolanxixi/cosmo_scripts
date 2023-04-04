import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from mycolor import drywet, custom_div_cmap
import matplotlib
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator

# 'hurs', 'ps', 'siconc', 'tas', 'ts'
def colorsetup(var):
    if var == 'tas':
        cmap1 = cmc.roma_r
        levels1 = np.linspace(-40, 40, 21, endpoint=True)
        norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

        cmap2 = custom_div_cmap(23, cmc.vik)
        levels2 = np.linspace(-10, 10, 20, endpoint=True)
        norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)

        cmaps = [cmap1, cmap1, cmap2]
        norms = [norm1, norm1, norm2]

        cbarlabel = '$^{o}$C'

        extends = ['both', 'both']

        ticks1 = np.linspace(-40, 40, 5, endpoint=True)
        ticks2 = np.linspace(-10, 10, 5, endpoint=True)
        tickss = [ticks1, ticks2]

    elif var == 'hurs':
        cmap1 = cmc.roma
        levels1 = np.linspace(0, 100, 26, endpoint=True)
        norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

        cmap2 = drywet(25, cmc.vik_r)
        levels2 = np.linspace(-40, 40, 20, endpoint=True)
        norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)

        cmaps = [cmap1, cmap1, cmap2]
        norms = [norm1, norm1, norm2]

        cbarlabel = '%'

        extends = ['neither', 'both']

        ticks1 = np.linspace(0, 100, 6, endpoint=True)
        ticks2 = np.linspace(-40, 40, 5, endpoint=True)
        tickss = [ticks1, ticks2]

    elif var == 'ps':
        cmap1 = cmc.roma_r
        levels1 = np.linspace(600, 1100, 31, endpoint=True)
        norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

        cmap2 = custom_div_cmap(23, cmc.vik)
        levels2 = np.linspace(-30, 30, 20, endpoint=True)
        norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)

        cmaps = [cmap1, cmap1, cmap2]
        norms = [norm1, norm1, norm2]

        cbarlabel = 'hPa'

        extends = ['both', 'both']

        ticks1 = np.linspace(600, 1100, 6, endpoint=True)
        ticks2 = np.linspace(-30, 30, 5, endpoint=True)
        tickss = [ticks1, ticks2]

    elif var == 'siconc':
        cmap1 = cmc.davos_r
        levels1 = np.linspace(0, 100, 21, endpoint=True)
        norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

        cmap2 = drywet(25, cmc.vik_r)
        levels2 = np.linspace(-10, 10, 21, endpoint=True)
        norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)

        cmaps = [cmap1, cmap1, cmap2]
        norms = [norm1, norm1, norm2]

        cbarlabel = '%'

        extends = ['neither', 'both']

        ticks1 = np.linspace(0, 100, 6, endpoint=True)
        ticks2 = np.linspace(-10, 10, 5, endpoint=True)
        tickss = [ticks1, ticks2]

    elif var == 'ts':
        cmap1 = cmc.roma_r
        levels1 = np.linspace(-40, 40, 20, endpoint=True)
        norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

        cmap2 = custom_div_cmap(23, cmc.vik)
        levels2 = np.linspace(-10, 10, 20, endpoint=True)
        norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)

        cmaps = [cmap1, cmap1, cmap2]
        norms = [norm1, norm1, norm2]

        cbarlabel = '$^{o}$C'

        extends = ['both', 'both']

        ticks1 = np.linspace(-40, 40, 5, endpoint=True)
        ticks2 = np.linspace(-10, 10, 5, endpoint=True)
        tickss = [ticks1, ticks2]

    elif var == 'zg':
        levels1 = np.linspace(5200, 6000, 28, endpoint=True)
        cmap1 = cmc.roma_r
        norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

        levels2 = np.linspace(-100, 100, 27, endpoint=True)
        cmap2 = drywet(27, cmc.vik_r)
        norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)

        cmaps = [cmap1, cmap1, cmap2]
        norms = [norm1, norm1, norm2]

        cbarlabel = 'gpm'

        extends = ['both', 'both']

        ticks1 = np.linspace(5200, 6000, 3, endpoint=True)
        ticks2 = np.linspace(-100, 100, 5, endpoint=True)
        tickss = [ticks1, ticks2]

    elif var == 'wind500':
        levels1 = MaxNLocator(nbins=20).tick_values(2, 6)
        cmap1 = cmc.roma_r
        norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

        levels2 = MaxNLocator(nbins=15).tick_values(-3, 3)
        cmap2 = custom_div_cmap(25, cmc.vik)
        norm2 = colors.TwoSlopeNorm(vmin=-3, vcenter=0., vmax=3)

        cmaps = [cmap1, cmap1, cmap2]
        norms = [norm1, norm1, norm2]

        cbarlabel = 'm s$^{-1}$'

        extends = ['both', 'both']

        ticks1 = [2, 3, 4, 5, 6]
        ticks2 = [-3, -2, -1, 0, 1, 2, 3]
        tickss = [ticks1, ticks2]

    return cmaps, norms, cbarlabel, extends, tickss


