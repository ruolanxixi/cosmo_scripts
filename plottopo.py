# -------------------------------------------------------------------------------
# modules
#
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.colors as colors
from copy import copy
from plotcosmomap import plotcosmo
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from cmcrameri import cm
from auxiliary import truncate_colormap, spat_agg_1d, spat_agg_2d
# -------------------------------------------------------------------------------
# import data
#
path = "/project/pr94/rxiang/data/extpar/"
file1 = 'extpar_12km_1118x670_MERIT_raw.nc'
file2 = 'extpar_EAS_ext_12km_merit_adj.nc'

ds = xr.open_dataset(path + file1)
elev_ctrl = ds["HSURF"].values[:, :]
lat = ds["lat"].values
lon = ds["lon"].values
ds.close()

ds = xr.open_dataset(path + file2)
elev_topo1 = ds["HSURF"].values[:, :]
ds.close()

elev_diff = elev_ctrl - elev_topo1
elev_diff = np.ma.masked_where(elev_diff < 1, elev_diff)

color1 = plt.get_cmap('terrain')(np.linspace(0.22, 1, 256))
all_colors = np.vstack(color1)
cmap1 = colors.LinearSegmentedColormap.from_list('terrain', all_colors)

color1 = plt.get_cmap('terrain')(np.linspace(0.22, 0.9, 256))
all_colors = np.vstack(color1)
cmap2 = colors.LinearSegmentedColormap.from_list('terrain', all_colors)

palette = copy(cmap2)
palette.set_under('white', 0)
palette.set_bad(color='white')

ticks = np.arange(0., 6500.0, 500.0)
cmap1 = truncate_colormap(cm.bukavu, 0.55, 1.0)
levels1 = np.arange(0., 6500.0, 500.0)
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, extend="max")

levels2 = np.arange(0., 3000.0, 250.0)
ticks = np.arange(0., 3000.0, 500.0)
cmap2 = cm.lajolla
norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, extend="max")
# -------------------------------------------------------------------------------
# plot
#
def topo(ax0, ax1, ax2):
    ax0 = plotcosmo(ax0)
    cs0 = ax0.pcolormesh(lon, lat, elev_ctrl, transform=ccrs.PlateCarree(), cmap=cmap1, norm=norm1)
    ax0.contour(lon, lat, elev_ctrl, levels=levels1, colors='k', linewidths=.3)
    ax0.add_feature(cfeature.LAND)
    ax0.add_feature(cfeature.OCEAN, zorder=100, edgecolor='k')
    ax0.add_feature(cfeature.COASTLINE)
    ax0.add_feature(cfeature.BORDERS)
    ax0.add_feature(cfeature.LAKES, alpha=0.5)
    ax0.add_feature(cfeature.RIVERS)
    cs1 = ax1.pcolormesh(lon, lat, elev_topo1, transform=ccrs.PlateCarree(), cmap=cmap1, norm=norm1)
    ax1.contour(lon, lat, elev_topo1, levels=levels1, colors='k', linewidths=.3)
    ax1.add_feature(cfeature.LAND)
    ax1.add_feature(cfeature.OCEAN, zorder=100, edgecolor='k')
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.BORDERS)
    ax1.add_feature(cfeature.LAKES, alpha=0.5)
    ax1.add_feature(cfeature.RIVERS)
    cs2 = ax2.pcolormesh(lon, lat, elev_diff, transform=ccrs.PlateCarree(), cmap=cmap2, norm=norm2)
    ax2.contour(lon, lat, elev_diff, levels=levels2, colors='k', linewidths=.3)
    ax2.add_feature(cfeature.OCEAN, zorder=100, edgecolor='k')
    ax2.add_feature(cfeature.COASTLINE)
    ax2.add_feature(cfeature.BORDERS)
    ax2.add_feature(cfeature.LAKES, alpha=0.5)
    ax2.add_feature(cfeature.RIVERS)
    return ax0, ax1, ax2, cs0, cs1, cs2
