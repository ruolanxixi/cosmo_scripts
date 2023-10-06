import xarray as xr
from pyproj import CRS, Transformer
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib
import cmcrameri.cm as cmc
import numpy.ma as ma
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mycolor import custom_white_cmap
from plotcosmomap import plotcosmo_notick, pole, plotcosmo04_notick, plotcosmo_notick_lgm, plotcosmo04_notick_lgm

file = "/scratch/snx3000/rxiang/climateV.nc"

ds = xr.open_dataset(file)
cv = ds['climateV'].values  # replace 'variable_name' with the appropriate variable name
ds.close()

###############################################################################
# %% Plot
###############################################################################
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()
fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': rot_pole_crs})
ax = plotcosmo04_notick(ax)
cmap = cmc.roma_r
levels = MaxNLocator(nbins=100).tick_values(0, 15000)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
cs = ax.pcolormesh(rlon, rlat, cv, cmap=cmap, norm=norm, shading="auto")

plt.show()

