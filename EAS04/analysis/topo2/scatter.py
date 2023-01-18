# -------------------------------------------------------------------------------
# modules
#
from netCDF4 import Dataset
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from plotcosmomap import plotcosmo04sm_notick, pole04, colorbar
import cartopy.crs as ccrs
from numpy import inf
import matplotlib.gridspec as gridspec
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mycolor import custom_div_cmap, cbr_wet, cbr_drywet, drywet, custom_seq_cmap_, wind, hotcold, conv
from pyproj import Transformer
import scipy.ndimage as ndimage
import matplotlib

# -------------------------------------------------------------------------------
# read data
# %%
ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_BECCY_4.4km_merit_unmod_topo.nc')
ds = ds.sel(rlon=slice(-28.24, -2.25), rlat=slice(-12.7, 13.29))
hsurf_ctrl = ds['HSURF'].values[:, :]
ds.close()
ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_BECCY_4.4km_merit_env_topo_adj.nc')
ds = ds.sel(rlon=slice(-28.24, -2.25), rlat=slice(-12.7, 13.29))
hsurf_topo2 = ds['HSURF'].values[:, :]
ds.close()
hsurf_diff = hsurf_topo2 - hsurf_ctrl

ds = xr.open_dataset('/project/pr133/rxiang/data/cosmo/EAS04_ctrl/indices/day/2001-2005_smr_all_day_perc.nc')
mean_ctrl = ds['mean'].values[:, :]
ds.close()
ds = xr.open_dataset('/project/pr133/rxiang/data/cosmo/EAS04_topo2/indices/day/2001-2005_smr_all_day_perc.nc')
mean_topo2 = ds['mean'].values[:, :]
mean_diff = (mean_topo2 - mean_ctrl)/mean_ctrl

dt = np.transpose(np.array([hsurf_diff.flatten(), mean_diff.flatten()]))
df = pd.DataFrame(dt, columns=['elev','var'])

# plot
# %%
# Set the figure size
# plt.rcParams["figure.figsize"] = [7, 7]
# plt.rcParams["figure.autolayout"] = True
#
# # Scatter plot
# plt.scatter(hsurf_diff.flatten(), mean_diff.flatten(), s=0.5, marker='o')
# plt.ylim([-10,10])
# plt.show()

# Load the planets dataset and initialize the figure
g = sns.displot(df, x="elev", y="var", binwidth=(20, 0.1), cbar=True, vmin=0, vmax=20)
g.set(ylim=(-10, 10))
plt.show()

# g = sns.lmplot(data=df, x="elev", y="var", scatter_kws={"color": "white"})
# g.set(ylim=(-10, 10))
# plt.show()
