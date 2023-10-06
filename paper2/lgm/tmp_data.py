# Description: Plot pollen precipitation proxy data from Sun et al. (2021b)
#              (Fig. 1)
#
# Author: Christian Steger, September 2023

# Load modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import pandas as pd
from cmcrameri import cm
import cartopy.crs as ccrs
import cartopy.feature as feature
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
import cmcrameri.cm as cmc

mpl.style.use("classic")

# Change latex fonts
mpl.rcParams["mathtext.fontset"] = "custom"
# custom mathtext font (set default to Bitstream Vera Sans)
mpl.rcParams["mathtext.default"] = "rm"
mpl.rcParams["mathtext.rm"] = "Bitstream Vera Sans"

# Paths to folders
path_data = "/project/pr133/rxiang/script/mapping_cosmo/paper2/lgm/"

###############################################################################
# Plot data
###############################################################################

# Load data
df = pd.read_excel(path_data + "tmp.xlsx", index_col=0)
lon = df["Longitude"].values
lat = df["Latitude"].values
annt = df["TANN"].values

# Colormap
levels = MaxNLocator(nbins=50).tick_values(-10, 0)
cmap = cmc.davos
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

# Plot
fig = plt.figure(figsize=(10, 7))
gs = gridspec.GridSpec(2, 1, left=0.1, bottom=0.1, right=0.9, top=0.9,
                       hspace=0.08, wspace=0.05,
                       height_ratios=[1.0, 0.05])
# -----------------------------------------------------------------------------
ax = plt.subplot(gs[0], projection=ccrs.PlateCarree())
# plt.scatter(lon, lat, c=annp, s=80, cmap=cmap, norm=norm)  # annual
plt.scatter(lon, lat, c=annt, s=80, cmap=cmap, norm=norm)  # JJA
ax.add_feature(feature.BORDERS, linestyle="-", linewidth=0.6)
ax.add_feature(feature.COASTLINE, linestyle="-", linewidth=0.6)
ax.set_aspect("auto")
ax.set_extent([70.0, 150.0, 10.0, 60.0], crs=ccrs.PlateCarree())
# -----------------------------------------------------------------------------
ax = plt.subplot(gs[1])
cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, ticks=[-10, -8, -6, -4, -2, 0],
                               orientation="horizontal")
cb.ax.tick_params(labelsize=10)
plt.xlabel("[K]", fontsize=10)

plt.show()
plotpath = "/project/pr133/rxiang/figure/paper2/results/lgm/"
fig.savefig(plotpath + 'tmp_proxy.png', dpi=500, transparent='True')
