# -------------------------------------------------------------------------------
# Plot atmospheric river
# -------------------------------------------------------------------------------
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotcosmomap import plotcosmo, pole, colorbar
import pandas as pd
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

ds = xr.open_dataset('/project/pr133/rxiang/data/era5/pr/mo/era5.mo.1980-2022.nc')
tp = np.sum(ds['tp'].values[:, :, :], axis=0)*1000
lat = ds['latitude'].values
lon = ds['longitude'].values
# %%
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson(central_longitude=180, globe=None)})
levels = np.linspace(0, 3000, 21, endpoint=True)
cmap = cmc.roma
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
q = ax.pcolormesh(lon, lat, tp, cmap=cmap, norm=norm, shading="auto", transform=ccrs.PlateCarree())
ax.set_title('Annual Total Pecipitation (1980-2022)', fontweight='bold', pad=12, fontsize=12)
cax = fig.add_axes([ax.get_position().x0+0.15, ax.get_position().y0 - 0.1, ax.get_position().width*0.7, 0.03])
cb = fig.colorbar(q, cax=cax, orientation='horizontal',extend='max', ticks=np.linspace(0, 1000, 11, endpoint=True))
cb.set_label('mm', fontsize=11)
# adjust figure
fig.show()
# save figure
plotpath = "/project/pr133/rxiang/figure/paper1/results/IVT/"
fig.savefig(plotpath + f'annual.png', dpi=300)
# plt.close(fig)
