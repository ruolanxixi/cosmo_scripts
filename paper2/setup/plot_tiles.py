# Description: plot the titles
###############################################################################
# Modules
###############################################################################
import xarray as xr
from plotcosmomap import pole
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from auxiliary import truncate_colormap
from matplotlib.colors import BoundaryNorm
import cmcrameri.cm as cmc
import matplotlib.ticker as mticker
import cartopy.feature as cfeature

data = '/project/pr133/rxiang/data/extpar/extpar_EAS_ext_12km_merit_unmod_topo.nc'
ds = xr.open_dataset(data)
hsurf = ds['HSURF'].values[...]
lat_ = ds["lat"].values[...]
lon_ = ds["lon"].values[...]
ds.close()

###############################################################################
# %% Plot
###############################################################################
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()

levels = np.arange(0., 6500.0, 500.0)
ticks = np.arange(0., 6500.0, 1000.0)
cmap = truncate_colormap(cmc.bukavu, 0.55, 1.0)
norm = BoundaryNorm(levels, ncolors=cmap.N, extend="max")

ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([20, 180, -30, 90], crs=ccrs.PlateCarree())
ax.pcolormesh(lon_, lat_, hsurf, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, linewidth=1,
                      color='grey', alpha=1, linestyle='-')
gl.xlocator = mticker.FixedLocator([0, 30, 60, 90, 120, 150, 180])
gl.ylocator = mticker.FixedLocator([-30, 0, 30, 60, 90])

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAKES, alpha=0.5)

plt.show()

