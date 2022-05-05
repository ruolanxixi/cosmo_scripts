# Load modules
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time
import matplotlib as mpl

# Path(s)
path_in = "/Users/kaktus/Documents/ETH/BECCY/myscripts/data/ctrl/"
path_plot = "/Users/kaktus/Documents/ETH/BECCY/myscripts/data/ctrl/"

###############################################################################
# Load data
###############################################################################
file = "test.nc"
ds = xr.open_dataset(path_in + file)
prec = ds["TOT_PREC"].values[0, :, :]
lat = ds["lat"].values
lon = ds["lon"].values
pole_lat = ds["rotated_pole"].grid_north_pole_latitude
pole_lon = ds["rotated_pole"].grid_north_pole_longitude
rlat = ds["rlat"].values
rlon = ds["rlon"].values
ds.close()
###############################################################################
# Plots
###############################################################################
# Use plt.contourf() -> longitudinal artefact
t_beg = time.time()
proj = ccrs.PlateCarree()
fig = plt.figure()
ax = plt.axes(projection=proj)
cs = ax.contourf(lon, lat, prec,
                 levels=np.linspace(0, 20, 21), cmap='YlGnBu', vmin=0,
                 vmax=20, extend='max')
ax.set_title("Summer Precipitation")
ax.set_extent([78, 150, 7, 55])
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)
ax.gridlines(draw_labels=True, linewidth=1, color='grey', alpha=0.5,
             linestyle='--')
cb = fig.colorbar(cs, orientation='horizontal', shrink=0.7, ax=ax, pad=0.1)
cb.set_label('mm/day')
fig.savefig(path_plot + 'test_1.png', dpi=300)
plt.close(fig)
print("Elapsed time: %.1f" % (time.time() - t_beg) + " s")

# Use plt.contourf() -> works correctly but is very slow...
lon_cont = lon.copy()
lon_cont[lon_cont < 0.0] += 360.0
# -> solution to issue: make longitudinal coordinates continuous
t_beg = time.time()
proj = ccrs.PlateCarree()
fig = plt.figure()
ax = plt.axes(projection=proj)
cs = ax.contourf(lon_cont, lat, prec,
                 levels=np.linspace(0, 20, 21), cmap='YlGnBu', vmin=0,
                 vmax=20, extend='max')
ax.set_title("Summer Precipitation")
ax.set_extent([78, 150, 7, 55])
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)
ax.gridlines(draw_labels=True, linewidth=1, color='grey', alpha=0.5,
             linestyle='--')
cb = fig.colorbar(cs, orientation='horizontal', shrink=0.7, ax=ax, pad=0.1)
cb.set_label('mm/day')
fig.savefig(path_plot + 'test_2.png', dpi=300)
plt.close(fig)
print("Elapsed time: %.1f" % (time.time() - t_beg) + " s")

# -----------------------------------------------------------------------------
# Alternatives
# -----------------------------------------------------------------------------
# Plot settings
cmap = plt.get_cmap("YlGnBu")
levels = np.linspace(0, 20, 21)
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, extend="max")
# Use plt.pcolormesh() -> I'm only using this plot function
t_beg = time.time()
proj = ccrs.PlateCarree()
fig = plt.figure()
ax = plt.axes(projection=proj)
cs = ax.pcolormesh(lon, lat, prec,
                   cmap=cmap, norm=norm, shading="auto")
ax.set_title("Summer Precipitation")
ax.set_extent([78, 150, 7, 55])
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)
ax.gridlines(draw_labels=True, linewidth=1, color='grey', alpha=0.5,
             linestyle='--')
cb = fig.colorbar(cs, orientation='horizontal', shrink=0.7, ax=ax, pad=0.1)
cb.set_label('mm/day')
fig.savefig(path_plot + 'test_3.png', dpi=300)
plt.close(fig)
print("Elapsed time: %.1f" % (time.time() - t_beg) + " s")

# Plot in rotated latitude/longitude coordinates -> this is the way I normally
# plot maps...
t_beg = time.time()
rot_pole_crs = ccrs.RotatedPole(pole_latitude=pole_lat,
                                pole_longitude=pole_lon)
fig = plt.figure()
ax = plt.axes(projection=rot_pole_crs)
cs = ax.pcolormesh(rlon, rlat, prec, cmap=cmap, norm=norm, shading="auto")
ax.set_title("Summer Precipitation")
ax.set_extent([78, 150, 7, 55], crs=proj)
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
             linewidth=1, color='grey', alpha=0.5, linestyle='--')
cb = fig.colorbar(cs, orientation='horizontal', shrink=0.7, ax=ax, pad=0.1)
cb.set_label('mm/day')
fig.savefig(path_plot + 'test_4.png', dpi=300)
plt.close(fig)
print("Elapsed time: %.1f" % (time.time() - t_beg) + " s")

