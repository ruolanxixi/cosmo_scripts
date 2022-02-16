import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from plotcosmomap import plotcosmo, add_gridline_labels
import matplotlib.gridspec as gridspec

path1 = "/Users/kaktus/Documents/ETH/BECCY/myscripts/data/ctrl_ex/"
file1 = '01_TOT_PREC_DJF.nc'
ds = xr.open_dataset(path1 + file1)
prec_ctrl_DJF = ds["TOT_PREC"].values[0, :, :]
lat = ds["lat"].values
lon = ds["lon"].values
pole_lat = ds["rotated_pole"].grid_north_pole_latitude
pole_lon = ds["rotated_pole"].grid_north_pole_longitude
rlat = ds["rlat"].values
rlon = ds["rlon"].values
ds.close()

ncol = 1
nrow = 1

gs = gridspec.GridSpec(nrow, ncol)
rot_pole_crs = ccrs.RotatedPole(pole_latitude=pole_lat, pole_longitude=pole_lon)
ax = plt.subplot(gs[0], projection=rot_pole_crs)
cs = ax.pcolormesh(rlon, rlat, prec_ctrl_DJF, cmap='YlGnBu', vmin=0, vmax=20, shading="auto")
ax = plotcosmo(ds, ax)

plt.show()

