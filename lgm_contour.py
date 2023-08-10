# Load modules
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Get files
path = "/Users/kaktus/Documents/ETH/BECCY/myscripts/data/ctrl/"
file = 'test.nc'

# open file
ds = xr.open_dataset(path + file)
prec = ds["TOT_PREC"].values[0, :, :]
lat = ds["lat"].values
lon = ds["lon"].values
ds.close()

###############################################################################
# Normalize colorbar
###############################################################################
color1 = plt.get_cmap('terrain')(np.linspace(0.22, 1, 256))
all_colors = np.vstack(color1)
cmap = colors.LinearSegmentedColormap.from_list('terrain', all_colors)


###############################################################################
# Plot
###############################################################################
def main():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.plot(projection=ccrs.PlateCarree())

    cs = ax.contourf(lon, lat, prec, transform=ccrs.PlateCarree(), levels=np.linspace(0, 20, 21), cmap='YlGnBu', vmin=0,
                     vmax=20, extend='max')

    ax.set_title("Summer Precipitation")
    ax.set_extent([78, 150, 7, 55], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

    ax.gridlines(draw_labels=True, linewidth=1, color='grey', alpha=0.5, linestyle='--')

    cb = fig.colorbar(cs, orientation='horizontal', shrink=0.7, ax=ax, pad=0.1)
    cb.set_label('mm/day')

    fig.savefig('test.png', dpi=300)


if __name__ == '__main__':
    main()
###############################################################################
