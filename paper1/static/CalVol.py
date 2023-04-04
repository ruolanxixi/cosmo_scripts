from netCDF4 import Dataset
import math
import numpy as np
from scipy.interpolate import interp2d

path = "/project/pr133/rxiang/data/extpar/"
file1 = 'extpar_EAS_ext_12km_merit_unmod_topo.nc'
file2 = 'extpar_12km_878x590_topo1.nc'


def distance(lat1, lon1, lat2, lon2):
    # convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    r = 6371  # Earth's radius in km
    return r * c


# open netCDF file
nc = Dataset(path + file1, 'r')

# get lat and lon variables
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]

xb = np.arange(0, 1119)
yb = np.arange(0, 671)

x = np.linspace(0.5, 1117.5, 1118)
y = np.linspace(0.5, 669.5, 670)

interp_func = interp2d(x, y, lat, kind='linear')
lat_bc = interp_func(xb, yb)

interp_func = interp2d(x, y, lon, kind='linear')
lon_bc = interp_func(xb, yb)

#%%
# calculate size of each grid in km
dlat = distance(lat[0], lon[0], lat[1], lon[0])
dlon = distance(lat[0], lon[0], lat[0], lon[1])
grid_size_km = dlat * dlon

# calculate domain size in km
min_lat, max_lat = min(lat), max(lat)
min_lon, max_lon = min(lon), max(lon)
domain_size_km = distance(min_lat, min_lon, max_lat, max_lon) * distance(0, 0, dlat, dlon)

# print results
print(f"Grid size: {grid_size_km:.2f} km")
print(f"Domain size: {domain_size_km:.2f} km")
