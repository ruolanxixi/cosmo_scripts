import h5py
import numpy as np
import datetime as dt
import xarray as xr
from netCDF4 import Dataset
from netCDF4 import date2num,num2date
import sys

# input_file = sys.argv[1]
input_file = '/project/pr133/rxiang/data/obs/pr/IMERG/v06/3B-MO.MS.MRG.3IMERG.20010101-S000000-E235959.01.V06B.HDF5'
f = h5py.File(input_file, 'r')
# %%
time_ = f['Grid']['time'][:]

pr = f['Grid']['precipitation'][:]
pr[pr == -9999.9] = np.nan
pr = pr.swapaxes(1, 2) * 24

lat_ = np.linspace(-90 + 0.05, 90 - 0.05, 1800)
lon_ = np.linspace(-180 + 0.05, 180 - 0.05, 3600)
# %%
ncfile = Dataset(input_file[:-4] + 'nc', mode='w', format='NETCDF4_CLASSIC')
print(ncfile)
# %%
lat_dim = ncfile.createDimension('lat', 1800)     # latitude axis
lon_dim = ncfile.createDimension('lon', 3600)    # longitude axis
time_dim = ncfile.createDimension('time', None) # unlimited axis (can be appended to).
for dim in ncfile.dimensions.items():
    print(dim)

ncfile.title='GPM IMERG Final Precipitation L3 1 day 0.1 degree x 0.1 degree (GPM_3IMERGDF)'
print(ncfile.title)

lat = ncfile.createVariable('lat', np.float32, ('lat',))
lat.units = 'degrees_north'
lat.long_name = 'latitude'
lon = ncfile.createVariable('lon', np.float32, ('lon',))
lon.units = 'degrees_east'
lon.long_name = 'longitude'
time = ncfile.createVariable('time', np.float64, ('time',))
time.units = 'seconds since 1970-01-01 00:00:00'
time.long_name = 'time'
# Define a 3D variable to hold the data
precipitation = ncfile.createVariable('precipitation',np.float64,('time','lat','lon')) # note: unlimited dimension is leftmost
precipitation.units = 'mm/day'
precipitation.standard_name = 'precipitation' # this is a CF standard name

lat[:] = lat_
lon[:] = lon_
precipitation[:, :, :] = pr
time[:] = time_

ncfile.close()
