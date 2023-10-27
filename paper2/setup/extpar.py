# This script is used for updating the extpar file used for paleoclimaate simulations
# Author: Ruolan Xiang (ruolan.xiang@env.ethz.ch)

# load modules
import xarray as xr
import dask.array as da
from pyproj import CRS, Transformer
import numpy as np
from scipy import interpolate
import time
##################################################################
# pre-processing
##################################################################

##################################################################
# Process MERIT data
##################################################################
# DEM tiles
# tiles_dem = ("MERIT_N90-N60_E000-E030.nc", "MERIT_N90-N60_E030-E060.nc"
#              "MERIT_N90-N60_E060-E090.nc", "MERIT_N90-N60_E090-E120.nc",
#              "MERIT_N90-N60_E120-E150.nc", "MERIT_N90-N60_E150-E180.nc",
#              "MERIT_N90-N60_W180-W150.nc",
#              "MERIT_N60-N30_E000-E030.nc", "MERIT_N60-N30_E030-E060.nc",
#              "MERIT_N60-N30_E060-E090.nc", "MERIT_N60-N30_E090-E120.nc",
#              "MERIT_N60-N30_E120-E150.nc", "MERIT_N60-N30_E150-E180.nc",
#              "MERIT_N60-N30_W180-W150.nc",
#              "MERIT_N30-N00_E000-E030.nc", "MERIT_N30-N00_E030-E060.nc",
#              "MERIT_N30-N00_E060-E090.nc", "MERIT_N30-N00_E090-E120.nc",
#              "MERIT_N30-N00_E120-E150.nc", "MERIT_N30-N00_E150-E180.nc",
#              "MERIT_N30-N00_W180-W150.nc",
#              "MERIT_N00-S30_E000-E030.nc", "MERIT_N00-S30_E030-E060.nc",
#              "MERIT_N00-S30_E060-E090.nc", "MERIT_N00-S30_E090-E120.nc",
#              "MERIT_N00-S30_E120-E150.nc", "MERIT_N00-S30_E150-E180.nc",
#              "MERIT_N00-S30_W180-W150.nc")
tiles_dem1 = ("MERIT_N90-N60_E000-E030.nc", "MERIT_N90-N60_E030-E060.nc"
              "MERIT_N90-N60_E060-E090.nc", "MERIT_N90-N60_E090-E120.nc",
              "MERIT_N90-N60_E120-E150.nc", "MERIT_N90-N60_E150-E180.nc",
              "MERIT_N90-N60_W180-W150.nc")
tiles_dem2 = ("MERIT_N60-N30_E000-E030.nc", "MERIT_N60-N30_E030-E060.nc",
              "MERIT_N60-N30_E060-E090.nc", "MERIT_N60-N30_E090-E120.nc",
              "MERIT_N60-N30_E120-E150.nc", "MERIT_N60-N30_E150-E180.nc",
              "MERIT_N60-N30_W180-W150.nc")
tiles_dem3 = ("MERIT_N30-N00_E000-E030.nc", "MERIT_N30-N00_E030-E060.nc",
              "MERIT_N30-N00_E060-E090.nc", "MERIT_N30-N00_E090-E120.nc",
              "MERIT_N30-N00_E120-E150.nc", "MERIT_N30-N00_E150-E180.nc",
              "MERIT_N30-N00_W180-W150.nc")
tiles_dem4 = ("MERIT_N00-S30_E000-E030.nc", "MERIT_N00-S30_E030-E060.nc",
              "MERIT_N00-S30_E060-E090.nc", "MERIT_N00-S30_E090-E120.nc",
              "MERIT_N00-S30_E120-E150.nc", "MERIT_N00-S30_E150-E180.nc",
              "MERIT_N00-S30_W180-W150.nc")
path_dem = "/scratch/snx3000/rxiang/EXTPAR/input_linked/"
path_out = "/scratch/snx3000/rxiang/EXTPAR/input_modified/"

# load ocean bathymetry data
ds = xr.open_dataset('/scratch/snx3000/rxiang/EXTPAR/input_modified/ETOPO1_Bed_g_gmt4.grd')
btmt = ds["z"].values
blon, blat = ds["x"].values, ds["y"].values
print("Size of bathymetry data: %.2f" % (btmt.nbytes / (10.0 ** 9)) + " GB")

# load land-sea mask
ds = xr.open_dataset('/project/pr133/rxiang/data/echam5_raw/LGM/input/T159_jan_surf.lgm.veg.nc')
slm = ds["SLM"].values
mlon, mlat = ds["lon"].values, ds["lat"].values

#%%
# Loop through tiles and process
lon_concat = []
# lat_concat = []

# Loop over the netCDF files and concatenate the longitude and latitude arrays
for file in tiles_dem1:
    ds = xr.open_dataset(file)
    lon = da.from_array(ds['lon'].data, chunks=1000)
    lat = da.from_array(ds['lat'].data, chunks=1000)
    lon_concat.append(lon)
    # lat_concat.append(lat)

# Concatenate the longitude and latitude arrays along the desired dimension
lon = xr.concat(lon_concat, dim='lon')
# lat = xr.concat(lat_concat, dim='lat')

# Create a new dataset with only the longitude and latitude variables
coords_ds = xr.Dataset({'lon': lon, 'lat': lat})

# %%
# Write the longitude and latitude variables to a new netCDF file
# coords_ds.to_netcdf('coords.nc')

for i in tiles_dem1:

    print((" Process tile " + i + " ").center(60, "#"))

    # Load DEM data
    ds = xr.open_dataset(path_dem + i, mask_and_scale=False)
    lon_tile, lat_tile = ds["lon"].values, ds["lat"].values
    # topo_fill_val = ds["Elevation"]._FillValue
    # topo = ds["Elevation"].values  # 16-bit integer
    lon, lat = ds["lon"].values, ds["lat"].values
    ds.close()
    # print("Size of DEM data: %.2f" % (topo.nbytes / (10.0 ** 9)) + " GB")

    # # Convert topography back to 16-bit integer
    # topo = topo.astype(np.int16)
    # if np.any(topo[mask_water] != topo_fill_val):
    #     print("water grid cell(s) modified -> reset")
    #     topo[mask_water] = topo_fill_val
    # del mask_water



    # # Save modified topography in MERIT file
    # slic = (slice(ind_lat_0, ind_lat_0 + topo.shape[0]),
    #         slice(ind_lon_0, ind_lon_0 + topo.shape[1]))
    # ds = xr.open_dataset(path_dem + i, mask_and_scale=False)
    # ds["Elevation"][slic] = topo
    # if fac_red_out:
    #     fac_red_tile = np.zeros(ds["Elevation"].shape, dtype=np.float32)
    #     fac_red_tile[slic] = fac_red
    #     ds["fac_red"] = (("lat", "lon"), fac_red_tile)
    # ds.to_netcdf(path_out + i, format="NETCDF4",
    #              encoding={"lat": {"_FillValue": None},
    #                        "lon": {"_FillValue": None}})
