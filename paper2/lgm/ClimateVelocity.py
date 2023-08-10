import xarray as xr
import numpy as np
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree

# --------------------------------------------------------------------
file_pd = '/project/pr133/rxiang/data/cosmo/EAS11_ctrl/monsoon/T_2M/01-05.T_2M.cpm.nc'  # present
file_lgm = '/project/pr133/rxiang/data/cosmo/EAS11_lgm/monsoon/T_2M/01-05.T_2M.cpm.nc'  # future
path = '/project/pr133/rxiang/data/cosmo/ClimateVelocity/'  # where you want to write your climate velocity files

distance_unit = 12  # km the distance unit of your tiff files
time_interval = 21  # ka the time difference from present to future

# --------------------------------------------------------------------
# 1. read NetCDF
# present data
ds = xr.open_dataset(file_pd)
present = np.nanmean(ds['T_2M'].values, axis=0)  # replace 'variable_name' with the appropriate variable name
ds.close()
# future data
ds = xr.open_dataset(file_lgm)
future = np.nanmean(ds['T_2M'].values, axis=0)  # replace 'variable_name' with the appropriate variable name
lon = ds["lon"].values  # [degree]
lat = ds["lat"].values  # [degree]
ds.close()

# --------------------------------------------------------------------
# 2. user-defined threshold
# Search radius for (nearest) neighbour grid cells
rad_search = 50.0 * 1000.0  # [m]

# --------------------------------------------------------------------
# 3. calculate climate velocity
# Compute ECEF coordinates and construct tree
crs_ecef = CRS.from_dict({"proj": "geocent", "ellps": "sphere"})
crs_latlon = CRS.from_dict({"proj": "latlong", "ellps": "sphere"})
trans = Transformer.from_crs(crs_latlon, crs_ecef, always_xy=True)
x_ecef, y_ecef, z_ecef = trans.transform(lon, lat, np.zeros_like(lon))
pts_gc = np.vstack((x_ecef.ravel(), y_ecef.ravel(), z_ecef.ravel())).transpose()
tree = cKDTree(pts_gc)

distance = np.empty(present.shape, dtype=np.float32)

# for col in range(present.shape[1]):
#     for row in range(present.shape[0]):
#         ind_2d_ta = (row, col)
#         ind_lin = tree.query_ball_point([x_ecef[ind_2d_ta], y_ecef[ind_2d_ta], z_ecef[ind_2d_ta]], r=rad_search)

northDifference = np.full(present.shape, np.nan)
eastDifference = np.full(present.shape, np.nan)

uniqueP = np.unique(present)
Findex = []
for i_uniqueP in uniqueP:
    Findex.append(np.where(future == i_uniqueP))

for col in range(present.shape[1]):
    for row in range(present.shape[0]):
        try:
            temindex = np.where(uniqueP == present[row][col])[0][0]
            Findexrow = np.array(Findex[temindex][0])
            Findexcol = np.array(Findex[temindex][1])
            temdistance = np.square(Findexrow - row) + np.square(Findexcol - col)
            temmindistance = np.sqrt(np.min(temdistance))
            temminindex = np.argmin(temdistance)

            distance[row][col] = temmindistance
            northDifference[row][col] = (Findexrow[temminindex] - row) / temmindistance
            eastDifference[row][col] = (Findexcol[temminindex] - col) / temmindistance
        except:
            continue

# ####################################################################
# ####################################################################
# 4 write climate velocity to NetCDF file
climateV = distance * distance_unit / time_interval
northV = northDifference * distance_unit / time_interval
eastV = eastDifference * distance_unit / time_interval

climateV_dataset = xr.Dataset({'climateV': (('rlat', 'rlon'), climateV)})
northV_dataset = xr.Dataset({'northV': (('rlat', 'rlon'), northV)})
eastV_dataset = xr.Dataset({'eastV': (('rlat', 'rlon'), eastV)})

climateV_dataset.to_netcdf(path + 'climateV.nc', format="NETCDF4", encoding={"time": {"_FillValue": None}})
northV_dataset.to_netcdf(path + 'northV.nc', format="NETCDF4", encoding={"time": {"_FillValue": None}})
eastV_dataset.to_netcdf(path + 'eastV.nc', format="NETCDF4", encoding={"time": {"_FillValue": None}})
