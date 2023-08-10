import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from skimage.measure import find_contours
import pickle

ds = xr.open_dataset('/project/pr133/rxiang/data/extpar/extpar_EAS_ext_12km_merit_LGM_consistent_TCL.nc')
hsurf = ds['HSURF'].values[...]
rlon = ds['rlon'].values[...]
rlat = ds['rlat'].values[...]
pole_lat = ds["rotated_pole"].grid_north_pole_latitude
pole_lon = ds["rotated_pole"].grid_north_pole_longitude
rot_pole_crs = ccrs.RotatedPole(pole_latitude=pole_lat, pole_longitude=pole_lon)


def coastline_contours(lon, lat, mask_bin):
    """Compute coastline contours.

    Compute coastline contours from binary land-sea mask.

    Parameters
    ----------
    lon : ndarray of double
        Array (1-dimensional) with geographic longitude [degree]
    lat: ndarray of double
        Array (1-dimensional) with geographic latitude [degree]
    mask_bin: str
        Array (2-dimensional) with binary land-sea mask (0: water, 1: land)

    Returns
    -------
    contours_latlon : list
        List with contour lines in latitude/longitude coordinates [degree]"""

    # Check arguments
    if (lat.ndim != 1) or (lon.ndim != 1):
        raise ValueError("Input coordinates arrays must be 1-dimensional")
    if (mask_bin.shape[0] != len(lat)) or (mask_bin.shape[1] != len(lon)):
        raise ValueError("Input data has inconsistent dimension length(s)")
    if (mask_bin.dtype != "uint8") or (len(np.unique(mask_bin)) != 2) \
            or (not np.all(np.unique(mask_bin) == [0, 1])):
        raise ValueError("'mask_bin' must be of type 'uint8' and may "
                         + "only contain 0 and 1")

    # Compute contour lines
    contours = find_contours(mask_bin, 0.5, fully_connected="high")

    # Get latitude/longitude coordinates of contours
    lon_ind = np.linspace(lon[0], lon[-1], len(lon) * 2 - 1)
    lat_ind = np.linspace(lat[0], lat[-1], len(lat) * 2 - 1)
    contours_latlon = []
    for i in contours:
        pts_latlon = np.empty(i.shape, dtype=np.float64)
        pts_latlon[:, 0] = lon_ind[(i[:, 1] * 2).astype(np.int32)]
        pts_latlon[:, 1] = lat_ind[(i[:, 0] * 2).astype(np.int32)]
        contours_latlon.append(pts_latlon)


    return contours_latlon

# %%
mask_water = hsurf > 0

mask_bin = (~mask_water).astype(np.uint8)  # (0: water, 1: land)

contours_rlatrlon = coastline_contours(rlon, rlat, mask_bin)

pts_latlon = np.vstack(([i for i in contours_rlatrlon]))

# Assuming contours_rlatrlon is already defined as a list
with open('/project/pr133/rxiang/data/extpar/lgm_contour.pkl', 'wb') as file:
    pickle.dump(contours_rlatrlon, file)

fig = plt.figure(figsize=(5, 4))
ext = [65, 173, 7, 61]
axs = fig.add_subplot(projection=rot_pole_crs)
axs.set_extent(ext, crs=ccrs.PlateCarree())

for contour in contours_rlatrlon:
    plt.plot(contour[:, 0], contour[:, 1], c='blue', transform=rot_pole_crs)

plt.show()
