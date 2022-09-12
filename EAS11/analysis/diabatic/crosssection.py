import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cmcrameri.cm as cmc
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
from netCDF4 import Dataset

from wrf import to_np, getvar, CoordPair, vertcross

# Open the NetCDF file
ncfile = Dataset('/project/pr133/rxiang/data/cosmo/EAS11_ctrl/24h3D/SOHR_SUM/01_SOHR_SUM.nc')

# Extract the model height and variable
z = getvar(ncfile, "SOHR_SUM")
wspd = getvar(ncfile, "uvmet_wspd_wdir", units="kt")[0,:]

