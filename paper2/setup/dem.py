# Description: adapt input DEM for EXTPAR
###############################################################################
# Modules
###############################################################################
import xarray as xr
import numpy as np

###############################################################################
# Data
###############################################################################
path_dem = '/net/o3/hymet/rxiang/EXTPAR/input'
path_out = '/net/o3/hymet/rxiang/EXTPAR/input_modified'

# DEM tiles
tiles_dem = ("N90-N60_E030-E060", "N90-N60_E060-E090", "N90-N60_E090-E120",
             "N90-N60_E120-E150", "N90-N60_E150-E180", "N90-N60_W180-W150",

             "N60-N30_E000-E030", "N60-N30_E030-E060", "N60-N30_E060-E090",
             "N60-N30_E090-E120", "N60-N30_E120-E150", "N60-N30_E150-E180",
             "N60-N30_W180-W150", "N60-N30_W150-W120",

             "N30-N00_E030-E060", "N30-N00_E060-E090", "N30-N00_E090-E120",
             "N30-N00_E120-E150", "N30-N00_E150-E180", "N30-N00_W180-W150",

             "N00-S30_E060-E090", "N00-S30_E090-E120", "N00-S30_E120-E150",
             "N00-S30_E150-E180")

# Loop through tiles and process
def modify_and_save(tile_id):

    print((" Process tile " + tile_id + " ").center(60, "#"))

    # Load DEM data
    ds = xr.open_dataset(path_dem + "/" + "MERIT_" + tile_id + ".nc", chunks={"lat": 500, "lon": 500})
    dem = ds["Elevation"].values  # 16-bit integer
    ds.close()

    ds = xr.open_dataset(path_out + "/slm/" + "SLM_" + tile_id + ".nc", chunks={"lat": 500, "lon": 500})
    slm = ds["SLM"].values  # 16-bit integer
    ds.close()

    ds = xr.open_dataset(path_out + "/bed/" + "BED_" + tile_id + ".nc", chunks={"lat": 500, "lon": 500})
    bathymetry = ds["z"].values  # 16-bit integer
    ds.close()

    # Lift the DEM by 120 m
    dem += 120

    # Identify the cells where the DEM is nan and the mask is not 0
    fill_cells = np.isnan(dem) & (slm != 0)

    # Fill the DEM cells with bathymetry + 120 m
    dem = dem.where(~fill_cells, other=bathymetry + 120)

    # Identify the cells where the mask is 0
    nan_cells = slm == 0

    # Set the DEM cells to nan where the mask is 0
    dem = dem.where(~nan_cells, other=np.nan)

    # Count the number of cells that have undergone changes in step 2 and step 3
    num_fill_cells = np.count_nonzero(fill_cells)
    num_nan_cells = np.count_nonzero(nan_cells)
    print(f"Number of cells filled with bathymetry: {num_fill_cells}")
    print(f"Number of cells set to nan: {num_nan_cells}")

    ds = xr.open_dataset(path_dem + "/" + "MERIT_" + tile_id + ".nc", mask_and_scale=False)
    ds["Elevation"] = dem
    ds.to_netcdf(path_out + "/merit/" + "MERIT_" + tile_id + ".nc", format="NETCDF4",
                  encoding={"lat": {"_FillValue": None},
                            "lon": {"_FillValue": None}})

    del dem, slm, bathymetry, num_fill_cells, num_nan_cells, ds

for tile_id in tiles_dem:
    modify_and_save(tile_id)




