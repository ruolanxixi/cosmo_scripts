import xarray as xr

tiles_dem = ("N90-N60_E030-E060", "N90-N60_E060-E090", "N90-N60_E090-E120",
             "N90-N60_E120-E150", "N90-N60_E150-E180", "N90-N60_W180-W150",

             "N60-N30_E000-E030", "N60-N30_E030-E060", "N60-N30_E060-E090",
             "N60-N30_E090-E120", "N60-N30_E120-E150", "N60-N30_E150-E180",
             "N60-N30_W180-W150", "N60-N30_W150-W120",

             "N30-N00_E030-E060", "N30-N00_E060-E090", "N30-N00_E090-E120",
             "N30-N00_E120-E150", "N30-N00_E150-E180", "N30-N00_W180-W150",

             "N00-S30_E060-E090", "N00-S30_E090-E120", "N00-S30_E120-E150",
             "N00-S30_E150-E180")

path_dem = "/net/o3/hymet/rxiang/EXTPAR/input"
path_out = "/net/o3/hymet/rxiang/EXTPAR/input_modified"

slm = xr.open_dataset(path_out + "/" + "SLM.nc", chunks={"lat": 50, "lon": 50})
bed = xr.open_dataset(path_out + "/" + "BED.nc", chunks={"lat": 50, "lon": 50})

def regrid_and_save(tile_id):
    print((" Process tile " + tile_id + " ").center(60, "#"))

    # Load DEM data
    ds = xr.open_dataset(path_dem + "/" + "MERIT_" + tile_id + ".nc", chunks={"lat": 500, "lon": 500})
    target_lats = ds["lat"]
    target_lons = ds["lon"]
    del ds

    # Regrid slm and bed datasets
    ds_out_slm = slm.interp(lat=target_lats, lon=target_lons, method="linear")
    ds_out_slm.to_netcdf(f"{path_out}/slm/SLM_{tile_id}.nc")
    del ds_out_slm

    ds_out_bed = bed.interp(lat=target_lats, lon=target_lons, method="linear")
    ds_out_bed.to_netcdf(f"{path_out}/bed/BED_{tile_id}.nc")
    del ds_out_bed

for tile_id in tiles_dem:
    regrid_and_save(tile_id)
