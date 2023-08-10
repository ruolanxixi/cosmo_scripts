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

path_dem = '/scratch/snx3000/rxiang/EXTPAR/input_linked'
path_out = '/scratch/snx3000/rxiang/EXTPAR/input_modified'

for i in tiles_dem:
    print((" Process tile " + i + " ").center(60, "#"))

    # Load DEM data
    ds = xr.open_dataset(path_dem + '/' + 'MERIT_' + i + '.nc', mask_and_scale=False)
    lon, lat = ds["lon"].values, ds["lat"].values
    lon_min = ds["lon"].values.min()
    lat_min = ds["lat"].values.min()
    ds.close()
    print("Size of DEM data: %.2f" % (lon.nbytes / (10.0 ** 9)) + " GB")

    with open(f'{path_out}/grid/{i}.txt', 'w') as file:
        file.write('gridtype = lonlat\n'
                   'xsize = 36000\n'
                   'ysize = 36000\n'
                   f'xfirst = {lon_min}\n'
                   f'yfirst = {lat_min}\n'
                   'xinc = 30/36000\n'
                   'yinc = 30/36000\n')
