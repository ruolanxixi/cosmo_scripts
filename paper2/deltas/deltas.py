# This script is used for plotting deltas computed for pgw4era5
###########################################
#%% load module
###########################################
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from mycolor import drywet, custom_div_cmap
from colorsetup import colorsetup
import matplotlib
import matplotlib.colors as colors

font = {'size': 14}
matplotlib.rc('font', **font)

###########################################
#%% load data
###########################################
delta_path = '/project/pr133/rxiang/data/pgw/deltas/native/day/ECHAM5'

var2d_names = ['hurs', 'ps', 'siconc', 'tas', 'ts', 'zg']
var3d_names = ['ua', 'va', 'zg', 'hur', 'ta']

climate1 = 'lgm'
climate2 = 'piControl'
climate3 = 'delta'

climates = [climate1, climate2, climate3]
labels = {'piControl': 'Pre-industrial', 'delta': 'LGM - PI', 'lgm': 'Last glacial maximum'}

data = {}

for j in range(len(climates)):
    climate = climates[j]
    data[climate] = {}
    data[climate]['label'] = labels[climate]
    for i in range(len(var2d_names)):
        var = var2d_names[i]
        data[climate][var] = {}
        ds = xr.open_dataset(f'{delta_path}/{var}_{climate}.nc', use_cftime=True)
        ds_jan = ds.sel(time=ds['time.month'] == 1)
        ds_jul = ds.sel(time=ds['time.month'] == 7)
        if var == 'zg':
            dt = np.nanmean(ds_jan[var].values[:, 15, :, :], axis=0)
        else:
            dt = np.nanmean(ds_jan[var].values[:, :, :], axis=0)
        if (var == 'tas' or var == 'ts') and (climate == 'piControl' or climate == 'lgm'):
            data[climate][var]['JAN'] = dt - 273.15
        elif var == 'ps':
            data[climate][var]['JAN'] = dt/100
        elif var == 'siconc':
            data[climate][var]['JAN'] = dt*100
        else:
            data[climate][var]['JAN'] = dt
        if var == 'zg':
            dt = np.nanmean(ds_jul[var].values[:, 15, :, :], axis=0)
        else:
            dt = np.nanmean(ds_jul[var].values[:, :, :], axis=0)
        if (var == 'tas' or var == 'ts') and (climate == 'piControl' or climate == 'lgm'):
            data[climate][var]['JUL'] = dt - 273.15
        elif var == 'ps':
            data[climate][var]['JUL'] = dt/100
        elif var == 'siconc':
            data[climate][var]['JUL'] = dt*100
        else:
            data[climate][var]['JUL'] = dt

lat = xr.open_dataset(f'{delta_path}/hurs_delta.nc')['lat'].values[:]
lon = xr.open_dataset(f'{delta_path}/hurs_delta.nc')['lon'].values[:]
lat_, lon_ = np.meshgrid(lon, lat)

###########################################
#%% plot
###########################################
# var2d_names = ['zg']
mons = ['JAN', 'JUL']
for ii in range(len(var2d_names)):
    var = var2d_names[ii]
    for jj in range(len(mons)):
        mon = mons[jj]

        ar = 1.0  # initial aspect ratio for first trial
        wi = 10  # width in inches #15
        hi = 2.7  # height in inches #10
        ncol = 3
        nrow = 1
        axs, cs, gl = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object')

        # cmap1 = cmc.davos_r
        # levels1 = np.linspace(0, 100, 21, endpoint=True)
        # norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
        #
        # cmap2 = drywet(25, cmc.vik_r)
        # levels2 = np.linspace(-100, 100, 21, endpoint=True)
        # norm2 = colors.TwoSlopeNorm(vmin=-100., vcenter=0., vmax=100.)


        # change here the lat and lon
        map_ext = [65, 173, 7, 61]

        fig = plt.figure(figsize=(wi, hi))
        left, bottom, right, top = 0.08, 0.27, 0.985, 0.98
        gs = gridspec.GridSpec(nrows=1, ncols=3, left=left, bottom=bottom, right=right, top=top,
                               wspace=0.1, hspace=0.13)

        cmaps, norms, clabel, extends, tickss = colorsetup(var)
        for i in range(3):
            sim = climates[i]
            cmap = cmaps[i]
            norm = norms[i]
            label = data[sim]['label']
            axs[0, i] = fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree())
            axs[0, i].set_extent(map_ext, crs=ccrs.PlateCarree())
            axs[0, i].coastlines(zorder=3)
            axs[0, i].stock_img()
            gl[0, i] = axs[0, i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=1, color='grey', alpha=0.5, linestyle='--')
            gl[0, i].right_labels = False
            gl[0, i].top_labels = False
            gl[0, i].left_labels = False
            gl[0, i].ylocator = plt.FixedLocator([10, 30, 50])
            cs[0, i] = axs[0, i].pcolormesh(lon, lat, data[sim][var][mon], cmap=cmap, norm=norm, shading="auto",
                                            transform=ccrs.PlateCarree())
            axs[0, i].text(0.5, 1.1, f'{label}', ha='center', va='center', fontsize=14, transform=axs[0, i].transAxes)

        gl[0, 0].left_labels = True

        cax = fig.add_axes(
            [axs[0, 0].get_position().x0 + 0.16, axs[0, 1].get_position().y0 - 0.16, axs[0, 1].get_position().width, 0.05])
        cbar = fig.colorbar(cs[0, 1], cax=cax, orientation='horizontal', extend=extends[0],
                            ticks=tickss[0])
        cbar.set_label(clabel)
        cbar.ax.minorticks_off()
        cbar.ax.tick_params(labelsize=14)

        cax = fig.add_axes(
            [axs[0, 2].get_position().x0, axs[0, 2].get_position().y0 - 0.16, axs[0, 2].get_position().width, 0.05])
        cbar = fig.colorbar(cs[0, 2], cax=cax, orientation='horizontal', extend=extends[1],
                            ticks=tickss[1])
        cbar.set_label(clabel)
        cbar.ax.minorticks_off()
        cbar.ax.tick_params(labelsize=14)


        axs[0, 0].text(-0.24, 0.5, f'{var}', ha='center', va='center', rotation='vertical',
                       transform=axs[0, 0].transAxes, fontsize=15)
        axs[0, 0].text(-0.2, 1.1, f'{mon}', ha='left', va='center', rotation='horizontal',
                       transform=axs[0, 0].transAxes, fontsize=14)
        # axs[0, 2].text(1.07, 1.09, '[%]', ha='center', va='center', rotation='horizontal',
        #                transform=axs[0, 2].transAxes, fontsize=12)

        plotpath = "/project/pr133/rxiang/figure/paper2/delta/"
        fig.savefig(plotpath + f'{var}' + '_' + f'{mon}_delta1.png', dpi=500)
        fig.show()
