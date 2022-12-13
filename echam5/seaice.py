# -------------------------------------------------------------------------------
# modules
#
from netCDF4 import Dataset
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotcosmomap import plotcosmo, pole, colorbar
import cartopy.crs as ccrs
from numpy import inf
import matplotlib.gridspec as gridspec
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mycolor import custom_div_cmap, cbr_wet, cbr_drywet, drywet, custom_seq_cmap_, wind
from pyproj import Transformer
import scipy.ndimage as ndimage
import matplotlib

font = {'size': 11}
matplotlib.rc('font', **font)

# -------------------------------------------------------------------------------
# read data
# %%
sims = ['PI', 'PD', 'LGM', 'PLIO']
mdpath = "/scratch/snx3000/rxiang/echam5"
friac = {}
fname = {'01': 'jan.nc', '07': 'jul.nc'}
labels = {'PI': 'Pre-industrial', 'PD': 'Present day (1970-1995)', 'LGM': 'Last glacial maximum', 'PLIO': 'Mid-Pliocene'}
month = {'01': 'JAN', '07': 'JUL'}

for s in range(len(sims)):
    sim = sims[s]
    friac[sim] = {}
    friac[sim]['label'] = labels[sim]
    for mon in ['01', '07']:
        friac[sim][mon] = {}
        name = fname[mon]
        data = xr.open_dataset(f'{mdpath}/{sim}/analysis/friac/mon/{name}')
        precip = data['friac'].values[0, :, :] * 100
        friac[sim][mon]['friac'] = precip
# %%
lat = xr.open_dataset(f'{mdpath}/LGM/analysis/friac/mon/{name}')['lat'].values[:]
lon = xr.open_dataset(f'{mdpath}/LGM/analysis/friac/mon/{name}')['lon'].values[:]
lat_, lon_ = np.meshgrid(lon, lat)
# -------------------------------------------------------------------------------
# plot
# %%
ar = 1.0  # initial aspect ratio for first trial
wi = 16  # height in inches #15
hi = 4.3  # width in inches #10
ncol = 4  # edit here
nrow = 2
axs, cs, ct, topo, q, qk, topo1 = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), np.empty(
    shape=(nrow, ncol), dtype='object')

cmap1 = cmc.davos_r
levels1 = np.linspace(0, 100, 20, endpoint=True)
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

cmap2 = drywet(25, cmc.vik_r)
levels2 = np.linspace(0, 40, 11, endpoint=True)
norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)

for mon in ['01', '07']:
    m = month[mon]
    fig = plt.figure(figsize=(wi, hi))
    left, bottom, right, top = 0.03, 0.01, 0.94, 0.95
    gs = gridspec.GridSpec(nrows=2, ncols=4, left=left, bottom=bottom, right=right, top=top,
                           wspace=0.015, hspace=0.15)

    for i in range(4):
        sim = sims[i]
        label = friac[sim]['label']
        axs[0, i] = fig.add_subplot(gs[0, i], projection=ccrs.Robinson(central_longitude=180, globe=None))
        axs[0, i].coastlines(zorder=3)
        axs[0, i].stock_img()
        axs[0, i].gridlines()
        cs[0, i] = axs[0, i].pcolormesh(lon, lat, friac[sim][mon]['friac'], cmap=cmap1, norm=norm1, shading="auto",
                                        transform=ccrs.PlateCarree())
        # ct[0, i] = axs[0, i].contour(lon, lat, friac[sim]['wssms'], levels=np.linspace(2, 14, 5, endpoint=True),
        #                              colors='maroon', linewidths=1, transform=ccrs.PlateCarree())
        # clabel = axs[0, i].clabel(ct[0, i], [2, 5, 8, 11, 14], inline=True, fontsize=13, use_clabeltext=True)
        # for l in clabel:
        #     l.set_rotation(0)
        # [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]   
        # label = friac[sim]['label']
        axs[0, i].set_title(f'{label}', fontweight='bold', pad=6, fontsize=13, loc='center')

    for i in range(3):
        sim = sims[i + 1]
        axs[1, i + 1] = fig.add_subplot(gs[1, i + 1], projection=ccrs.Robinson(central_longitude=180, globe=None))
        axs[1, i + 1].coastlines(zorder=3)
        axs[1, i + 1].stock_img()
        axs[1, i + 1].gridlines()
        cs[1, i + 1] = axs[1, i + 1].pcolormesh(lon, lat, friac[sim][mon]['friac'] - friac['PI'][mon]['friac'], cmap=cmap2,
                                                clim=(-50, 50), shading="auto", transform=ccrs.PlateCarree())

    cax = fig.add_axes(
        [axs[0, 3].get_position().x1 + 0.01, axs[0, 3].get_position().y0, 0.01, axs[0, 3].get_position().height])
    cbar = fig.colorbar(cs[0, 1], cax=cax, orientation='vertical',
                        ticks=np.linspace(0, 100, 6, endpoint=True))
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.set_xlabel('mm day$^{-1}$', fontsize=13)

    cax = fig.add_axes(
        [axs[1, 3].get_position().x1 + 0.01, axs[1, 3].get_position().y0, 0.01, axs[1, 3].get_position().height])
    cbar = fig.colorbar(cs[1, 1], cax=cax, orientation='vertical', extend='both',
                        ticks=np.linspace(-50, 50, 5, endpoint=True))
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.set_xlabel('mm day$^{-1}$', fontsize=13)

    axs[0, 0].text(-0.08, 0.5, 'Sea ice', ha='center', va='center', rotation='vertical',
                   transform=axs[0, 0].transAxes, fontsize=13, fontweight='bold')
    axs[1, 1].text(-0.08, 0.5, 'Anomalies', ha='center', va='center', rotation='vertical',
                   transform=axs[1, 1].transAxes, fontsize=13, fontweight='bold')
    axs[0, 3].text(1.07, 1.06, '[%]', ha='center', va='center', rotation='horizontal',
                   transform=axs[0, 3].transAxes, fontsize=12)
    axs[0, 0].text(-0.09, 1.06, f'{m}', ha='left', va='center', rotation='horizontal',
                   transform=axs[0, 0].transAxes, fontsize=13, fontweight='bold')

    fig.show()
    plotpath = "/project/pr133/rxiang/figure/echam5/"
    fig.savefig(plotpath + 'friac' + f'{mon}.png', dpi=500)
    plt.close(fig)






