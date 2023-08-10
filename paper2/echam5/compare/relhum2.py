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
import metpy.calc as mpcalc
from metpy.units import units
import matplotlib.colors as colors

font = {'size': 11}
matplotlib.rc('font', **font)

# -------------------------------------------------------------------------------
# define function
def calc(dew, p, t):
    t0 = 273.16
    Rdry = 287.0597
    Rvap = 461.5250
    if dew > 273.15:
        a1, a3, a4 = 611.21, 17.502, 32.19
    else:
        a1, a3, a4 = 611.21, 22.587, -0.7
    e = a1*np.exp(a3*(dew-t0)/(dew-a4))
    if t > 273.15:
        a1, a3, a4 = 611.21, 17.502, 32.19
    else:
        a1, a3, a4 = 611.21, 22.587, -0.7
    es = a1*np.exp(a3*(t-t0)/(t-a4))
    q = (Rdry/Rvap)*e/(p-(1-Rdry/Rvap)*e)*1000
    rh = e/es*100
    return q, rh

# -------------------------------------------------------------------------------
# read data
# %%
sims = ['PI', 'PD', 'LGM', 'PLIO']
mdpath = "/scratch/snx3000/rxiang/echam5"
relhum2 = {}
fname = {'01': 'jan.nc', '07': 'jul.nc'}
labels = {'PI': 'Pre-industrial', 'PD': 'Present day (1970-1995)', 'LGM': 'Last glacial maximum', 'PLIO': 'Mid-Pliocene'}
month = {'01': 'JAN', '07': 'JUL'}

for s in range(len(sims)):
    sim = sims[s]
    relhum2[sim] = {}
    relhum2[sim]['label'] = labels[sim]
    for mon in ['01', '07']:
        relhum2[sim][mon] = {}
        name = fname[mon]
        data = xr.open_dataset(f'{mdpath}/{sim}/analysis/temp2/mon/{name}')
        t2m = data['temp2'].values[0, :, :]
        dew2 = data['dew2'].values[0, :, :]
        data = xr.open_dataset(f'{mdpath}/{sim}/analysis/aps/mon/{name}')
        aps = data['aps'].values[0, :, :]
        q2, rh2 = [], []
        for i in range(240):
            for j in range(480):
                q, rh = calc(dew2[i, j], aps[i, j], t2m[i, j])
                q2.append(q)
                rh2.append(rh)
        q2_ = np.array(q2).reshape(240, 480)
        rh2_ = np.array(rh2).reshape(240, 480)
        relhum2[sim][mon]['relhum2'], relhum2[sim][mon]['q2'] = rh2_, q2_

# %%
lat = xr.open_dataset(f'{mdpath}/LGM/analysis/temp2/mon/{name}')['lat'].values[:]
lon = xr.open_dataset(f'{mdpath}/LGM/analysis/temp2/mon/{name}')['lon'].values[:]
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

cmap1 = cmc.roma
levels1 = np.linspace(0, 100, 20, endpoint=True)
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

cmap2 = drywet(25, cmc.vik_r)
levels2 = np.linspace(-30, 30, 10, endpoint=True)
norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)

for mon in ['01', '07']:
    m = month[mon]
    fig = plt.figure(figsize=(wi, hi))
    left, bottom, right, top = 0.03, 0.01, 0.94, 0.95
    gs = gridspec.GridSpec(nrows=2, ncols=4, left=left, bottom=bottom, right=right, top=top,
                           wspace=0.015, hspace=0.15)

    for i in range(4):
        sim = sims[i]
        label = relhum2[sim]['label']
        axs[0, i] = fig.add_subplot(gs[0, i], projection=ccrs.Robinson(central_longitude=180, globe=None))
        axs[0, i].coastlines(zorder=3)
        axs[0, i].stock_img()
        axs[0, i].gridlines()
        cs[0, i] = axs[0, i].pcolormesh(lon, lat, relhum2[sim][mon]['relhum2'], cmap=cmap1, norm=norm1, shading="auto",
                                        transform=ccrs.PlateCarree())
        # ct[0, i] = axs[0, i].contour(lon, lat, relhum2[sim]['wssms'], levels=np.linspace(2, 14, 5, endpoint=True),
        #                              colors='maroon', linewidths=1, transform=ccrs.PlateCarree())
        # clabel = axs[0, i].clabel(ct[0, i], [2, 5, 8, 11, 14], inline=True, fontsize=13, use_clabeltext=True)
        # for l in clabel:
        #     l.set_rotation(0)
        # [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]
        # label = relhum2[sim]['label']
        axs[0, i].set_title(f'{label}', fontweight='bold', pad=6, fontsize=13, loc='center')

    for i in range(3):
        sim = sims[i + 1]
        axs[1, i + 1] = fig.add_subplot(gs[1, i + 1], projection=ccrs.Robinson(central_longitude=180, globe=None))
        axs[1, i + 1].coastlines(zorder=3)
        axs[1, i + 1].stock_img()
        axs[1, i + 1].gridlines()
        cs[1, i + 1] = axs[1, i + 1].pcolormesh(lon, lat, relhum2[sim][mon]['relhum2'] - relhum2['PI'][mon]['relhum2'], cmap=cmap2,
                                                clim=(-30, 30), shading="auto", transform=ccrs.PlateCarree())

    cax = fig.add_axes(
        [axs[0, 3].get_position().x1 + 0.01, axs[0, 3].get_position().y0, 0.01, axs[0, 3].get_position().height])
    cbar = fig.colorbar(cs[0, 1], cax=cax, orientation='vertical', extend='max', ticks=np.linspace(0, 100, 6, endpoint=True))
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.set_xlabel('$^{o}$C', fontsize=13)

    cax = fig.add_axes(
        [axs[1, 3].get_position().x1 + 0.01, axs[1, 3].get_position().y0, 0.01, axs[1, 3].get_position().height])
    cbar = fig.colorbar(cs[1, 1], cax=cax, orientation='vertical', extend='both',
                        ticks=np.linspace(-30, 30, 5, endpoint=True))
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.set_xlabel('$^{o}$C', fontsize=13)

    axs[0, 0].text(-0.08, 0.5, '2m RH', ha='center', va='center', rotation='vertical',
                   transform=axs[0, 0].transAxes, fontsize=13, fontweight='bold')
    axs[1, 1].text(-0.08, 0.5, 'Anomalies', ha='center', va='center', rotation='vertical',
                   transform=axs[1, 1].transAxes, fontsize=13, fontweight='bold')
    axs[0, 3].text(1.07, 1.06, '[%]', ha='center', va='center', rotation='horizontal',
                   transform=axs[0, 3].transAxes, fontsize=12)
    axs[0, 0].text(-0.09, 1.06, f'{m}', ha='left', va='center', rotation='horizontal',
                   transform=axs[0, 0].transAxes, fontsize=13, fontweight='bold')

    fig.show()
    plotpath = "/project/pr133/rxiang/figure/echam5/"
    fig.savefig(plotpath + 'relhum2' + f'{mon}.png', dpi=500)
    plt.close(fig)






