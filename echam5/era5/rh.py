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
def Tetens_formula(temp, matter="water"):
    """Source: IFS Documentation, Cy41r2, Part IV: Physical Processes,
       equation 7.5
    """
    a_1 = 611.21  # Pa
    if matter == "water":
        a_3 = 17.502  # Pa
        a_4 = 32.19  # K
    elif matter == "ice":
        a_3 = 22.587  # Pa
        a_4 = -0.7  # K
    else:
        raise ValueError("Value for 'matter' unknown")
    temp_0 = 273.16  # K
    e_sat = a_1 * np.exp(a_3 * (temp - temp_0) / (temp - a_4))
    return e_sat
# ----------------------------------------------------------------------------
def mixed_phase(temp):
    """Source: IFS Documentation, Cy41r2, Part IV: Physical Processes,
       equation 7.6
    """
    temp_0 = 273.16  # [K]
    temp_ice = 250.16  # [K]
    alpha = np.empty_like(temp)
    alpha[temp <= temp_ice] = 0.0
    mask = (temp > temp_ice) & (temp < temp_0)
    alpha[mask] = ((temp[mask] - temp_ice) / (temp_0 - temp_ice)) ** 2
    alpha[temp >= temp_0] = 1.0
    return alpha
# ----------------------------------------------------------------------------
def sat_specific_hum(temp, press, method="water"):
    """Source: IFS Documentation, Cy41r2, Part IV: Physical Processes,
       equation 7.4
    """
    epsilon = 0.621981  # R_dry / R_vap
    if method == "water":
        q_sat = (epsilon * Tetens_formula(temp, matter="water")) \
                / (press - (1.0 - epsilon) * Tetens_formula(temp, matter="water"))
    elif method == "echam":
        e_sat = Tetens_formula(temp, matter="water")
        mask = (temp < 273.15)
        e_sat[mask] = Tetens_formula(temp[mask], matter="ice")
        q_sat = (epsilon * e_sat) \
                / (press - (1.0 - epsilon) * e_sat)
    elif method == "ifs":
        e_sat_water = Tetens_formula(temp, matter="water")
        e_sat_ice = Tetens_formula(temp, matter="ice")
        alpha = mixed_phase(temp)
        e_sat = (alpha * e_sat_water) + (1.0 - alpha) * e_sat_ice
        q_sat = (epsilon * e_sat) \
                / (press - (1.0 - epsilon) * e_sat)
    else:
        raise ValueError("Value for 'method' unknown")
    return q_sat
# ----------------------------------------------------------------------------
def rel_humidity(d2m, t2m, method="water"):
    """Source: IFS Documentation, Cy41r2, Part IV: Physical Processes,
       equation 7.89
    """
    epsilon = 0.621981  # R_dry / R_vap
    if method == "water":
        e_sat = Tetens_formula(t2m, matter="water")
        e = Tetens_formula(d2m, matter="water")
        rel_hum = e / e_sat * 100
        # rel_hum = (press * q_s * (1.0 / epsilon)) \
        #     / (e_sat * (1.0 + q_s * (1.0 / epsilon - 1.0)))
    elif method == "echam":
        e_sat = Tetens_formula(t2m, matter="water")
        mask = (t2m < 273.15)
        e_sat[mask] = Tetens_formula(t2m[mask], matter="ice")
        e = Tetens_formula(d2m, matter="water")
        mask = (d2m < 273.15)
        e[mask] = Tetens_formula(d2m[mask], matter="ice")
        rel_hum = e / e_sat * 100
        # rel_hum = (press * q_s * (1.0 / epsilon)) \
        #     / (e_sat * (1.0 + q_s * (1.0 / epsilon - 1.0)))
    elif method == "ifs":
        e_sat_water = Tetens_formula(t2m, matter="water")
        e_sat_ice = Tetens_formula(t2m, matter="ice")
        alpha = mixed_phase(t2m)
        e_sat = (alpha * e_sat_water) + (1.0 - alpha) * e_sat_ice
        e_water = Tetens_formula(d2m, matter="water")
        e_ice = Tetens_formula(d2m, matter="ice")
        alpha = mixed_phase(d2m)
        e = (alpha * e_water) + (1.0 - alpha) * e_ice
        rel_hum = e / e_sat * 100
        # rel_hum = (press * q_s * (1.0 / epsilon)) \
        #     / (e_sat * (1.0 + q_s * (1.0 / epsilon - 1.0)))
    else:
        raise ValueError("Value for 'method' unknown")
    return rel_hum

# read data
# %%
sims = ['PD', 'ERA5']
mdpath = "/scratch/snx3000/rxiang/echam5"
erapath = "/project/pr133/rxiang/data/era5/ot/remap"
hu = {}
fname = {'01': 'jan.nc', '07': 'jul.nc'}
labels = {'PD': 'ECHAM5', 'ERA5': 'ERA5'}
month = {'01': 'JAN', '07': 'JUL'}

hu['PD'], hu['ERA5'] = {}, {}
hu['PD']['label'] = labels['PD']
hu['ERA5']['label'] = labels['ERA5']
for mon in ['01', '07']:
    hu['PD'][mon] = {}
    name = fname[mon]
    data = xr.open_dataset(f'{mdpath}/PD/analysis/temp2/mon/{name}')
    t2m = data['temp2'].values[0, :, :]
    dew2 = data['dew2'].values[0, :, :]
    data = xr.open_dataset(f'{mdpath}/PD/analysis/aps/mon/{name}')
    aps = data['aps'].values[0, :, :]

    q_s = sat_specific_hum(dew2, aps, "echam")
    rel_hum = rel_humidity(dew2, t2m, "echam")
    # q2, rh2 = [], []
    # for i in range(240):
    #     for j in range(480):
    #         q, rh = calc(dew2[i, j], aps[i, j], t2m[i, j])
    #         q2.append(q)
    #         rh2.append(rh)
    # q2_ = np.array(q2).reshape(240, 480)
    # rh2_ = np.array(rh2).reshape(240, 480)
    hu['PD'][mon]['q2'], hu['PD'][mon]['rh2'] = q_s, rel_hum

#%%
data = xr.open_dataset(f'{erapath}/era5.mo.1970-1995.jan.remap.echam5.nc')
hu['ERA5']['01'], hu['ERA5']['07'] = {}, {}
q_s = sat_specific_hum(data['d2m'].values[:, :, :], data['sp'].values[:, :, :], "echam")
rel_hum = rel_humidity(data['d2m'].values[:, :, :], data['t2m'].values[:, :, :], "echam")
hu['ERA5']['01']['q2'], hu['ERA5']['01']['rh2'] = np.nanmean(q_s, axis=0), np.nanmean(rel_hum, axis=0)
# q2, rh2 = [], []
# for i in range(26):
#     for j in range(240):
#         for k in range(480):
#             q, rh = calc(data['d2m'].values[i, j, k], data['sp'].values[i, j, k], data['t2m'].values[i, j, k])
#             q2.append(q)
#             rh2.append(rh)
# q2_ = np.nanmean(np.array(q2).reshape(26, 240, 480), axis=0)
# rh2_ = np.nanmean(np.array(q2).reshape(26, 240, 480), axis=0)


data = xr.open_dataset(f'{erapath}/era5.mo.1970-1995.jul.remap.echam5.nc')
q_s = sat_specific_hum(data['d2m'].values[:, :, :], data['sp'].values[:, :, :], "echam")
rel_hum = rel_humidity(data['d2m'].values[:, :, :], data['t2m'].values[:, :, :], "echam")
hu['ERA5']['07']['q2'], hu['ERA5']['07']['rh2'] = np.nanmean(q_s, axis=0), np.nanmean(rel_hum, axis=0)
# q2, rh2 = [], []
# for i in range(26):
#     for j in range(240):
#         for k in range(480):
#             q, rh = calc(data['d2m'].values[i, j, k], data['sp'].values[i, j, k], data['t2m'].values[i, j, k])
#             q2.append(q)
#             rh2.append(rh)
# q2_ = np.nanmean(np.array(q2).reshape(26, 240, 480), axis=0)
# rh2_ = np.nanmean(np.array(q2).reshape(26, 240, 480), axis=0)
# hu['ERA5']['07']['q2'], hu['ERA5']['07']['rh2'] = q2_, rh2_

# %%
lat = xr.open_dataset(f'{mdpath}/LGM/analysis/temp2/mon/{name}')['lat'].values[:]
lon = xr.open_dataset(f'{mdpath}/LGM/analysis/temp2/mon/{name}')['lon'].values[:]
lat_, lon_ = np.meshgrid(lon, lat)

# -------------------------------------------------------------------------------
# plot
# %%
ar = 1.0  # initial aspect ratio for first trial
wi = 8.7  # height in inches #15
hi = 4.3  # width in inches #10
ncol = 2  # edit here
nrow = 2
axs, cs, ct, topo, q, qk, topo1 = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), \
                                  np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol),
                                                                                         dtype='object'), np.empty(
    shape=(nrow, ncol), dtype='object')

cmap1 = plt.cm.get_cmap("Spectral").copy()
cmap1.set_extremes(over='black')
levels1 = np.linspace(0, 100, 21, endpoint=True)
bounds = np.linspace(0, 100, 21, endpoint=True)
norm1 = matplotlib.colors.BoundaryNorm(bounds, cmap1.N)
# norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

cmap2 = drywet(25, cmc.vik_r)
levels2 = np.linspace(-30, 30, 10, endpoint=True)
norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)

for mon in ['01', '07']:
    m = month[mon]
    fig = plt.figure(figsize=(wi, hi))
    left, bottom, right, top = 0.05, 0.01, 0.9, 0.94
    gs = gridspec.GridSpec(nrows=2, ncols=2, left=left, bottom=bottom, right=right, top=top,
                           wspace=0.01, hspace=0.15)

    for i in range(2):
        sim = sims[i]
        label = hu[sim]['label']
        axs[0, i] = fig.add_subplot(gs[0, i], projection=ccrs.Robinson(central_longitude=180, globe=None))
        axs[0, i].coastlines(zorder=3)
        axs[0, i].stock_img()
        axs[0, i].gridlines()
        cs[0, i] = axs[0, i].pcolormesh(lon, lat, hu[sim][mon]['rh2'], cmap=cmap1, norm=norm1, shading="auto",
                                        transform=ccrs.PlateCarree())
        axs[0, i].set_title(f'{label}', fontweight='bold', pad=6, fontsize=13, loc='center')

    for i in range(1):
        sim = sims[i + 1]
        axs[1, i + 1] = fig.add_subplot(gs[1, i + 1], projection=ccrs.Robinson(central_longitude=180, globe=None))
        axs[1, i + 1].coastlines(zorder=3)
        axs[1, i + 1].stock_img()
        axs[1, i + 1].gridlines()
        cs[1, i + 1] = axs[1, i + 1].pcolormesh(lon, lat, hu[sim][mon]['rh2'] - hu['PD'][mon]['rh2'], cmap=cmap2,
                                                clim=(-30,  30), shading="auto", transform=ccrs.PlateCarree())

    cax = fig.add_axes(
        [axs[0, 1].get_position().x1 + 0.01, axs[0, 1].get_position().y0, 0.02, axs[0, 1].get_position().height])
    cbar = fig.colorbar(cs[0, 1], cax=cax, orientation='vertical', extend='max', ticks=np.linspace(0, 100, 6, endpoint=True))
    cbar.ax.tick_params(labelsize=13)

    cax = fig.add_axes(
        [axs[1, 1].get_position().x1 + 0.01, axs[1, 1].get_position().y0, 0.02, axs[1, 1].get_position().height])
    cbar = fig.colorbar(cs[1, 1], cax=cax, orientation='vertical', extend='both',
                        ticks=np.linspace(-30, 30, 5, endpoint=True))
    cbar.ax.tick_params(labelsize=13)

    axs[0, 0].text(-0.05, 0.5, '2m RH', ha='center', va='center', rotation='vertical',
                   transform=axs[0, 0].transAxes, fontsize=13, fontweight='bold')
    axs[1, 1].text(-0.05, 0.5, 'Differences', ha='center', va='center', rotation='vertical',
                   transform=axs[1, 1].transAxes, fontsize=13, fontweight='bold')
    axs[0, 1].text(1.05, 1.08, '[%]', ha='center', va='center', rotation='horizontal',
                   transform=axs[0, 1].transAxes, fontsize=12)
    axs[0, 0].text(-0.09, 1.075, f'PD-{m}', ha='left', va='center', rotation='horizontal',
                   transform=axs[0, 0].transAxes, fontsize=13, fontweight='bold')

    fig.show()
    plotpath = "/project/pr133/rxiang/figure/echam5/"
    fig.savefig(plotpath + 'rh2.' + f'{mon}.era5.png', dpi=500)
    plt.close(fig)

#%%
cmap1 = plt.cm.get_cmap("Spectral")
levels1 = np.linspace(0, 20, 20, endpoint=True)
norm1 = matplotlib.colors.BoundaryNorm(levels1, cmap1.N)

cmap2 = drywet(25, cmc.vik_r)
levels2 = np.linspace(-5,  5, 10, endpoint=True)
norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)

for mon in ['01', '07']:
    m = month[mon]
    fig = plt.figure(figsize=(wi, hi))
    left, bottom, right, top = 0.05, 0.01, 0.9, 0.94
    gs = gridspec.GridSpec(nrows=2, ncols=2, left=left, bottom=bottom, right=right, top=top,
                           wspace=0.01, hspace=0.15)

    for i in range(2):
        sim = sims[i]
        label = hu[sim]['label']
        axs[0, i] = fig.add_subplot(gs[0, i], projection=ccrs.Robinson(central_longitude=180, globe=None))
        axs[0, i].coastlines(zorder=3)
        axs[0, i].stock_img()
        axs[0, i].gridlines()
        cs[0, i] = axs[0, i].pcolormesh(lon, lat, hu[sim][mon]['q2']*1000, cmap=cmap1, norm=norm1, shading="auto",
                                        transform=ccrs.PlateCarree())
        axs[0, i].set_title(f'{label}', fontweight='bold', pad=6, fontsize=13, loc='center')

    for i in range(1):
        sim = sims[i + 1]
        axs[1, i + 1] = fig.add_subplot(gs[1, i + 1], projection=ccrs.Robinson(central_longitude=180, globe=None))
        axs[1, i + 1].coastlines(zorder=3)
        axs[1, i + 1].stock_img()
        axs[1, i + 1].gridlines()
        cs[1, i + 1] = axs[1, i + 1].pcolormesh(lon, lat, hu[sim][mon]['q2']*1000 - hu['PD'][mon]['q2']*1000, cmap=cmap2,
                                                clim=(-2,  2), shading="auto", transform=ccrs.PlateCarree())

    cax = fig.add_axes(
        [axs[0, 1].get_position().x1 + 0.01, axs[0, 1].get_position().y0, 0.02, axs[0, 1].get_position().height])
    cbar = fig.colorbar(cs[0, 1], cax=cax, orientation='vertical', extend='max', ticks=np.linspace(0, 20, 5, endpoint=True))
    cbar.ax.tick_params(labelsize=13)

    cax = fig.add_axes(
        [axs[1, 1].get_position().x1 + 0.01, axs[1, 1].get_position().y0, 0.02, axs[1, 1].get_position().height])
    cbar = fig.colorbar(cs[1, 1], cax=cax, orientation='vertical', extend='both',
                        ticks=np.linspace(-2, 2, 5, endpoint=True))
    cbar.ax.tick_params(labelsize=13)

    axs[0, 0].text(-0.05, 0.5, '2m q$_v$', ha='center', va='center', rotation='vertical',
                   transform=axs[0, 0].transAxes, fontsize=13, fontweight='bold')
    axs[1, 1].text(-0.05, 0.5, 'Differences', ha='center', va='center', rotation='vertical',
                   transform=axs[1, 1].transAxes, fontsize=13, fontweight='bold')
    axs[0, 1].text(1.05, 1.08, '[g kg$^{-1}$]', ha='center', va='center', rotation='horizontal',
                   transform=axs[0, 1].transAxes, fontsize=12)
    axs[0, 0].text(-0.09, 1.075, f'PD-{m}', ha='left', va='center', rotation='horizontal',
                   transform=axs[0, 0].transAxes, fontsize=13, fontweight='bold')

    fig.show()
    plotpath = "/project/pr133/rxiang/figure/echam5/"
    fig.savefig(plotpath + 'q2.' + f'{mon}.era5.png', dpi=500)
    plt.close(fig)




