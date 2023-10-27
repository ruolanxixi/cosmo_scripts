# fsnow = MAX[0.01; MIN(1, Wsnow/0.015]

# -------------------------------------------------------------------------------
# modules
#
import xarray as xr
from pyproj import CRS, Transformer
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib
import cmcrameri.cm as cmc
import numpy.ma as ma
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mycolor import custom_white_cmap
from plotcosmomap import plotcosmo_notick, pole, plotcosmo04_notick, plotcosmo_notick_lgm, plotcosmo04_notick_lgm
import matplotlib.colors as colors
import scipy.ndimage as ndimage
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from metpy.units import units
import metpy.calc as mpcalc

def scf_echam5(h_sn, sigma_z, epsilon=1e-08):
    """
    h_sn:    snow depth (snow water equivalent) [m]
    sigma_z: subgrid-scale standard deviation of height [m]
    epsilon: small number to avoid division by zero [-]
    scf:     snow cover fraction [-]
    """
    gamma_1 = 0.95
    gamma_2 = 0.15
    scf = gamma_1 * np.tanh(100.0 * h_sn) \
        * np.sqrt((1000.0 * h_sn)
                  / (1000.0 * h_sn + gamma_2 * sigma_z + epsilon))
    return scf

###############################################################################
#%% Data
###############################################################################
path_cosmo = "/project/pr133/rxiang/data/cosmo/"
path_echam5 = "/project/pr133/rxiang/data/echam5_raw/"
path_temp = "/project/pr133/rxiang/data/BECCY_snow/"
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()

def compute_fsnow(W_SNOW):
    # Apply the formula element-wise
    fsnow = np.maximum(0.01, np.minimum(1, W_SNOW / 0.015))
    return fsnow

data_scd = {}

# for i in ("COSMO CTRL", "COSMO PGW"):
#     if i == "COSMO CTRL":
#         ds = xr.open_dataset(f'{path}' + 'EAS11_ctrl/ydaymean/W_SNOW/' + '01-05.W_SNOW.nc')
#         snow = ds['W_SNOW'].values[...]
#         fsnow = compute_fsnow(snow)
#         nsnow = np.sum(fsnow > 0.5, axis=0)
#         nsnow = np.ma.masked_where(nsnow < 1, nsnow)
#     else:
#         ds = xr.open_dataset(f'{path}' + 'EAS11_lgm/ydaymean/W_SNOW/' + '01-05.W_SNOW.nc')
#         snow = ds['W_SNOW'].values[...]
#         fsnow = compute_fsnow(snow)
#         nsnow = np.sum(fsnow > 0.5, axis=0)
#         nsnow = np.ma.masked_where(nsnow < 1, nsnow)
#     data_scd[i] = {"scd": nsnow,
#                    "x": ds["rlon"].values, "y": ds["rlat"].values,
#                    "crs": rot_pole_crs}

cosmo_run = {"COSMO CTRL": path_cosmo + "EAS11_ctrl/24h/W_SNOW/0?_W_SNOW.nc",
             "COSMO PGW": path_cosmo + "EAS11_lgm/24h/W_SNOW/0?_W_SNOW.nc"}
for i in cosmo_run.keys():
    ds = xr.open_mfdataset(cosmo_run[i])
    # ds = ds.isel(time=slice(0, -1))
    print(ds["time"][0].values, ds["time"][-1].values)
    _, index = np.unique(ds["time"], return_index=True)
    ds = ds.isel(time=index)
    print(np.all(np.diff(ds["time"]).astype("timedelta64[D]")
                 == np.timedelta64(1, "D")))
    ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))
    print(len(ds["time"]) / 5)
    # -------------------------------------------------------------------------
    delta_s = 0.015  # [m]
    scf = np.maximum(0.01, np.minimum(1.0, ds["W_SNOW"] / delta_s))
    # Source: COSMO documentation (version 6.0) - Part 2 - p. 121
    sc = (scf >= 0.5).astype(np.float32)  # binary snow cover [-]
    # -------------------------------------------------------------------------
    crs_cosmo = ccrs.RotatedPole(
        pole_latitude=ds["rotated_pole"].grid_north_pole_latitude,
        pole_longitude=ds["rotated_pole"].grid_north_pole_longitude)
    data_scd[i] = {"scd": np.ma.masked_where(sc.sum(axis=0) / 5.0 < 1, sc.sum(axis=0) / 5.0),
                   "x": ds["rlon"].values, "y": ds["rlat"].values,
                   "crs": crs_cosmo}
    ds.close()

data_scd['PGW-CTRL'] = {"scd": data_scd['COSMO PGW']['scd'] - data_scd['COSMO CTRL']['scd'],
                        "x": ds["rlon"].values, "y": ds["rlat"].values,
                        "crs": rot_pole_crs}

for i in ("ECHAM5 PI", "ECHAM5 LGM"):
    if i == "ECHAM5 PI":
        ds = xr.open_mfdataset(path_echam5 + "PI/input/T159_jan_surf.nc")
        oro_std = ds["OROSTD"].values  # [m]
        ds.close()
        ds = xr.open_dataset(path_temp + "ECHAM5_PI_sn_5years.nc")
        ds = ds.sel(time=~((ds["time.month"] == 2) & (ds["time.day"] == 29)))
        print(len(ds["time"]) / 5)
        sn = ds["sn"].values  # [m]
        ds.close()
    else:
        ds = xr.open_mfdataset(path_echam5
                               + "LGM/input/T159_jan_surf.lgm.veg.nc")
        oro_std = ds["OROSTD"].values  # [m]
        ds.close()
        ds = xr.open_dataset(path_temp + "ECHAM5_LGM_sn_5years.nc")
        ds = ds.sel(time=~((ds["time.month"] == 2) & (ds["time.day"] == 29)))
        print(len(ds["time"]) / 5)
        sn = ds["sn"].values  # [m]
        ds.close()
    sc = (scf_echam5(sn, oro_std) > 0.5).astype(np.float32)
    # binary snow cover [-]
    data_scd[i] = {"scd": np.ma.masked_where(sc.sum(axis=0) / 5.0 < 1, sc.sum(axis=0) / 5.0),
                   "x": ds["lon"].values, "y": ds["lat"].values,
                   "crs": ccrs.PlateCarree()}
    ds.close()

data_scd['LGM-PI'] = {"scd": data_scd['ECHAM5 LGM']['scd'] - data_scd['ECHAM5 PI']['scd'],
                      "x": ds["lon"].values, "y": ds["lat"].values,
                      "crs": ccrs.PlateCarree()}
###############################################################################
# %% Plot
###############################################################################
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()
fig = plt.figure(figsize=(11, 4))
gs1 = gridspec.GridSpec(2, 2, left=0.075, bottom=0.03, right=0.598,
                        top=0.96, hspace=0.02, wspace=0.05,
                        width_ratios=[1, 1], height_ratios=[1, 1])
gs2 = gridspec.GridSpec(2, 1, left=0.675, bottom=0.03, right=0.93,
                        top=0.96, hspace=0.02, wspace=0.05, height_ratios=[1, 1])
ncol = 3  # edit here
nrow = 2

axs, cs, ct, topo, q = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                       np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                       np.empty(shape=(nrow, ncol), dtype='object')

for i in range(nrow):
    axs[i, 0] = fig.add_subplot(gs1[i, 0], projection=rot_pole_crs)
    axs[i, 0] = plotcosmo_notick(axs[i, 0])
    axs[i, 1] = fig.add_subplot(gs1[i, 1], projection=rot_pole_crs)
    axs[i, 1] = plotcosmo_notick_lgm(axs[i, 1], diff=False)
    axs[i, 2] = fig.add_subplot(gs2[i, 0], projection=rot_pole_crs)
    axs[i, 2] = plotcosmo_notick_lgm(axs[i, 2], diff=False)

levels1 = MaxNLocator(nbins=100).tick_values(0, 250)
cmap1 = cmc.roma_r
# cmap1 = custom_white_cmap(100, cmc.roma_r)
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
levels2 = MaxNLocator(nbins=50).tick_values(0, 100)
# cmap2 = drywet(20, cmc.vik_r)
cmap2 = cmc.davos_r
norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)
# norm2 = colors.TwoSlopeNorm(vmin=-100., vcenter=0., vmax=100.)
# --
levels = [levels1, levels1, levels2]
norms = [norm1, norm1, norm2]
cmaps = [cmap1, cmap1, cmap2]
# --
sims = ["COSMO CTRL", "COSMO PGW", 'PGW-CTRL']
for j in range(ncol):
    sim = sims[j]
    cmap = cmaps[j]
    norm = norms[j]
    cs[0, j] = axs[0, j].pcolormesh(data_scd[sim]['x'], data_scd[sim]['y'], data_scd[sim]['scd'],
                                    cmap=cmap, norm=norm, shading="auto", transform=data_scd[sim]['crs'])
sims = ["ECHAM5 PI", "ECHAM5 LGM", 'LGM-PI']
for j in range(ncol):
    sim = sims[j]
    cmap = cmaps[j]
    norm = norms[j]
    cs[1, j] = axs[1, j].pcolormesh(data_scd[sim]['x'], data_scd[sim]['y'], data_scd[sim]['scd'],
                                    cmap=cmap, norm=norm, shading="auto", transform=data_scd[sim]['crs'])

for i in range(nrow):
    cax = fig.add_axes(
        [axs[i, 1].get_position().x1 + 0.01, axs[i, 1].get_position().y0, 0.015, axs[i, 1].get_position().height])
    cbar = fig.colorbar(cs[i, 1], cax=cax, orientation='vertical', extend='max', ticks=[0, 50, 100, 150, 200, 250])
    cbar.ax.minorticks_off()
    cbar.ax.tick_params(labelsize=13)
    cax = fig.add_axes(
        [axs[i, 2].get_position().x1 + 0.01, axs[i, 2].get_position().y0, 0.015, axs[i, 2].get_position().height])
    cbar = fig.colorbar(cs[i, 2], cax=cax, orientation='vertical', extend='max', ticks=[0, 20, 40, 60, 80, 100])
    cbar.ax.minorticks_off()
    cbar.ax.tick_params(labelsize=13)
# --
labels = ['CTRL | PI', 'PGW | LGM', 'PGW | LGM - CTRL | PI']
for j in range(ncol):
    label = labels[j]
    axs[0, j].set_title(f'{label}', pad=7, fontsize=14, loc='center')
# --
lefts = ['COSMO', 'ECHAM5']
for i in range(nrow):
    left = lefts[i]
    axs[i, 0].text(-0.2, 0.5, f'{left}', ha='right', va='center',
                   transform=axs[i, 0].transAxes, fontsize=14, rotation=90)
# --
for i in range(nrow):
    axs[i, 0].text(-0.008, 0.95, '50°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.77, '40°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.59, '30°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.41, '20°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.23, '10°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.05, '0°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)

for j in range(ncol):
    axs[1, j].text(0.12, -0.02, '80°E', ha='center', va='top', transform=axs[1, j].transAxes, fontsize=13)
    axs[1, j].text(0.32, -0.02, '100°E', ha='center', va='top', transform=axs[1, j].transAxes, fontsize=13)
    axs[1, j].text(0.52, -0.02, '120°E', ha='center', va='top', transform=axs[1, j].transAxes, fontsize=13)
    axs[1, j].text(0.72, -0.02, '140°E', ha='center', va='top', transform=axs[1, j].transAxes, fontsize=13)
    axs[1, j].text(0.92, -0.02, '160°E', ha='center', va='top', transform=axs[1, j].transAxes, fontsize=13)

plotpath = "/project/pr133/rxiang/figure/paper2/results/lgm/"
fig.savefig(plotpath + 'nsnow_large.png', dpi=500, transparent='True')
plt.show()
plt.close()

# %%
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()
fig = plt.figure(figsize=(11, 3.4))
gs1 = gridspec.GridSpec(2, 2, left=0.075, bottom=0.035, right=0.598,
                        top=0.94, hspace=0.02, wspace=0.05,
                        width_ratios=[1, 1], height_ratios=[1, 1])
gs2 = gridspec.GridSpec(2, 1, left=0.675, bottom=0.035, right=0.93,
                        top=0.94, hspace=0.02, wspace=0.05, height_ratios=[1, 1])
ncol = 3  # edit here
nrow = 2

index = [['a', 'b', 'c'], ['d', 'e', 'f']]

axs, cs, ct, topo, q = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                       np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                       np.empty(shape=(nrow, ncol), dtype='object')

for i in range(nrow):
    axs[i, 0] = fig.add_subplot(gs1[i, 0], projection=rot_pole_crs)
    axs[i, 0] = plotcosmo_notick(axs[i, 0])
    axs[i, 0].set_extent([-52.0, 34.0, -2.0, 40.0], crs=rot_pole_crs)
    axs[i, 1] = fig.add_subplot(gs1[i, 1], projection=rot_pole_crs)
    axs[i, 1] = plotcosmo_notick_lgm(axs[i, 1], diff=False)
    axs[i, 1].set_extent([-52.0, 34.0, -2.0, 40.0], crs=rot_pole_crs)
    axs[i, 2] = fig.add_subplot(gs2[i, 0], projection=rot_pole_crs)
    axs[i, 2] = plotcosmo_notick_lgm(axs[i, 2], diff=False)
    axs[i, 2].set_extent([-52.0, 34.0, -2.0, 40.0], crs=rot_pole_crs)

levels1 = MaxNLocator(nbins=100).tick_values(0, 250)
cmap1 = cmc.roma_r
# cmap1 = custom_white_cmap(100, cmc.roma_r)
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
levels2 = MaxNLocator(nbins=50).tick_values(0, 100)
# cmap2 = drywet(20, cmc.vik_r)
cmap2 = cmc.davos_r
norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)
# norm2 = colors.TwoSlopeNorm(vmin=-100., vcenter=0., vmax=100.)
# --
levels = [levels1, levels1, levels2]
norms = [norm1, norm1, norm2]
cmaps = [cmap1, cmap1, cmap2]
# --
sims = ["COSMO CTRL", "COSMO PGW", 'PGW-CTRL']
for j in range(ncol):
    sim = sims[j]
    cmap = cmaps[j]
    norm = norms[j]
    cs[0, j] = axs[0, j].pcolormesh(data_scd[sim]['x'], data_scd[sim]['y'], data_scd[sim]['scd'],
                                    cmap=cmap, norm=norm, shading="auto", transform=data_scd[sim]['crs'])
sims = ["ECHAM5 PI", "ECHAM5 LGM", 'LGM-PI']
for j in range(ncol):
    sim = sims[j]
    cmap = cmaps[j]
    norm = norms[j]
    cs[1, j] = axs[1, j].pcolormesh(data_scd[sim]['x'], data_scd[sim]['y'], data_scd[sim]['scd'],
                                    cmap=cmap, norm=norm, shading="auto", transform=data_scd[sim]['crs'])

for i in range(nrow):
    cax = fig.add_axes(
        [axs[i, 1].get_position().x1 + 0.01, axs[i, 1].get_position().y0, 0.015, axs[i, 1].get_position().height])
    cbar = fig.colorbar(cs[i, 1], cax=cax, orientation='vertical', extend='max', ticks=[0, 50, 100, 150, 200, 250])
    cbar.ax.minorticks_off()
    cbar.ax.tick_params(labelsize=13)
    cax = fig.add_axes(
        [axs[i, 2].get_position().x1 + 0.01, axs[i, 2].get_position().y0, 0.015, axs[i, 2].get_position().height])
    cbar = fig.colorbar(cs[i, 2], cax=cax, orientation='vertical', extend='max', ticks=[0, 20, 40, 60, 80, 100])
    cbar.ax.minorticks_off()
    cbar.ax.tick_params(labelsize=13)
# --
labels = ['CTRL | PI', 'PGW | LGM', 'PGW | LGM - CTRL | PI']
for j in range(ncol):
    label = labels[j]
    axs[0, j].set_title(f'{label}', pad=7, fontsize=14, loc='center')
# --
lefts = ['COSMO', 'ECHAM5']
for i in range(nrow):
    left = lefts[i]
    axs[i, 0].text(-0.2, 0.5, f'{left}', ha='right', va='center',
                   transform=axs[i, 0].transAxes, fontsize=14, rotation=90)
# --
for i in range(nrow):
    for j in range(ncol):
        id = index[i][j]
        t = axs[i, j].text(0.007, 0.988, f'({id})', ha='left', va='top',
                            transform=axs[i, j].transAxes, fontsize=12)
        t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))
# --
for i in range(nrow):
    axs[i, 0].text(-0.008, 0.94, '50°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.66, '40°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.38, '30°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.10, '20°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)

for j in range(ncol):
    axs[1, j].text(0.22, -0.02, '100°E', ha='center', va='top', transform=axs[1, j].transAxes, fontsize=13)
    axs[1, j].text(0.44, -0.02, '120°E', ha='center', va='top', transform=axs[1, j].transAxes, fontsize=13)
    axs[1, j].text(0.66, -0.02, '140°E', ha='center', va='top', transform=axs[1, j].transAxes, fontsize=13)
    axs[1, j].text(0.88, -0.02, '160°E', ha='center', va='top', transform=axs[1, j].transAxes, fontsize=13)
plotpath = "/project/pr133/rxiang/figure/paper2/results/lgm/"
fig.savefig(plotpath + 'nsnow_na.png', dpi=500, transparent='True')
plt.show()
plt.close()

# %%
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()
fig = plt.figure(figsize=(11, 6))
gs1 = gridspec.GridSpec(2, 2, left=0.05, bottom=0.03, right=0.585,
                        top=0.96, hspace=0.03, wspace=0.05,
                        width_ratios=[1, 1], height_ratios=[1, 1])
gs2 = gridspec.GridSpec(2, 1, left=0.664, bottom=0.03, right=0.925,
                        top=0.96, hspace=0.03, wspace=0.05, height_ratios=[1, 1])
ncol = 3  # edit here
nrow = 2

axs, cs, ct, topo, q = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                       np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                       np.empty(shape=(nrow, ncol), dtype='object')

for i in range(nrow):
    axs[i, 0] = fig.add_subplot(gs1[i, 0], projection=rot_pole_crs)
    axs[i, 0] = plotcosmo04_notick(axs[i, 0])
    axs[i, 1] = fig.add_subplot(gs1[i, 1], projection=rot_pole_crs)
    axs[i, 1] = plotcosmo04_notick_lgm(axs[i, 1], diff=False)
    axs[i, 2] = fig.add_subplot(gs2[i, 0], projection=rot_pole_crs)
    axs[i, 2] = plotcosmo04_notick_lgm(axs[i, 2], diff=False)

levels1 = MaxNLocator(nbins=100).tick_values(0, 250)
cmap1 = cmc.roma_r
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
levels2 = MaxNLocator(nbins=50).tick_values(0, 100)
# cmap2 = drywet(20, cmc.vik_r)
cmap2 = cmc.davos_r
norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)
# norm2 = colors.TwoSlopeNorm(vmin=-100., vcenter=0., vmax=100.)
# --
levels = [levels1, levels1, levels2]
norms = [norm1, norm1, norm2]
cmaps = [cmap1, cmap1, cmap2]
# --
sims = ["COSMO CTRL", "COSMO PGW", 'PGW-CTRL']
for j in range(ncol):
    sim = sims[j]
    cmap = cmaps[j]
    norm = norms[j]
    cs[0, j] = axs[0, j].pcolormesh(data_scd[sim]['x'], data_scd[sim]['y'], data_scd[sim]['scd'],
                                    cmap=cmap, norm=norm, shading="auto", transform=data_scd[sim]['crs'])
sims = ["ECHAM5 PI", "ECHAM5 LGM", 'LGM-PI']
for j in range(ncol):
    sim = sims[j]
    cmap = cmaps[j]
    norm = norms[j]
    cs[1, j] = axs[1, j].pcolormesh(data_scd[sim]['x'], data_scd[sim]['y'], data_scd[sim]['scd'],
                                    cmap=cmap, norm=norm, shading="auto", transform=data_scd[sim]['crs'])

# --
for i in range(nrow):
    cax = fig.add_axes(
        [axs[i, 1].get_position().x1 + 0.01, axs[i, 1].get_position().y0, 0.015, axs[i, 1].get_position().height])
    cbar = fig.colorbar(cs[0, 1], cax=cax, orientation='vertical', extend='max', ticks=[0, 50, 100, 150, 200, 250])
    cbar.ax.minorticks_off()
    cbar.ax.tick_params(labelsize=13)
    cax = fig.add_axes(
        [axs[i, 2].get_position().x1 + 0.01, axs[i, 2].get_position().y0, 0.015, axs[i, 2].get_position().height])
    cbar = fig.colorbar(cs[0, 2], cax=cax, orientation='vertical', extend='max', ticks=[0, 20, 40, 60, 80, 100])
    cbar.ax.minorticks_off()
    cbar.ax.tick_params(labelsize=13)
# --
labels = ['CTRL | PI', 'PGW | LGM', 'PGW | LGM - CTRL | PI']
for j in range(ncol):
    label = labels[j]
    axs[0, j].set_title(f'{label}', pad=7, fontsize=14, loc='center')
# --
lefts = ['COSMO', 'ECHAM5']
for i in range(nrow):
    left = lefts[i]
    axs[i, 0].text(-0.2, 0.5, f'{left}', ha='right', va='center',
                   transform=axs[i, 0].transAxes, fontsize=14, rotation=90)
# --
for i in range(nrow):
    axs[i, 0].text(-0.01, 0.83, '35°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.01, 0.57, '30°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.01, 0.31, '25°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.01, 0.05, '20°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)

for j in range(ncol):
    axs[1, j].text(0.06, -0.02, '90°E', ha='center', va='top', transform=axs[1, j].transAxes, fontsize=13)
    axs[1, j].text(0.46, -0.02, '100°E', ha='center', va='top', transform=axs[1, j].transAxes, fontsize=13)
    axs[1, j].text(0.86, -0.02, '110°E', ha='center', va='top', transform=axs[1, j].transAxes, fontsize=13)

plotpath = "/project/pr133/rxiang/figure/paper2/results/lgm/"
fig.savefig(plotpath + 'nsnow_local.png', dpi=500, transparent='True')
plt.show()





