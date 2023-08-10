# Description: plot the summer climatology: precipitation, water vapor flux, wind at 850 and at 200,
# Load modules
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
from mycolor import custom_div_cmap, cbr_wet, cbr_drywet, drywet, custom_seq_cmap_
from plotcosmomap import plotcosmo_notick, pole, plotcosmo_notick_nogrid
import matplotlib.colors as colors
import scipy.ndimage as ndimage
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from metpy.units import units
import metpy.calc as mpcalc

mpl.style.use("classic")
font = {'size': 14}
matplotlib.rc('font', **font)
plt.rcParams['hatch.linewidth'] = 0.5
plt.rcParams['hatch.color'] = 'dimgrey'

###############################################################################
# Function
###############################################################################
def compute_pvalue(ctrl, topo):
    ctrl = np.array(ctrl)
    topo = np.array(topo)
    p = np.zeros((int(ctrl.shape[1]), int(ctrl.shape[2]))) # make sure the shape tuple contains only integers
    for i in range(ctrl.shape[1]):
        for j in range(ctrl.shape[2]):
            ii, jj = mannwhitneyu(ctrl[:, i, j], topo[:, i, j], alternative='two-sided')
            p[i, j] = jj
    p_values = multipletests(p.flatten(), alpha=0.05, method='fdr_bh')[1].reshape((int(ctrl.shape[1]), int(ctrl.shape[2]))) # make sure the shape tuple contains only integers
    return p, p_values

###############################################################################
# Data
###############################################################################
sims = ['ctrl', 'lgm']
path = "/project/pr133/rxiang/data/cosmo/"

data = {}
labels = ['PD', 'LGM', 'LGM - PD']
lb = [['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i'], ['j', 'k', 'l']]

g = 9.80665

vars = ['TOT_PREC', 'T_2M', 'IUQ', 'IVQ', 'TQF', 'FI500', 'U500', 'V500', 'WS500', 'PMSL', 'U850', 'V850', 'WS850', 'T500']
# load data
for s in range(len(sims)):
    sim = sims[s]
    data[sim] = {}
    ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/monsoon/TOT_PREC/' + f'01-05.TOT_PREC.wtr.yearmean.nc')
    wtr = ds['TOT_PREC'].values[...]
    data[sim]['TOT_PREC'] = wtr
    ds = xr.open_dataset(f'{path}' + f'EAS11_lgm_ssu/monsoon/T_2M/' + f'01-05.T_2M.wtr.yearmean.nc')
    wtr = ds['T_2M'].values[...]
    data[sim]['T_2M'] = wtr
    ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/monsoon/IVT/' + f'01-05.IVT.wtr.yearmean.nc')
    iuq = ds['IUQ'].values[:, :, :]
    data[sim]['IUQ'] = iuq
    ivq = ds['IVQ'].values[:, :, :]
    data[sim]['IVQ'] = ivq
    data[sim]['TQF'] = np.sqrt(iuq ** 2 + ivq ** 2)
    ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/monsoon/FI/' + f'01-05.FI.50000.wtr.yearmean.nc')
    wtr = ds['FI'].values[:, 0, :, :] / g
    data[sim]['FI500'] = wtr
    ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/monsoon/U/' + f'01-05.U.50000.wtr.yearmean.nc')
    u = ds['U'].values[:, 0, :, :]
    data[sim]['U500'] = u * units('m/s')
    ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/monsoon/V/' + f'01-05.V.50000.wtr.yearmean.nc')
    v = ds['V'].values[:, 0, :, :]
    data[sim]['V500'] = v * units('m/s')
    data[sim]['WS500'] = np.nanmean(mpcalc.wind_speed(data[sim]['U500'], data[sim]['V500']), axis=0)
    ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/monsoon/U/' + f'01-05.U.85000.wtr.yearmean.nc')
    u = ds['U'].values[:, 0, :, :]
    data[sim]['U850'] = u * units('m/s')
    ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/monsoon/V/' + f'01-05.V.85000.wtr.yearmean.nc')
    v = ds['V'].values[:, 0, :, :]
    data[sim]['V850'] = v * units('m/s')
    data[sim]['WS850'] = np.nanmean(mpcalc.wind_speed(data[sim]['U850'], data[sim]['V850']), axis=0)
    ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/monsoon/PMSL/' + f'01-05.PMSL.wtr.yearmean.nc')
    wtr = ds['PMSL'].values[:, :, :] / 100
    data[sim]['PMSL'] = wtr
    ds = xr.open_dataset(f'{path}' + f'EAS11_{sim}/monsoon/T/' + f'01-05.T.50000.wtr.yearmean.nc')
    wtr = ds['T'][:, 0, :, :]
    data[sim]['T500'] = wtr

# compute difference
data['diff'] = {}
for v in range(len(vars)):
    var = vars[v]
    data['diff'][var] = data['lgm_ssu'][var] - data['ctrl'][var]

###############################################################################
# %% Compute p-value
###############################################################################
# p1, corr_p1 = compute_pvalue(data['ctrl']['TOT_PREC'], data['lgm_ssu']['TOT_PREC'])
# mask1 = np.full_like(p1, fill_value=np.nan)
# mask1[p1 > 0.05] = 1
# np.save('/project/pr133/rxiang/data/cosmo/sgnfctt/lgm_ssu_TOT_PREC_wtr.npy', mask1)
#
# # p2, corr_p2 = compute_pvalue(data['ctrl']['TQF'], data['lgm_ssu']['TQF'])
# # mask2 = np.full_like(p2, fill_value=np.nan)
# # mask2[p2 > 0.05] = 1
# # np.save('/project/pr133/rxiang/data/cosmo/sgnfctt/lgm_ssu_TQF_wtr.npy', mask2)
#
# p3, corr_p3 = compute_pvalue(data['ctrl']['PMSL'], data['lgm_ssu']['PMSL'])
# mask3 = np.full_like(p3, fill_value=np.nan)
# mask3[p3 > 0.05] = 1
# np.save('/project/pr133/rxiang/data/cosmo/sgnfctt/lgm_ssu_PMSL_wtr.npy', mask3)
#
# p4, corr_p4 = compute_pvalue(data['ctrl']['FI'], data['lgm_ssu']['FI'])
# mask4 = np.full_like(p4, fill_value=np.nan)
# mask4[p4 > 0.05] = 1
# np.save('/project/pr133/rxiang/data/cosmo/sgnfctt/lgm_ssu_FI500_wtr.npy', mask4)

# mask1 = np.load('/project/pr133/rxiang/data/cosmo/sgnfctt/lgm_ssu_TOT_PREC_wtr.npy')
# mask3 = np.load('/project/pr133/rxiang/data/cosmo/sgnfctt/lgm_ssu_PMSL_wtr.npy')
# mask4 = np.load('/project/pr133/rxiang/data/cosmo/sgnfctt/lgm_ssu_FI500_wtr.npy')

###############################################################################
# %% Plot
###############################################################################
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()
rlon_, rlat_ = np.meshgrid(rlon, rlat)
sims = ['ctrl', 'lgm_ssu', 'diff']
fig = plt.figure(figsize=(11, 7.5))
gs1 = gridspec.GridSpec(4, 2, left=0.05, bottom=0.03, right=0.585,
                        top=0.96, hspace=0.05, wspace=0.05,
                        width_ratios=[1, 1], height_ratios=[1, 1, 1, 1])
gs2 = gridspec.GridSpec(4, 1, left=0.664, bottom=0.03, right=0.925,
                        top=0.96, hspace=0.05, wspace=0.05)
ncol = 3  # edit here
nrow = 4

axs, cs, ct, topo, q = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                       np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), \
                       np.empty(shape=(nrow, ncol), dtype='object')

for i in range(nrow):
    for j in range(ncol - 1):
        axs[i, j] = fig.add_subplot(gs1[i, j], projection=rot_pole_crs)
        axs[i, j] = plotcosmo_notick(axs[i, j])
    axs[i, 2] = fig.add_subplot(gs2[i, 0], projection=rot_pole_crs)
    axs[i, 2] = plotcosmo_notick_nogrid(axs[i, 2])

# plot topo_diff
# for i in range(nrow):
#     axs[i, 2] = plotcosmo_notick(axs[i, 2])
#     topo[i, 2] = axs[i, 2].contour(lon_, lat_, hsurf_diff, levels=[500], colors='darkgreen', linewidths=1,
#                                    transform=ccrs.PlateCarree())

axs[1, 2] = plotcosmo_notick(axs[1, 2])
#############################################################################
# --- plot precipitation
levels1 = MaxNLocator(nbins=15).tick_values(0, 15)
cmap1 = cmc.davos_r
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
levels2 = MaxNLocator(nbins=11).tick_values(-3, 3)
cmap2 = drywet(19, cmc.vik_r)
norm2 = colors.TwoSlopeNorm(vmin=-3., vcenter=0., vmax=3.)
# --
levels = [levels1, levels1, levels2]
norms = [norm1, norm1, norm2]
cmaps = [cmap1, cmap1, cmap2]
# --
for j in range(ncol):
    sim = sims[j]
    cmap = cmaps[j]
    norm = norms[j]
    cs[0, j] = axs[0, j].pcolormesh(rlon, rlat, np.nanmean(data[sim]['TOT_PREC'], axis=0), cmap=cmap, norm=norm, shading="auto")
    axs[0, j].barbs(rlon[::55], rlat[::55], np.nanmean(data[sim]['U850'], axis=0)[::55, ::55].to('kt').m,
                    np.nanmean(data[sim]['V850'], axis=0)[::55, ::55].to('kt').m,
                    data[sim]['WS850'][::55, ::55].to('kt').m,
                    pivot='middle', color='black', length=4.5, linewidth=0.6,
                    sizes=dict(emptybarb=0.07, spacing=0.2))
# --
# ha = axs[0, 2].contourf(rlon, rlat, mask1, levels=1, colors='none', hatches=['////'], rasterized=True, zorder=101)
# --
cax = fig.add_axes(
    [axs[0, 1].get_position().x1 + 0.01, axs[0, 1].get_position().y0, 0.015, axs[0, 1].get_position().height])
cbar = fig.colorbar(cs[0, 1], cax=cax, orientation='vertical', extend='max', ticks=[0, 3, 6, 9, 12, 15])
cbar.ax.tick_params(labelsize=13)
cax = fig.add_axes(
    [axs[0, 2].get_position().x1 + 0.01, axs[0, 2].get_position().y0, 0.015, axs[0, 2].get_position().height])
cbar = fig.colorbar(cs[0, 2], cax=cax, orientation='vertical', extend='both', ticks=[-3, -2, -1, 0, 1, 2, 3])
cbar.ax.tick_params(labelsize=13)
# --
for j in range(ncol):
    label = labels[j]
    axs[0, j].set_title(f'{label}', pad=7, fontsize=14, loc='center')

#############################################################################
# --- plot water vapor flux stream
levels1 = MaxNLocator(nbins=15).tick_values(50, 350)
# cmap1 = plt.cm.get_cmap("Spectral")
cmap1 = cmc.roma
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
levels2 = MaxNLocator(nbins=17).tick_values(-10, 10)
cmap2 = drywet(25, cmc.vik_r)
norm2 = colors.TwoSlopeNorm(vmin=-10., vcenter=0., vmax=10.)
# --
levels = [levels1, levels1, levels2]
norms = [norm1, norm1, norm2]
cmaps = [cmap1, cmap1, cmap2]
scales = [6000, 6000, 2000]
# --
for j in range(ncol):
    sim = sims[j]
    cmap = cmaps[j]
    norm = norms[j]
    scale = scales[j]
    # cs[1, j] = axs[1, j].pcolormesh(rlon, rlat, data[sim]['TQF'], cmap=cmap, norm=norm, shading="auto")
    q[1, j] = axs[1, j].quiver(rlon[::15], rlat[::15], np.nanmean(data[sim]['IUQ'][:, ::15, ::15], axis=0),
                               np.nanmean(data[sim]['IVQ'][:, ::15, ::15], axis=0),
                               np.nanmean(data[sim]['TQF'][:, ::15, ::15], axis=0),
                               cmap=cmap, norm=norm, scale=scale, headaxislength=3.5,
                               headwidth=5, minshaft=0)
# --
cax = fig.add_axes(
    [axs[1, 1].get_position().x1 + 0.01, axs[1, 1].get_position().y0, 0.015, axs[1, 1].get_position().height])
cbar = fig.colorbar(q[1, 1], cax=cax, orientation='vertical', extend='both', ticks=[50, 100, 150, 200, 250, 300, 350])
cbar.ax.tick_params(labelsize=13)
cax = fig.add_axes(
    [axs[1, 2].get_position().x1 + 0.01, axs[1, 2].get_position().y0, 0.015, axs[1, 2].get_position().height])
cbar = fig.colorbar(q[1, 2], cax=cax, orientation='vertical', extend='both')
cbar.ax.tick_params(labelsize=13)

# --- plot SLP
# cmap1 = plt.cm.get_cmap("Spectral")
# levels1 = np.linspace(230, 290, 31, endpoint=True)
levels1 = np.linspace(1000, 1050, 28, endpoint=True)
cmap1 = cmc.roma_r
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
cmap2 = custom_div_cmap(25, cmc.vik)
norm2 = colors.TwoSlopeNorm(vmin=-10., vcenter=0., vmax=20.)
# --
levels1 = np.linspace(1000, 1050, 9, endpoint=True)
levels2 = [-20, -15, -10, -5, 5, 10, 15, 20]
levels = [levels1, levels1, levels2]
norms = [norm1, norm1, norm2]
cmaps = [cmap1, cmap1, cmap2]
# --
for j in range(ncol):
    sim = sims[j]
    cmap = cmaps[j]
    norm = norms[j]
    level = levels[j]
    cs[2, j] = axs[2, j].pcolormesh(rlon, rlat, np.nanmean(data[sim]['PMSL'], axis=0), cmap=cmap, norm=norm, shading="auto")
    # ct[2, j] = axs[2, j].contour(rlon, rlat, np.nanmean(data[sim]['PMSL'], axis=0), levels=level,
    #                              colors='k', linewidths=.8)
    # clabel = axs[2, j].clabel(ct[2, j], levels=level, inline=True, fontsize=10,
    #                           use_clabeltext=True)
    # for l in clabel:
    #     l.set_rotation(0)
# --
# ha = axs[2, 2].contourf(rlon, rlat, mask3, levels=1, colors='none', hatches=['////'], rasterized=True, zorder=101)
# --
cax = fig.add_axes(
    [axs[2, 1].get_position().x1 + 0.01, axs[2, 1].get_position().y0, 0.015, axs[2, 1].get_position().height])
cbar = fig.colorbar(cs[2, 1], cax=cax, orientation='vertical', extend='both', ticks=np.linspace(1000, 1050, 6, endpoint=True))
cbar.ax.tick_params(labelsize=13)
cbar.ax.ticklabel_format(useOffset=False)
cax = fig.add_axes(
    [axs[2, 2].get_position().x1 + 0.01, axs[2, 2].get_position().y0, 0.015, axs[2, 2].get_position().height])
cbar = fig.colorbar(cs[2, 2], cax=cax, orientation='vertical', extend='both', ticks=[-10, -5, 0, 10, 20])
cbar.ax.tick_params(labelsize=13)
# --
for j in range(ncol):
    label = labels[j]
    axs[0, j].set_title(f'{label}', pad=7, fontsize=14, loc='center')

# # --- plot geopotential height
# levels1 = np.linspace(5090, 5930, 15, endpoint=True)
# # cmap1 = plt.cm.get_cmap("Spectral")
# cmap1 = cmc.roma_r
# norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
# cmap2 = custom_div_cmap(17, cmc.vik)
# norm2 = colors.TwoSlopeNorm(vmin=-24., vcenter=0., vmax=8.)
# # --
# norms = [norm1, norm1, norm2]
# cmaps = [cmap1, cmap1, cmap2]
# scales = [6000, 6000, 1000]
# # --
# for j in range(ncol):
#     sim = sims[j]
#     cmap = cmaps[j]
#     norm = norms[j]
#     scale = scales[j]
#     cs[2, j] = axs[2, j].pcolormesh(rlon, rlat, data[sim]['FI'], cmap=cmap, norm=norm, shading="auto")
#
# # --
# cax = fig.add_axes(
#     [axs[2, 1].get_position().x1 + 0.01, axs[2, 1].get_position().y0, 0.015, axs[2, 1].get_position().height])
# cbar = fig.colorbar(cs[2, 1], cax=cax, orientation='vertical', extend='both',
#                     ticks=np.linspace(5150, 5870, 7, endpoint=True))
# cbar.ax.tick_params(labelsize=13)
# cax = fig.add_axes(
#     [axs[2, 2].get_position().x1 + 0.01, axs[2, 2].get_position().y0, 0.015, axs[2, 2].get_position().height])
# cbar = fig.colorbar(cs[2, 2], cax=cax, orientation='vertical', extend='both', ticks=[-24, -18, -12, -6, 0, 2, 4, 6, 8])
# cbar.ax.tick_params(labelsize=13)

# --- plot wind 200
# levels1 = np.linspace(5, 40, 15, endpoint=True)
# # cmap1 = plt.cm.get_cmap("Spectral")
# cmap1 = cmc.roma_r
# norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)
# levels2 = MaxNLocator(nbins=15).tick_values(-1, 1)
# cmap2 = custom_div_cmap(17, cmc.vik)
# norm2 = colors.TwoSlopeNorm(vmin=-1, vcenter=0., vmax=1.)
# # --
# levels = [levels1, levels1, levels2]
# norms = [norm1, norm1, norm2]
# cmaps = [cmap1, cmap1, cmap2]
# # --
# for j in range(ncol):
#     sim = sims[j]
#     cmap = cmaps[j]
#     norm = norms[j]
#     scale = scales[j]
#     # cs[1, j] = axs[1, j].pcolormesh(rlon, rlat, data[sim]['TQF'], cmap=cmap, norm=norm, shading="auto")
#     q[3, j] = axs[3, j].streamplot(rlon, rlat, data[sim]['U200'], data[sim]['V200'], color=data[sim]['WS200'],
#                                    density=1, cmap=cmap, norm=norm)
# # --
# cax = fig.add_axes(
#     [axs[3, 1].get_position().x1 + 0.01, axs[3, 1].get_position().y0, 0.015, axs[3, 1].get_position().height])
# cbar = fig.colorbar(q[3, 1].lines, cax=cax, orientation='vertical', extend='both', ticks=[5, 10, 15, 20, 25, 30, 35, 40])
# cbar.ax.tick_params(labelsize=13)
# cax = fig.add_axes(
#     [axs[3, 2].get_position().x1 + 0.01, axs[3, 2].get_position().y0, 0.015, axs[3, 2].get_position().height])
# cbar = fig.colorbar(q[3, 2].lines, cax=cax, orientation='vertical', extend='both', ticks=[-1, -0.5, 0, 0.5, 1])
# cbar.ax.tick_params(labelsize=13)

# --- plot wind 500
cmap1 = cmc.roma_r
norm1 = BoundaryNorm(np.linspace(232, 272, 41, endpoint=True), ncolors=cmap1.N, clip=True)
cmap2 = custom_div_cmap(25, cmc.vik)
norm2 = colors.TwoSlopeNorm(vmin=-10, vcenter=0., vmax=10.)

levels1 = np.linspace(5200, 5900, 8, endpoint=True)
levels2 = [-60, -40, -20, 20, 40, 60]
# --
levels = [levels1, levels1, levels2]
norms = [norm1, norm1, norm2]
cmaps = [cmap1, cmap1, cmap2]
# --
for j in range(ncol):
    sim = sims[j]
    cmap = cmaps[j]
    norm = norms[j]
    level = levels[j]
    cs[3, j] = axs[3, j].pcolormesh(rlon, rlat, np.nanmean(data[sim]['T500'], axis=0), cmap=cmap, norm=norm,
                                    shading="auto")
    ct[3, j] = axs[3, j].contour(rlon, rlat, np.nanmean(data[sim]['FI500'], axis=0), levels=level,
                                 colors='k', linewidths=.8)
    wind_slice = (slice(None, None, 5), slice(None, None, 5))

    clabel = axs[3, j].clabel(ct[3, j], levels=level, inline=True, fontsize=10,
                              use_clabeltext=True)
    for l in clabel:
        l.set_rotation(0)
# --
# ha = axs[3, 2].contourf(rlon, rlat, mask3, levels=1, colors='none', hatches=['////'], rasterized=True, zorder=101)
# --
cax = fig.add_axes(
    [axs[3, 1].get_position().x1 + 0.01, axs[3, 1].get_position().y0, 0.015, axs[3, 1].get_position().height])
cbar = fig.colorbar(cs[3, 1], cax=cax, orientation='vertical', extend='both', ticks=np.linspace(235, 270, 8, endpoint=True))
cbar.ax.tick_params(labelsize=13)
cax = fig.add_axes(
    [axs[3, 2].get_position().x1 + 0.01, axs[3, 2].get_position().y0, 0.015, axs[3, 2].get_position().height])
cbar = fig.colorbar(cs[3, 2], cax=cax, orientation='vertical', extend='both', ticks=[-10, -5, 0, 5, 10])
cbar.ax.tick_params(labelsize=13)

#############################################################################
for i in range(nrow):
    for j in range(ncol):
        label = lb[i][j]
        t = axs[i, j].text(0.01, 0.985, f'({label})', ha='left', va='top',
                           transform=axs[i, j].transAxes, fontsize=14)
        t.set_bbox(dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))

for i in range(nrow):
    axs[i, 0].text(-0.008, 0.95, '50°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.77, '40°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.59, '30°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.41, '20°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.23, '10°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.05, '0°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)

for j in range(ncol):
    axs[3, j].text(0.12, -0.02, '80°E', ha='center', va='top', transform=axs[3, j].transAxes, fontsize=13)
    axs[3, j].text(0.32, -0.02, '100°E', ha='center', va='top', transform=axs[3, j].transAxes, fontsize=13)
    axs[3, j].text(0.52, -0.02, '120°E', ha='center', va='top', transform=axs[3, j].transAxes, fontsize=13)
    axs[3, j].text(0.72, -0.02, '140°E', ha='center', va='top', transform=axs[3, j].transAxes, fontsize=13)
    axs[3, j].text(0.92, -0.02, '160°E', ha='center', va='top', transform=axs[3, j].transAxes, fontsize=13)


plt.show()
plotpath = "/project/pr133/rxiang/figure/paper2/results/lgm/"
fig.savefig(plotpath + 'wtr.png', dpi=500)
plt.close(fig)







