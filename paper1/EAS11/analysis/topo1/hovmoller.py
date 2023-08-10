# -------------------------------------------------------------------------------
# modules
#
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import metpy.calc as mpcalc
import numpy as np
import xarray as xr
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mycolor import custom_div_cmap, drywet

# -------------------------------------------------------------------------------
# # %% read data
# #
# font = {'size': 14}
# matplotlib.rc('font', **font)
#
# lonmin = [70, 80, 90, 100, 85, 95, 110, 130]
# lonmax = [80, 90, 100, 110, 95, 105, 120, 140]
#
# # lonmin = [70]
# # lonmax = [80]
#
# for i, j in zip(lonmin, lonmax):
#     data = xr.open_dataset('/project/pr133/rxiang/data/cosmo/EAS11_ctrl/monsoon/TOT_PREC/hovmoller/' + f'01-05.TOT_PREC.cpm.{i}-{j}.nc')
#     cpm_ctrl = data['TOT_PREC'].values[...]
#     data = xr.open_dataset(f'/project/pr133/rxiang/data/cosmo/EAS11_topo1/monsoon/TOT_PREC/hovmoller/' + f'01-05.TOT_PREC.cpm.{i}-{j}.nc')
#     cpm_topo1 = data['TOT_PREC'].values[...]
#
#     lat = data['lat'].values[...]
#     time = data['time'].values[...]
#
#     time_, lat_ = np.meshgrid(time, lat)
#
#     # plot
#     hi = 4  # height in inches
#     wi = 12
#     ncol = 4  # edit here
#     nrow = 1
#     axs, cs, ct = np.empty(4, dtype='object'), np.empty(4, dtype='object'), np.empty(4, dtype='object')
#
#     fig = plt.figure(figsize=(wi, hi))
#     left, bottom, right, top = 0.05, 0.18, 0.99, 0.93
#     gs = gridspec.GridSpec(nrows=1, ncols=4, width_ratios=[.59, 2.5, 2.5, 2.5], left=left, bottom=bottom, right=right, top=top, wspace=0.07, hspace=0.1)
#
#     y_tick_labels = [u'5\N{DEGREE SIGN}N', u'10\N{DEGREE SIGN}N', u'15\N{DEGREE SIGN}N', u'20\N{DEGREE SIGN}N', u'25\N{DEGREE SIGN}N',
#                      u'30\N{DEGREE SIGN}N', u'35\N{DEGREE SIGN}N', u'40\N{DEGREE SIGN}N', u'45\N{DEGREE SIGN}N',
#                      u'50\N{DEGREE SIGN}N']
#
#     # Left plot for geographic reference (makes small map)
#     axs[k, 0] = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=0))
#     axs[k, 0].set_extent([i, j, 5, 50], ccrs.PlateCarree(central_longitude=0))
#     axs[k, 0].set_xticks([i, j])
#     axs[k, 0].set_xticklabels([f'{i}°E', f'{j}°E'])
#     axs[k, 0].set_yticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
#     axs[k, 0].set_yticklabels(y_tick_labels)
#     axs[k, 0].grid(linestyle='dotted', linewidth=2)
#
#     # Add geopolitical boundaries for map reference
#     axs[k, 0].add_feature(cfeature.COASTLINE.with_scale('50m'))
#     axs[k, 0].add_feature(cfeature.LAKES.with_scale('50m'), linewidths=0.5)
#     axs[k, 0].add_feature(cfeature.LAND.with_scale('50m'), edgecolor='k')
#     axs[k, 0].add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')
#     axs[k, 0].add_feature(cfeature.RIVERS.with_scale('50m'))
#
#     # right plot for Hovmoller diagram
#     axs[k, 1] = fig.add_subplot(gs[0, 1])
#     axs[k, 2] = fig.add_subplot(gs[0, 2])
#     axs[k, 3] = fig.add_subplot(gs[0, 3])
#     # Plot of chosen variable averaged over longitude and slightly smoothed
#     levels = MaxNLocator(nbins=25).tick_values(0, 25)
#     cmap = cmc.davos_r
#     norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
#
#     cf2 = axs[k, 1].pcolormesh(time_, lat_, np.transpose(cpm_ctrl[:, :, 0]), cmap=cmap, norm=norm)
#     cf3 = axs[k, 2].pcolormesh(time_, lat_, np.transpose(cpm_topo1[:, :, 0]), cmap=cmap, norm=norm)
#
#     levels = MaxNLocator(nbins=23).tick_values(-10, 10)
#     cmap = drywet(25, cmc.vik_r)
#     norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
#     cf4 = axs[k, 3].pcolormesh(time_, lat_, np.transpose(cpm_topo1[:, :, 0]) - np.transpose(cpm_ctrl[:, :, 0]), cmap=cmap, clim=(-10, 10))
#     ct2 = axs[k, 1].contour(time_, lat_, mpcalc.smooth_n_point(np.transpose(cpm_ctrl[:, :, 0]), 5, 1),
#                       levels=[5, 10, 15, 20], colors='k', linewidths=1)
#     ct3 = axs[k, 2].contour(time_, lat_, mpcalc.smooth_n_point(np.transpose(cpm_topo1[:, :, 0]), 5, 1),
#                       levels=[5, 10, 15, 20], colors='k', linewidths=1)
#     ct4 = axs[k, 3].contour(time_, lat_, mpcalc.smooth_n_point(np.transpose(cpm_topo1[:, :, 0]) - np.transpose(cpm_ctrl[:, :, 0]), 9, 1),
#                       levels=[-9, -6, -3, 3, 6, 9], colors='k', linewidths=1)
#
#     axs[k, 1].text(0, 1.01, 'CTRL11', ha='left', va='bottom', transform=axs[k, 1].transAxes, fontsize=14)
#     axs[k, 2].text(0, 1.01, 'TRED11', ha='left', va='bottom', transform=axs[k, 2].transAxes, fontsize=14)
#     axs[k, 3].text(0, 1.01, 'TRED11 - CTRL11', ha='left', va='bottom', transform=axs[k, 3].transAxes, fontsize=14)
#
#     for ax, ct in zip([axs[k, 1], axs[k, 2], axs[k, 3]], [ct2, ct3, ct4]):
#         clabel = ax.clabel(ct, inline=True, use_clabeltext=True, fontsize=13)
#         for l in clabel:
#             l.set_rotation(0)
#         [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]
#
#     cax = fig.add_axes([axs[k, 1].get_position().x0, axs[k, 1].get_position().y0 - 0.115, axs[k, 1].get_position().width*2+0.02, 0.035])
#     cbar1 = fig.colorbar(cf2, cax=cax, ticks=np.linspace(0, 24, 13, endpoint=True), orientation='horizontal', extend='max')
#     cbar1.ax.tick_params(labelsize=13)
#     # cbar1.ax.set_xlabel('mm day$^{-1}$', fontsize=13, labelpad=-0.01)
#
#     cax = fig.add_axes(
#         [axs[k, 3].get_position().x0, axs[k, 3].get_position().y0 - 0.115, axs[k, 3].get_position().width, 0.035])
#     cbar2 = fig.colorbar(cf4, cax=cax, ticks=np.linspace(-10, 10, 11, endpoint=True), orientation='horizontal', extend='both')
#     cbar2.ax.tick_params(labelsize=13)
#     # cbar2.ax.set_xlabel('mm day$^{-1}$', fontsize=13, labelpad=-0.01)
#
#     # Make some ticks and tick labels
#     for ax in axs[k, 1], axs[k, 2], axs[k, 3]:
#         ax.set_xticks(time[3::6])
#         ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
#         ax.set_ylim(5, 50)
#         ax.set_yticklabels([])
#         ax.tick_params(axis='y', length=0.1)
#         # ax.set_yticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
#         # ax.set_yticklabels(y_tick_labels)
#         ax.grid(linestyle='dotted', linewidth=2)
#
#     # fig.suptitle(f'Precipitation Climatology ({i}°-{j}°E)', fontsize=16, fontweight='bold')
#
#     # Set some titles
#     # plt.title('250-hPa V-wind', loc='left', fontsize=10)
#     # plt.title('Time Range: {0:%Y%m%d %HZ} - {1:%Y%m%d %HZ}'.format(vtimes[0], vtimes[-1]),
#               # loc='right', fontsize=10)
#
#     plt.show()
#     plotpath = "/project/pr133/rxiang/figure/paper1/results/TRED/hovmoller/"
#     fig.savefig(plotpath + f'{i}°-{j}°E.png', dpi=500)
#     plt.close(fig)
# %%
font = {'size': 14}
matplotlib.rc('font', **font)

lonmin = [85, 95, 110]
lonmax = [95, 105, 120]

# plot
hi = 10  # height in inches
wi = 12
ncol = 4  # edit here
nrow = 3
axs, cs, ct = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object')

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.05, 0.085, 0.99, 0.97
gs = gridspec.GridSpec(nrows=3, ncols=4, width_ratios=[.59, 2.5, 2.5, 2.5], height_ratios=[1, 1, 1], left=left, bottom=bottom, right=right,
                       top=top, wspace=0.07, hspace=0.15)

y_tick_labels = [u'5\N{DEGREE SIGN}N', u'10\N{DEGREE SIGN}N', u'15\N{DEGREE SIGN}N', u'20\N{DEGREE SIGN}N',
                 u'25\N{DEGREE SIGN}N',
                 u'30\N{DEGREE SIGN}N', u'35\N{DEGREE SIGN}N', u'40\N{DEGREE SIGN}N', u'45\N{DEGREE SIGN}N',
                 '']

k = 0
lb = ['a', 'b', 'c']

for i, j in zip(lonmin, lonmax):
    data = xr.open_dataset('/project/pr133/rxiang/data/cosmo/EAS11_ctrl/monsoon/TOT_PREC/hovmoller/' + f'01-05.TOT_PREC.cpm.{i}-{j}.nc')
    cpm_ctrl = data['TOT_PREC'].values[...]
    data = xr.open_dataset(f'/project/pr133/rxiang/data/cosmo/EAS11_topo1/monsoon/TOT_PREC/hovmoller/' + f'01-05.TOT_PREC.cpm.{i}-{j}.nc')
    cpm_topo1 = data['TOT_PREC'].values[...]

    lat = data['lat'].values[...]
    time = data['time'].values[...]

    time_, lat_ = np.meshgrid(time, lat)

    # Left plot for geographic reference (makes small map)
    axs[k, 0] = fig.add_subplot(gs[k, 0], projection=ccrs.PlateCarree(central_longitude=0))
    axs[k, 0].set_extent([i, j, 5, 50], ccrs.PlateCarree(central_longitude=0))
    axs[k, 0].set_xticks([i, j])
    axs[k, 0].set_xticklabels([f'{i}°E', f'{j}°E'])
    axs[k, 0].set_yticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    axs[k, 0].set_yticklabels(y_tick_labels)
    axs[k, 0].grid(linestyle='dotted', linewidth=2)

    # Add geopolitical boundaries for map reference
    axs[k, 0].add_feature(cfeature.COASTLINE.with_scale('50m'))
    axs[k, 0].add_feature(cfeature.LAKES.with_scale('50m'), linewidths=0.5)
    axs[k, 0].add_feature(cfeature.LAND.with_scale('50m'), edgecolor='k')
    axs[k, 0].add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')
    axs[k, 0].add_feature(cfeature.RIVERS.with_scale('50m'))

    # right plot for Hovmoller diagram
    axs[k, 1] = fig.add_subplot(gs[k, 1])
    axs[k, 2] = fig.add_subplot(gs[k, 2])
    axs[k, 3] = fig.add_subplot(gs[k, 3])
    # Plot of chosen variable averaged over longitude and slightly smoothed
    levels = MaxNLocator(nbins=25).tick_values(0, 25)
    cmap = cmc.davos_r
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    cf2 = axs[k, 1].pcolormesh(time_, lat_, np.transpose(cpm_ctrl[:, :, 0]), cmap=cmap, norm=norm)
    cf3 = axs[k, 2].pcolormesh(time_, lat_, np.transpose(cpm_topo1[:, :, 0]), cmap=cmap, norm=norm)

    levels = MaxNLocator(nbins=23).tick_values(-10, 10)
    cmap = drywet(25, cmc.vik_r)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    cf4 = axs[k, 3].pcolormesh(time_, lat_, np.transpose(cpm_topo1[:, :, 0]) - np.transpose(cpm_ctrl[:, :, 0]), cmap=cmap, clim=(-10, 10))
    ct2 = axs[k, 1].contour(time_, lat_, mpcalc.smooth_n_point(np.transpose(cpm_ctrl[:, :, 0]), 5, 1),
                      levels=[5, 10, 15, 20], colors='k', linewidths=1)
    ct3 = axs[k, 2].contour(time_, lat_, mpcalc.smooth_n_point(np.transpose(cpm_topo1[:, :, 0]), 5, 1),
                      levels=[5, 10, 15, 20], colors='k', linewidths=1)
    ct4 = axs[k, 3].contour(time_, lat_, mpcalc.smooth_n_point(np.transpose(cpm_topo1[:, :, 0]) - np.transpose(cpm_ctrl[:, :, 0]), 9, 1),
                      levels=[-9, -6, -3, 3, 6, 9], colors='k', linewidths=1)

    for ax, ct in zip([axs[k, 1], axs[k, 2], axs[k, 3]], [ct2, ct3, ct4]):
        clabel = ax.clabel(ct, inline=True, use_clabeltext=True, fontsize=13)
        for l in clabel:
            l.set_rotation(0)
        [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]

    # Make some ticks and tick labels
    for ax in axs[k, 1], axs[k, 2], axs[k, 3]:
        ax.set_xticks(time[3::6])
        ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        ax.set_ylim(5, 50)
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0.1)
        # ax.set_yticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        # ax.set_yticklabels(y_tick_labels)
        ax.grid(linestyle='dotted', linewidth=2)

    label = lb[k]
    t = axs[k, 0].text(-0.2, 1, f'({label})', ha='right', va='center',
                       transform=axs[k, 0].transAxes, fontsize=15)

    k += 1

axs[0, 1].text(0, 1.01, 'CTRL11', ha='left', va='bottom', transform=axs[0, 1].transAxes, fontsize=15)
axs[0, 2].text(0, 1.01, 'TRED11', ha='left', va='bottom', transform=axs[0, 2].transAxes, fontsize=15)
axs[0, 3].text(0, 1.01, 'TRED11 - CTRL11', ha='left', va='bottom', transform=axs[0, 3].transAxes, fontsize=15)

cax = fig.add_axes([axs[2, 1].get_position().x0, axs[2, 1].get_position().y0 - 0.055, axs[2, 1].get_position().width*2+0.017, 0.017])
cbar1 = fig.colorbar(cf2, cax=cax, ticks=np.linspace(0, 24, 13, endpoint=True), orientation='horizontal', extend='max')
cbar1.ax.tick_params(labelsize=13)
# cbar1.ax.set_xlabel('mm day$^{-1}$', fontsize=13, labelpad=-0.01)

cax = fig.add_axes(
    [axs[2, 3].get_position().x0, axs[2, 3].get_position().y0 - 0.055, axs[2, 3].get_position().width, 0.017])
cbar2 = fig.colorbar(cf4, cax=cax, ticks=np.linspace(-10, 10, 11, endpoint=True), orientation='horizontal', extend='both')
cbar2.ax.tick_params(labelsize=13)
# cbar2.ax.set_xlabel('mm day$^{-1}$', fontsize=13, labelpad=-0.01)

# fig.suptitle(f'Precipitation Climatology ({i}°-{j}°E)', fontsize=16, fontweight='bold')

# Set some titles
plt.show()
plotpath = "/project/pr133/rxiang/figure/paper1/results/TRED/hovmoller/"
fig.savefig(plotpath + 'hovmoller.png', dpi=500)
plt.close(fig)










