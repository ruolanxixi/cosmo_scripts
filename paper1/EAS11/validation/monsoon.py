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
# read data
#
font = {'size': 14}
matplotlib.rc('font', **font)

lonmin = [70, 80, 90, 100, 110, 120, 130, 140]
lonmax = [80, 90, 100, 110, 120, 130, 140, 150]

lonmin = [70, 110]
lonmax = [80, 120]

lb = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
# %%
k = 0
for i, j in zip(lonmin, lonmax):
    data = xr.open_dataset(
        '/project/pr133/rxiang/data/cosmo/EAS11_ctrl/monsoon/TOT_PREC/hovmoller/' + f'01-05.TOT_PREC.cpm.{i}-{j}.nc')
    cpm_ctrl = data['TOT_PREC'].values[...]
    lat1 = data['lat'].values[...]
    time1 = data['time'].values[...]
    data = xr.open_dataset('/project/pr133/rxiang/data/obs/pr/IMERG/hovmoller/' + f'IMERG.2001-2005.cpm.{i}-{j}.nc')
    cpm_imerg = data['precipitation_corr'].values[...]
    lat2 = data['lat'].values[...]
    time2 = data['time'].values[...]

    time1_, lat1_ = np.meshgrid(time1, lat1)
    time2_, lat2_ = np.meshgrid(time1, lat2)

    # plot
    ar = 2.1  # initial aspect ratio for first trial
    hi = 4  # height in inches
    wi = hi * ar  # height in inches
    ncol = 3  # edit here
    nrow = 1
    axs, cs, ct = np.empty(3, dtype='object'), np.empty(3, dtype='object'), np.empty(3, dtype='object')

    fig = plt.figure(figsize=(wi, hi))
    left, bottom, right, top = 0.07, 0.2, 0.99, 0.93
    gs = gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[.56, 2.5, 2.5], left=left, bottom=bottom, right=right,
                           top=top, wspace=0.07, hspace=0.4)

    y_tick_labels = [u'5\N{DEGREE SIGN}N', u'10\N{DEGREE SIGN}N', u'15\N{DEGREE SIGN}N', u'20\N{DEGREE SIGN}N',
                     u'25\N{DEGREE SIGN}N',
                     u'30\N{DEGREE SIGN}N', u'35\N{DEGREE SIGN}N', u'40\N{DEGREE SIGN}N', u'45\N{DEGREE SIGN}N',
                     u'50\N{DEGREE SIGN}N']

    # Left plot for geographic reference (makes small map)
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=0))
    ax1.set_extent([i, j, 5, 50], ccrs.PlateCarree(central_longitude=0))
    ax1.set_xticks([i, j])
    ax1.set_xticklabels([f'{i}°E', f'{j}°E'])
    ax1.set_yticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax1.set_yticklabels(y_tick_labels)
    ax1.grid(linestyle='dotted', linewidth=2)

    # Add geopolitical boundaries for map reference
    ax1.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax1.add_feature(cfeature.LAKES.with_scale('50m'), linewidths=0.5)
    ax1.add_feature(cfeature.LAND.with_scale('50m'), edgecolor='k')
    ax1.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')
    ax1.add_feature(cfeature.RIVERS.with_scale('50m'))

    label = lb[k]
    t = ax1.text(0.5, 1.013, f'({label})', ha='center', va='bottom',
                 transform=ax1.transAxes, fontsize=14)

    k = k+1

    # right plot for Hovmoller diagram
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    # Plot of chosen variable averaged over longitude and slightly smoothed
    levels = MaxNLocator(nbins=25).tick_values(0, 25)
    cmap = cmc.davos_r
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    cf2 = ax2.pcolormesh(time1_, lat1_, np.transpose(cpm_ctrl[:, :, 0]), cmap=cmap, norm=norm)
    cf3 = ax3.pcolormesh(time2_, lat2_, np.transpose(cpm_imerg[:, :, 0]), cmap=cmap, norm=norm)

    ct2 = ax2.contour(time1_, lat1_, mpcalc.smooth_n_point(np.transpose(cpm_ctrl[:, :, 0]), 5, 1),
                      levels=[5, 10, 15, 20], colors='k', linewidths=1)
    ct3 = ax3.contour(time2_, lat2_, mpcalc.smooth_n_point(np.transpose(cpm_imerg[:, :, 0]), 5, 1),
                      levels=[5, 10, 15, 20], colors='k', linewidths=1)

    ax2.text(0, 1.01, 'CTRL11', ha='left', va='bottom', transform=ax2.transAxes, fontsize=14)
    ax3.text(0, 1.01, 'IMERG', ha='left', va='bottom', transform=ax3.transAxes, fontsize=14)

    for ax, ct in zip([ax2, ax3], [ct2, ct3]):
        clabel = ax.clabel(ct, inline=True, use_clabeltext=True, fontsize=13)
        for l in clabel:
            l.set_rotation(0)
        [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]

    cax = fig.add_axes(
        [ax2.get_position().x0, ax2.get_position().y0 - 0.13, ax3.get_position().x1 - ax2.get_position().x0, 0.04])
    cbar1 = fig.colorbar(cf2, cax=cax, ticks=np.linspace(0, 24, 13, endpoint=True), orientation='horizontal',
                         extend='max')
    cbar1.ax.tick_params(labelsize=13)
    # cbar1.ax.set_xlabel('mm/day', fontsize=13, labelpad=-0.01)

    # Make some ticks and tick labels
    for ax in ax2, ax3:
        ax.set_xticks(time1[3::6])
        # ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        ax.set_ylim(5, 50)
        ax.set_yticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0.1)
        ax.grid(linestyle='dotted', linewidth=2)

    # fig.suptitle(f'Precipitation Climatology ({i}°-{j}°E)', fontsize=16, fontweight='bold')

    # Set some titles
    # plt.title('250-hPa V-wind', loc='left', fontsize=10)
    # plt.title('Time Range: {0:%Y%m%d %HZ} - {1:%Y%m%d %HZ}'.format(vtimes[0], vtimes[-1]),
    # loc='right', fontsize=10)

    plt.show()
    plotpath = "/project/pr133/rxiang/figure/paper1/validation/LSM/"
    fig.savefig(plotpath + f'hovmoller.{i}°-{j}°E.png', dpi=500)
    plt.close(fig)


