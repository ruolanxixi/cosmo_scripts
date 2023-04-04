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

lonmin = [70, 85, 95, 110, 130]
lonmax = [80, 95, 105, 120, 140]

# lonmin = [70]
# lonmax = [80]

for i, j in zip(lonmin, lonmax):
    data = xr.open_dataset('/project/pr133/rxiang/data/cosmo/EAS11_ctrl/monsoon/TOT_PREC/hovmoller/' + f'01-05.TOT_PREC.cpm.{i}-{j}.nc')
    cpm_ctrl = data['TOT_PREC'].values[...]
    data = xr.open_dataset(f'/project/pr133/rxiang/data/cosmo/EAS11_topo1/monsoon/TOT_PREC/hovmoller/' + f'01-05.TOT_PREC.cpm.{i}-{j}.nc')
    cpm_topo1 = data['TOT_PREC'].values[...]

    lat = data['lat'].values[...]
    time = data['time'].values[...]

    time_, lat_ = np.meshgrid(time, lat)

    # plot
    ar = 3.3  # initial aspect ratio for first trial
    hi = 5.8  # height in inches
    wi = hi * ar  # height in inches
    ncol = 4  # edit here
    nrow = 1
    axs, cs, ct = np.empty(4, dtype='object'), np.empty(4, dtype='object'), np.empty(4, dtype='object')

    fig = plt.figure(figsize=(wi, hi))
    left, bottom, right, top = 0.035, 0.18, 0.99, 0.89
    gs = gridspec.GridSpec(nrows=1, ncols=4, width_ratios=[.75, 4, 4, 4], left=left, bottom=bottom, right=right, top=top, wspace=0.2, hspace=0.1)

    y_tick_labels = [u'5\N{DEGREE SIGN}N', u'10\N{DEGREE SIGN}N', u'15\N{DEGREE SIGN}N', u'20\N{DEGREE SIGN}N', u'25\N{DEGREE SIGN}N',
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

    # right plot for Hovmoller diagram
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    # Plot of chosen variable averaged over longitude and slightly smoothed
    levels = MaxNLocator(nbins=25).tick_values(0, 25)
    cmap = cmc.davos_r
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    cf2 = ax2.pcolormesh(time_, lat_, np.transpose(cpm_ctrl[:, :, 0]), cmap=cmap, norm=norm)
    cf3 = ax3.pcolormesh(time_, lat_, np.transpose(cpm_topo1[:, :, 0]), cmap=cmap, norm=norm)

    levels = MaxNLocator(nbins=23).tick_values(-10, 10)
    cmap = drywet(25, cmc.vik_r)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    cf4 = ax4.pcolormesh(time_, lat_, np.transpose(cpm_topo1[:, :, 0]) - np.transpose(cpm_ctrl[:, :, 0]), cmap=cmap, clim=(-10, 10))
    ct2 = ax2.contour(time_, lat_, mpcalc.smooth_n_point(np.transpose(cpm_ctrl[:, :, 0]), 5, 1),
                      levels=[5, 10, 15, 20], colors='k', linewidths=1)
    ct3 = ax3.contour(time_, lat_, mpcalc.smooth_n_point(np.transpose(cpm_topo1[:, :, 0]), 5, 1),
                      levels=[5, 10, 15, 20], colors='k', linewidths=1)
    ct4 = ax4.contour(time_, lat_, mpcalc.smooth_n_point(np.transpose(cpm_topo1[:, :, 0]) - np.transpose(cpm_ctrl[:, :, 0]), 9, 1),
                      colors='k', linewidths=1)

    ax2.text(0, 1.01, 'CTRL', ha='left', va='bottom', transform=ax2.transAxes, fontsize=14)
    ax3.text(0, 1.01, 'TRED', ha='left', va='bottom', transform=ax3.transAxes, fontsize=14)
    ax4.text(0, 1.01, 'TRED - CTRL', ha='left', va='bottom', transform=ax4.transAxes, fontsize=14)

    for ax, ct in zip([ax2, ax3, ax4], [ct2, ct3, ct4]):
        clabel = ax.clabel(ct, inline=True, use_clabeltext=True, fontsize=13)
        for l in clabel:
            l.set_rotation(0)
        [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]

    cax = fig.add_axes([ax2.get_position().x0, ax2.get_position().y0 - 0.09, ax2.get_position().width*2+0.04, 0.02])
    cbar1 = fig.colorbar(cf2, cax=cax, ticks=np.linspace(0, 24, 13, endpoint=True), orientation='horizontal', extend='max')
    cbar1.ax.tick_params(labelsize=13)
    cbar1.ax.set_xlabel('mm day$^{-1}$', fontsize=13, labelpad=-0.01)

    cax = fig.add_axes(
        [ax4.get_position().x0, ax4.get_position().y0 - 0.09, ax4.get_position().width, 0.02])
    cbar2 = fig.colorbar(cf4, cax=cax, ticks=np.linspace(-10, 10, 11, endpoint=True), orientation='horizontal', extend='both')
    cbar2.ax.tick_params(labelsize=13)
    cbar2.ax.set_xlabel('mm day$^{-1}$', fontsize=13, labelpad=-0.01)

    # Make some ticks and tick labels
    for ax in ax2, ax3, ax4:
        ax.set_xticks(time[3::6])
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.set_ylim(5, 50)
        ax.set_yticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
        ax.set_yticklabels(y_tick_labels)
        ax.grid(linestyle='dotted', linewidth=2)

    fig.suptitle(f'Precipitation Climatology ({i}°-{j}°E)', fontsize=16, fontweight='bold')

    # Set some titles
    # plt.title('250-hPa V-wind', loc='left', fontsize=10)
    # plt.title('Time Range: {0:%Y%m%d %HZ} - {1:%Y%m%d %HZ}'.format(vtimes[0], vtimes[-1]),
              # loc='right', fontsize=10)

    # plt.show()
    plotpath = "/project/pr133/rxiang/figure/analysis/EAS11/topo1/hovmoller/"
    fig.savefig(plotpath + f'{i}°-{j}°E.png', dpi=500)
    plt.close(fig)












