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
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from mycolor import custom_div_cmap
import matplotlib.colors as colors
import matplotlib.colors as mcolors

# -------------------------------------------------------------------------------
# read data
#
font = {'size': 14}
matplotlib.rc('font', **font)

data = xr.open_dataset('/scratch/snx3000/rxiang/data/cosmo/EAS11_ctrl/monsoon/uv/' + '01-05.uv.85000.cpm.5-20.nc')
u_ctrl = data['ugeo'].values[...]
v_ctrl = data['vgeo'].values[...]
data = xr.open_dataset('/scratch/snx3000/rxiang/data/cosmo/EAS11_topo1/monsoon/uv/' + '01-05.uv.85000.cpm.5-20.nc')
u_topo1 = data['ugeo'].values[...]
v_topo1 = data['vgeo'].values[...]

# data = xr.open_dataset('/project/pr133/rxiang/data/cosmo/EAS11_ctrl/monsoon/V/' + '01-05.V.cpm.5-20.nc')
# v_ctrl = data['V'].values[...]
# data = xr.open_dataset('/project/pr133/rxiang/data/cosmo/EAS11_topo1/monsoon/V/' + '01-05.V.cpm.5-20.nc')
# v_topo1 = data['V'].values[...]

data = xr.open_dataset('/scratch/snx3000/rxiang/data/cosmo/EAS11_ctrl/monsoon/ATHB_T/' + '01-05.ATHB_T.cpm.5-20.nc')
olr_ctrl = -data['ATHB_T'].values[...]
data = xr.open_dataset('/scratch/snx3000/rxiang/data/cosmo/EAS11_topo1/monsoon/ATHB_T/' + '01-05.ATHB_T.cpm.5-20.nc')
olr_topo1 = -data['ATHB_T'].values[...]

lon = data['lon'].values[...]
time = data['time'].values[...]

time_, lon_ = np.meshgrid(time, lon)

# -------------------------------------------------------------------------------
# %% plot
#
ar = 2.2  # initial aspect ratio for first trial
wi = 6.5  # width in inches
hi = wi * ar  # height in inches
ncol = 1  # edit here
nrow = 4
axs, cs, ct = np.empty(4, dtype='object'), np.empty(4, dtype='object'), np.empty(4, dtype='object')

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.13, 0.4, 0.99, 0.95
gs1 = gridspec.GridSpec(nrows=3, ncols=1, height_ratios=[1.2, 4, 4], left=left, bottom=bottom, right=right, top=top,
                       wspace=0.2, hspace=0.3)
left, bottom, right, top = 0.13, 0.09, 0.99, 0.29
gs2 = gridspec.GridSpec(nrows=1, ncols=1, left=left, bottom=bottom, right=right, top=top,
                       wspace=0.2, hspace=0.3)


x_tick_labels = [u'80\N{DEGREE SIGN}E', u'100\N{DEGREE SIGN}E',
                 u'120\N{DEGREE SIGN}E', u'140\N{DEGREE SIGN}E',
                 u'160\N{DEGREE SIGN}E']

# Top plot for geographic reference (makes small map)
ax1 = fig.add_subplot(gs1[0, 0], projection=ccrs.PlateCarree(central_longitude=0))
ax1.set_extent([70, 170, 5, 20], ccrs.PlateCarree(central_longitude=0))
ax1.set_xticks(np.linspace(80, 160, 5, endpoint=True))
ax1.set_xticklabels(x_tick_labels)
ax1.set_yticks([5, 10, 15, 20])
# ax1.set_yticks([25, 30, 35])
ax1.set_yticklabels(['5°N', '10°N', '15°N', '20°N'])
ax1.grid(linestyle='dotted', linewidth=2)

# Add geopolitical boundaries for map reference
ax1.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax1.add_feature(cfeature.LAKES.with_scale('50m'), linewidths=0.5)
ax1.add_feature(cfeature.LAND.with_scale('50m'), edgecolor='k')
ax1.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')
ax1.add_feature(cfeature.RIVERS.with_scale('50m'))

# right plot for Hovmoller diagram
ax2 = fig.add_subplot(gs1[1, 0])
ax3 = fig.add_subplot(gs1[2, 0])
ax4 = fig.add_subplot(gs2[0, 0])
# Plot of chosen variable averaged over longitude and slightly smoothed
levels = MaxNLocator(nbins=18).tick_values(150, 240)

color1 = cmc.roma(np.linspace(0, 1, 20))
white = np.array([1, 1, 1, 1])
colorss = np.vstack((color1, white))
cmap = LinearSegmentedColormap.from_list(name=white, colors=colorss, N=18)

# cmap = cmc.roma

norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

cf2 = ax2.pcolormesh(lon_[:, 11:44], time_[:, 11:44], np.transpose(olr_ctrl[11:44, 0, :]), cmap=cmap, norm=norm)
cf3 = ax3.pcolormesh(lon_[:, 11:44], time_[:, 11:44], np.transpose(olr_topo1[11:44, 0, :]), cmap=cmap, norm=norm)

cmap = custom_div_cmap(25, cmc.vik_r)
divnorm = colors.TwoSlopeNorm(vmin=-50, vcenter=0., vmax=50)
cf4 = ax4.pcolormesh(lon_[:, 11:44], time_[:, 11:44], np.transpose(olr_topo1[11:44, 0, :]) - np.transpose(olr_ctrl[11:44, 0, :]), cmap=cmap, norm=divnorm)
ct2 = ax2.contour(lon_[:, 11:44], time_[:, 11:44], mpcalc.smooth_n_point(np.transpose(olr_ctrl[11:44, 0, :]), 5, 1),
                  levels=[180, 200, 220, 240], colors='k', linewidths=1)
ct3 = ax3.contour(lon_[:, 11:44], time_[:, 11:44], mpcalc.smooth_n_point(np.transpose(olr_topo1[11:44, 0, :]), 5, 1),
                  levels=[180, 200, 220, 240], colors='k', linewidths=1)
ct4 = ax4.contour(lon_[:, 11:44], time_[:, 11:44], mpcalc.smooth_n_point(np.transpose(olr_topo1[11:44, 0, :]) - np.transpose(olr_ctrl[11:44, 0, :]), 9, 1),
                  levels=[-16., -8, 8, 16], colors='k', linewidths=1)

for ax, ct in zip([ax2, ax3, ax4], [ct2, ct3, ct4]):
    clabel = ax.clabel(ct, inline=True, use_clabeltext=True, fontsize=13)
    for l in clabel:
        l.set_rotation(0)
    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0, alpha=0.5)) for txt in clabel]

cax = fig.add_axes([ax3.get_position().x0, ax3.get_position().y0 - 0.045, ax3.get_position().width, 0.015])
cbar1 = fig.colorbar(cf3, cax=cax, ticks=np.linspace(150, 240, 4, endpoint=True), orientation='horizontal', extend='both')
cbar1.ax.tick_params(labelsize=13)
cbar1.ax.set_xlabel('W m$^{-2}$', fontsize=13, labelpad=-0.01)

cax = fig.add_axes(
    [ax4.get_position().x0, ax4.get_position().y0 - 0.045, ax4.get_position().width, 0.015])
cbar2 = fig.colorbar(cf4, cax=cax, ticks=np.linspace(-50, 50, 11, endpoint=True), orientation='horizontal', extend='both')
cbar2.ax.tick_params(labelsize=13)
cbar2.ax.set_xlabel('W m$^{-2}$', fontsize=13, labelpad=-0.01)

q2 = ax2.quiver(lon_[::70, 11:44], time_[::70, 11:44], np.transpose(u_ctrl[11:44, 0, 0, ::70]),
                np.transpose(v_ctrl[11:44, 0, 0, ::70]), color='black', scale=180)
ax2.quiverkey(q2, 0.87, 1.04, 10, r'$10\ m\ s^{-1}$', labelpos='E', transform=ax2.transAxes, labelsep=0.03,
              fontproperties={'size': 11})
q3 = ax3.quiver(lon_[::70, 11:44], time_[::70, 11:44], np.transpose(u_topo1[11:44, 0, 0, ::70]),
                np.transpose(v_topo1[11:44, 0, 0, ::70]), color='black', scale=180)
ax3.quiverkey(q3, 0.87, 1.04, 10, r'$10\ m\ s^{-1}$', labelpos='E', transform=ax3.transAxes, labelsep=0.03,
              fontproperties={'size': 11})
q4 = ax4.quiver(lon_[::70, 11:44], time_[::70, 11:44],
                np.transpose(u_topo1[11:44, 0, 0, ::70]) - np.transpose(u_ctrl[11:44, 0, 0, ::70]),
                np.transpose(v_topo1[11:44, 0, 0, ::70]) - np.transpose(v_ctrl[11:44, 0, 0, ::70]),
                color='black', scale=30)
ax4.quiverkey(q4, 0.89, 1.04, 1, r'$1\ m\ s^{-1}$', labelpos='E', transform=ax4.transAxes, labelsep=0.03,
              fontproperties={'size': 11})

ax2.text(0, 1.01, 'CTRL', ha='left', va='bottom', transform=ax2.transAxes, fontsize=14)
ax3.text(0, 1.01, 'TRED', ha='left', va='bottom', transform=ax3.transAxes, fontsize=14)
ax4.text(0, 1.01, 'TRED - CTRL', ha='left', va='bottom', transform=ax4.transAxes, fontsize=14)

for ax in ax2, ax3, ax4:
    ax.set_yticks(time[12:43][::3])
    ax.set_yticklabels(['1 Mar', '16 Mar', '1 Apr', '16 Apr', '1 May', '16 May', '1 Jun', '16 Jun', '1 Jul', '16 Jul', '1 Aug'])
    ax.set_xlim(70, 170)
    ax.set_xticks(np.linspace(80, 160, 5, endpoint=True))
    ax.set_xticklabels(x_tick_labels)
    ax.invert_yaxis()
    # ax.grid(linestyle='dotted', linewidth=2)

# fig.suptitle('Pentad mean wind and OLR at 850 hPa (5°-20°N)', fontsize=16, fontweight='bold')

plt.show()

# plotpath = "/project/pr133/rxiang/figure/analysis/EAS11/topo1/"
# fig.savefig(plotpath + 'lw_5-20.png', dpi=500)
# plt.close(fig)
