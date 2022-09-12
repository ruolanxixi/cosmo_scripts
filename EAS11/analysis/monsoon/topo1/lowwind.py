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
from mycolor import custom_div_cmap
import matplotlib.colors as colors

# -------------------------------------------------------------------------------
# read data
#
font = {'size': 14}
matplotlib.rc('font', **font)

data = xr.open_dataset('/project/pr133/rxiang/data/cosmo/EAS11_ctrl/monsoon/U/' + '01-05.U.olr.5-35.nc')
u_ctrl = data['U'].values[...]
data = xr.open_dataset('/project/pr133/rxiang/data/cosmo/EAS11_topo1/monsoon/U/' + '01-05.U.olr.5-35.nc')
u_topo1 = data['U'].values[...]

data = xr.open_dataset('/project/pr133/rxiang/data/cosmo/EAS11_ctrl/monsoon/V/' + '01-05.V.olr.5-35.nc')
v_ctrl = data['V'].values[...]
data = xr.open_dataset('/project/pr133/rxiang/data/cosmo/EAS11_topo1/monsoon/V/' + '01-05.V.olr.5-35.nc')
v_topo1 = data['V'].values[...]

data = xr.open_dataset('/project/pr133/rxiang/data/cosmo/EAS11_ctrl/monsoon/ATHB_T/' + '01-05.ATHB_T.olr.5-35.nc')
olr_ctrl = data['ATHB_T'].values[...]
data = xr.open_dataset('/project/pr133/rxiang/data/cosmo/EAS11_topo1/monsoon/ATHB_T/' + '01-05.ATHB_T.olr.5-35.nc')
olr_topo1 = data['ATHB_T'].values[...]

lon = data['lon'].values[...]
time = data['time'].values[...]

time_, lon_ = np.meshgrid(time, lon)

# -------------------------------------------------------------------------------
# plot
#
ar = 2.2  # initial aspect ratio for first trial
wi = 6.5  # width in inches
hi = wi * ar  # height in inches
ncol = 1  # edit here
nrow = 4
axs, cs, ct = np.empty(4, dtype='object'), np.empty(4, dtype='object'), np.empty(4, dtype='object')

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.13, 0.1, 0.99, 0.95
gs = gridspec.GridSpec(nrows=4, ncols=1, height_ratios=[1.2, 4, 4, 4], left=left, bottom=bottom, right=right, top=top,
                       wspace=0.2, hspace=0.3)

x_tick_labels = [u'80\N{DEGREE SIGN}E', u'100\N{DEGREE SIGN}E',
                 u'120\N{DEGREE SIGN}E', u'140\N{DEGREE SIGN}E',
                 u'160\N{DEGREE SIGN}E']

# Top plot for geographic reference (makes small map)
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=0))
ax1.set_extent([70, 170, 5, 20], ccrs.PlateCarree(central_longitude=0))
ax1.set_xticks(np.linspace(80, 160, 5, endpoint=True))
ax1.set_xticklabels(x_tick_labels)
ax1.set_yticks([5, 10, 15, 20])
ax1.set_yticklabels(['5°N', '10°N', '15°N', '20°N'])
ax1.grid(linestyle='dotted', linewidth=2)

# Add geopolitical boundaries for map reference
ax1.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax1.add_feature(cfeature.LAKES.with_scale('50m'), linewidths=0.5)
ax1.add_feature(cfeature.LAND.with_scale('50m'), edgecolor='k')
ax1.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')
ax1.add_feature(cfeature.RIVERS.with_scale('50m'))

# right plot for Hovmoller diagram
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[3, 0])
# Plot of chosen variable averaged over longitude and slightly smoothed
cmap = cmc.davos_r
divnorm = colors.TwoSlopeNorm(vmin=-30, vcenter=0., vmax=60)

cf2 = ax2.pcolormesh(lon_, time_, np.transpose(olr_ctrl[:, :, 0]), cmap=cmap, norm=divnorm)
cf3 = ax3.pcolormesh(lon_, time_, np.transpose(olr_topo1[:, :, 0]), cmap=cmap, norm=divnorm)

levels = MaxNLocator(nbins=23).tick_values(-30, 30)
cmap = custom_div_cmap(25, cmc.vik_r)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
cf4 = ax4.pcolormesh(lon_, time_, np.transpose(olr_ctrl[:, :, 0]) - np.transpose(olr_topo1[:, :, 0]), cmap=cmap, norm=norm)
ct2 = ax2.contour(lon_, time_, mpcalc.smooth_n_point(np.transpose(olr_ctrl[:, :, 0]), 5, 1),
                  colors='k', linewidths=1)
ct3 = ax3.contour(lon_, time_, mpcalc.smooth_n_point(np.transpose(olr_topo1[:, :, 0]), 5, 1),
                  colors='k', linewidths=1)
ct4 = ax4.contour(lon_, time_, mpcalc.smooth_n_point(np.transpose(olr_ctrl[:, :, 0]) - np.transpose(olr_topo1[:, :, 0]), 9, 1),
                  colors='k', linewidths=1)

q2 = ax2.quiver(lon_[::70, 11:44], time_[::70, 11:44], np.transpose(u_ctrl[11:44, 0, 0, ::70]),
                np.transpose(v_ctrl[11:44, 0, 0, ::70]), color='black', scale=150)
ax2.quiverkey(q2, 0.94, 1.12, 10, r'$10\ m\ s^{-1}$', labelpos='S', transform=ax2.transAxes, labelsep=0.03,
              fontproperties={'size': 11})
q3 = ax3.quiver(lon_[::70, 11:44], time_[::70, 11:44], np.transpose(u_topo1[11:44, 0, 0, ::70]),
                np.transpose(v_topo1[11:44, 0, 0, ::70]), color='black', scale=150)
ax3.quiverkey(q3, 0.94, 1.12, 10, r'$10\ m\ s^{-1}$', labelpos='S', transform=ax3.transAxes, labelsep=0.03,
              fontproperties={'size': 11})
q4 = ax4.quiver(lon_[::70, 11:44], time_[::70, 11:44],
                np.transpose(u_ctrl[11:44, 0, 0, ::70]) - np.transpose(u_topo1[11:44, 0, 0, ::70]),
                np.transpose(v_ctrl[11:44, 0, 0, ::70]) - np.transpose(v_topo1[11:44, 0, 0, ::70]),
                color='black', scale=50)
ax4.quiverkey(q4, 0.95, 1.12, 1, r'$1\ m\ s^{-1}$', labelpos='S', transform=ax4.transAxes, labelsep=0.03,
              fontproperties={'size': 11})

ax2.text(0, 1.01, 'Control', ha='left', va='bottom', transform=ax2.transAxes, fontsize=14)
ax3.text(0, 1.01, 'Reduced topography', ha='left', va='bottom', transform=ax3.transAxes, fontsize=14)
ax4.text(0, 1.01, 'Control - Reduced topography', ha='left', va='bottom', transform=ax4.transAxes, fontsize=14)

for ax in ax2, ax3, ax4:
    ax.set_yticks(time[12:43][::3])
    ax.set_yticklabels(['1 Mar', '16 Mar', '1 Apr', '16 Apr', '1 May', '16 May', '1 Jun', '16 Jun', '1 Jul', '16 Jul', '1 Aug'])
    ax.set_xlim(70, 170)
    ax.set_xticks(np.linspace(80, 160, 5, endpoint=True))
    ax.set_xticklabels(x_tick_labels)
    ax.invert_yaxis()
    # ax.grid(linestyle='dotted', linewidth=2)

fig.suptitle('Pentad mean wind at 850 hPa (5°-20°N)', fontsize=16, fontweight='bold')

plt.show()
