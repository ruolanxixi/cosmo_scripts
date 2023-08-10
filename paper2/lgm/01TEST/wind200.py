# -------------------------------------------------------------------------------
# modules
#
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.colors as colors
from copy import copy
from plotcosmomap import plotcosmo_notick, pole, colorbar
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import cmcrameri.cm as cmc
from auxiliary import read_topo
import matplotlib.gridspec as gridspec
from pyproj import Transformer
from mycolor import custom_div_cmap, drywet
from numpy import inf

# -------------------------------------------------------------------------------
# import data
#
season = 'JJA'
mdvname = 'U'
year = '01'
ctrlpath = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/szn"
lgmpath = "/scratch/snx3000/rxiang/data/cosmo/EAS11_lgm/szn"
g = 9.80665

# -------------------------------------------------------------------------------
# %% read data
#
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()

u200_ctrl = xr.open_dataset(f'{ctrlpath}/{mdvname}/{year}_{mdvname}_{season}.nc')[mdvname].values[0, 1, :, :]
u200_lgm = xr.open_dataset(f'{lgmpath}/{mdvname}/{year}_{mdvname}_{season}.nc')[mdvname].values[0, 1, :, :]

mdvname = 'V'
v200_ctrl = xr.open_dataset(f'{ctrlpath}/{mdvname}/{year}_{mdvname}_{season}.nc')[mdvname].values[0, 1, :, :]
v200_lgm = xr.open_dataset(f'{lgmpath}/{mdvname}/{year}_{mdvname}_{season}.nc')[mdvname].values[0, 1, :, :]

ws200_ctrl = np.sqrt(u200_ctrl**2 + v200_ctrl**2)
ws200_lgm = np.sqrt(u200_lgm**2 + v200_lgm**2)

u200_diff = u200_lgm - u200_ctrl
v200_diff = v200_lgm - v200_ctrl
ws200_diff = ws200_lgm - ws200_ctrl
# -------------------------------------------------------------------------------
# %%
nrow=1
ncol=3
gs = gridspec.GridSpec(1, 3)
gs.update(left=0.043, right=0.99, top=0.98, bottom=0.18, hspace=0.1, wspace=0.06)

fig = plt.figure(figsize=(12, 3.3), constrained_layout=True)
axs = np.empty(shape=(1, 3), dtype='object')
cs = np.empty(shape=(1, 3), dtype='object')
q = np.empty(shape=(1, 3), dtype='object')

axs[0, 0] = plt.subplot(gs[0], projection=rot_pole_crs)
axs[0, 0] = plotcosmo_notick(axs[0, 0])
axs[0, 1] = plt.subplot(gs[1], projection=rot_pole_crs)
axs[0, 1] = plotcosmo_notick(axs[0, 1])
axs[0, 2] = plt.subplot(gs[2], projection=rot_pole_crs)
axs[0, 2] = plotcosmo_notick(axs[0, 2])

# plot data
levels = MaxNLocator(nbins=15).tick_values(5, 20)
# cmap1 = plt.cm.get_cmap("Spectral")
cmap = cmc.roma_r
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

cs[0, 0] = axs[0, 0].streamplot(rlon, rlat, u200_ctrl, v200_ctrl, color=ws200_ctrl,
                                    density=1, cmap=cmap, norm=norm)
cs[0, 1] = axs[0, 1].streamplot(rlon, rlat, u200_lgm, v200_lgm, color=ws200_lgm,
                                    density=1, cmap=cmap, norm=norm)

levels = MaxNLocator(nbins=15).tick_values(-2, 2)
cmap = custom_div_cmap(25, cmc.vik)
norm = colors.TwoSlopeNorm(vmin=-2., vcenter=0., vmax=2.)
cs[0, 2] = axs[0, 2].streamplot(rlon, rlat, u200_diff, v200_diff, color=ws200_diff,
                                    density=1, cmap=cmap, norm=norm)

# q[0, 0] = axs[0, 0].quiver(rlon[::30], rlat[::30], da_u.sel(sim='ctrl').values[::30, ::30],
#                            da_v.sel(sim='ctrl').values[::30, ::30], color='black', scale=150)
# axs[0, 0].quiverkey(q[0, 0], 0.92, 1.12, 10, r'$10\ m\ s^{-1}$', labelpos='S', transform=axs[0, 0].transAxes,
#                      fontproperties={'size': 11})
# q[0, 1] = axs[0, 1].quiver(rlon[::30], rlat[::30], da_u.sel(sim='topo1').values[::30, ::30],
#                            da_v.sel(sim='topo1').values[::30, ::30], color='black', scale=150)
# axs[0, 1].quiverkey(q[0, 1], 0.92, 1.12, 10, r'$10\ m\ s^{-1}$', labelpos='S', transform=axs[0, 1].transAxes,
#                      fontproperties={'size': 11})
# q[0, 2] = axs[0, 2].quiver(rlon[::30], rlat[::30], da_u.sel(sim='diff').values[::30, ::30],
#                            da_v.sel(sim='diff').values[::30, ::30], color='black', scale=50)
# axs[0, 2].quiverkey(q[0, 2], 0.94, 1.12, 2, r'$2\ m\ s^{-1}$', labelpos='S', transform=axs[0, 2].transAxes,
#                      fontproperties={'size': 11})

axs[0, 0].set_title('PD', fontweight='bold', pad=8, fontsize=13)
axs[0, 1].set_title('LGM', fontweight='bold', pad=8, fontsize=13)
axs[0, 2].set_title('LGM - PD', fontweight='bold', pad=8, fontsize=13)

cax1 = fig.add_axes(
    [axs[0, 0].get_position().x0+0.17, axs[0, 0].get_position().y0 - 0.12, axs[0, 0].get_position().width, 0.035])
cb1 = fig.colorbar(cs[0, 0].lines, cax=cax1, orientation='horizontal', extend='both', ticks=[6, 10, 14, 18])
cb1.set_label('m/s', fontsize=11)
cb1.ax.tick_params(labelsize=11)
cax2 = fig.add_axes(
    [axs[0, 2].get_position().x0, axs[0, 2].get_position().y0 - .12, axs[0, 2].get_position().width, 0.035])
cb2 = fig.colorbar(cs[0, 2].lines, cax=cax2, orientation='horizontal', extend='both', ticks=[-2, -1, 0, 1, 2])
cb2.set_label('m/s', fontsize=11)
cb2.ax.tick_params(labelsize=11)

for i in range(nrow):
    axs[i, 0].text(-0.008, 0.95, '50°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.77, '40°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.59, '30°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.41, '20°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.23, '10°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)
    axs[i, 0].text(-0.008, 0.05, '0°N', ha='right', va='center', transform=axs[i, 0].transAxes, fontsize=13)

for j in range(ncol):
    axs[nrow-1, j].text(0.12, -0.02, '80°E', ha='center', va='top', transform=axs[nrow-1, j].transAxes, fontsize=13)
    axs[nrow-1, j].text(0.32, -0.02, '100°E', ha='center', va='top', transform=axs[nrow-1, j].transAxes, fontsize=13)
    axs[nrow-1, j].text(0.52, -0.02, '120°E', ha='center', va='top', transform=axs[nrow-1, j].transAxes, fontsize=13)
    axs[nrow-1, j].text(0.72, -0.02, '140°E', ha='center', va='top', transform=axs[nrow-1, j].transAxes, fontsize=13)
    axs[nrow-1, j].text(0.92, -0.02, '160°E', ha='center', va='top', transform=axs[nrow-1, j].transAxes, fontsize=13)

plt.show()
plotpath = "/project/pr133/rxiang/figure/paper2/test/"
fig.savefig(plotpath + 'wind200.png', dpi=300)
plt.close(fig)

# fig = plt.figure(figsize=(6, 4.5))
# left, bottom, right, top = 0.09, 0.13, 0.99, 0.95
# gs = gridspec.GridSpec(1, 1, left=left, bottom=bottom, right=right, top=top, wspace=0.12, hspace=0.1)
# ax = fig.add_subplot(gs[0], projection=rot_pole_crs)
# ax=plotcosmo(ax)
#
# cmap = drywet(27, cmc.vik_r)
#
# cs = ax.pcolormesh(rlon, rlat, -da.sel(sim='diff').values[:, :], cmap=cmap, clim=(-10, 10), shading="auto")
# # q = ax.quiver(rlon[::30], rlat[::30], -da_u.sel(sim='diff').values[::30, ::30],
#                            # -da_v.sel(sim='diff').values[::30, ::30], color='black', scale=50)
# # qk = ax.quiverkey(q, 0.92, 0.03, 2, r'$2\ \frac{m}{s}$', labelpos='E', transform=ax.transAxes,
#                      # fontproperties={'size': 9})
# ax.text(0, 1.02, 'Reduced topography - Control', ha='left', va='bottom', transform=ax.transAxes, fontsize=11)
# ax.text(1, 1.02, '2001-2005 JJA', ha='right', va='bottom', transform=ax.transAxes, fontsize=11)
# ax.set_title('Difference in Precipitation', fontweight='bold', pad=24, fontsize=12)
# cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 - 0.1, ax.get_position().width, 0.03])
# cb = fig.colorbar(cs, cax=cax, orientation='horizontal', extend='both')
# cb.ax.tick_params(labelsize=11)
# cb.set_label('mm/day', fontsize=11)
# # adjust figure
# fig.show()
# # save figure
# plotpath = "/project/pr133/rxiang/figure/EAS11/analysis/JJA/topo1/"
# fig.savefig(plotpath + 'tmp_diff.png', dpi=300)
# plt.close(fig)
