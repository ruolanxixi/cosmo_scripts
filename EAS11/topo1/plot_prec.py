# -------------------------------------------------------------------------------
# Compare
# 1. topography
# 2. Precipitation and wind at 850 hPa
# 3. Vertical velocity and IVT
# 4. Summer temperature and geopotential height at 500 hPa
# -------------------------------------------------------------------------------
# modules
#
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotcosmomap import plotcosmo, pole, colorbar
import cartopy.crs as ccrs
from numpy import inf
import matplotlib.gridspec as gridspec
from plottopo import topo
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mycolor import custom_div_cmap
from pyproj import Transformer
# -------------------------------------------------------------------------------
# import data
#
season = 'JJA'
mdvnames = ['TOT_PREC', 'U', 'V', 'TWATFLXU', 'TWATFLXV', 'W', 'T', 'FI']  # edit here
year = '2001-2005'
ctrlpath = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/szn/"
topo1path = "/project/pr133/rxiang/data/cosmo/EAS11_topo1/szn/"
sims = ['ctrl', 'ctrl']
g = 9.80665

# -------------------------------------------------------------------------------
# read data
#
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()


def read_data(mdvname):
    filename = f'{year}.{mdvname}.{season}.nc'
    if mdvname in ('U', 'V', 'T', 'W', 'FI'):
        data_ctrl = xr.open_dataset(f'{ctrlpath}{mdvname}/{filename}')[mdvname].values[0, 0, :, :]
        data_topo1 = xr.open_dataset(f'{topo1path}{mdvname}/{filename}')[mdvname].values[0, 0, :, :]
        data_diff = data_topo1 - data_ctrl
    else:
        data_ctrl = xr.open_dataset(f'{ctrlpath}{mdvname}/{filename}')[mdvname].values[0, :, :]
        data_topo1 = xr.open_dataset(f'{topo1path}{mdvname}/{filename}')[mdvname].values[0, :, :]
        if mdvname == 'TOT_PREC':
            np.seterr(divide='ignore', invalid='ignore')
            data_diff = (data_topo1 - data_ctrl) / data_topo1 * 100
            data_diff[np.isnan(data_diff)] = 0
            data_diff[data_diff == -inf] = -100
            np.seterr(divide='warn', invalid='warn')
        else:
            data_diff = data_topo1 - data_ctrl
    data_diff = - data_diff
    data = np.dstack((data_ctrl, data_topo1, data_diff))
    da = xr.DataArray(data=data,
                      coords={"rlat": rlat,
                              "rlon": rlon,
                              "sim": ["ctrl", "topo1", "diff"]},
                      dims=["rlat", "rlon", "sim"])

    return da


# -------------------------------------------------------------------------------
# plot
#

#
ar = 1.0  # initial aspect ratio for first trial
hi = 14  # height in inches
wi = hi / ar  # width in inches
# fig = plt.figure(figsize=(wi, hi))
#
ncol = 4  # edit here
nrow = 3

gs1 = gridspec.GridSpec(2, 4)
gs1.update(left=0.05, right=0.99, top=0.96, bottom=0.41, hspace=0.007, wspace=0.1)
gs2 = gridspec.GridSpec(1, 4)
gs2.update(left=0.05, right=0.99, top=0.33, bottom=0.07, wspace=0.1)

fig = plt.figure(figsize=(wi, hi), constrained_layout=True)
# spec = gridspec.GridSpec(ncols=ncol, nrows=nrow, figure=fig)

axs = np.empty(shape=(nrow, ncol), dtype='object')
cs = np.empty(shape=(nrow, ncol), dtype='object')
q = np.empty(shape=(nrow, 2), dtype='object')
ct = np.empty(shape=(nrow, 1), dtype='object')

# -------------------------
# panel plot
for i in (0, 1):
    for j in range(ncol):
        axs[i, j] = plt.subplot(gs1[i, j], projection=rot_pole_crs)
        axs[i, j] = plotcosmo(axs[i, j])

for j in range(ncol):
    axs[2, j] = plt.subplot(gs2[0, j], projection=rot_pole_crs)
    axs[2, j] = plotcosmo(axs[2, j])

# -------------------------
# plot topo
[axs[0, 0], axs[1, 0], axs[2, 0], cs[0, 0], cs[1, 0], cs[2, 0]] = topo(axs[0, 0], axs[1, 0], axs[2, 0])

# -------------------------
# plot precipitation
da = read_data("TOT_PREC")
levels = MaxNLocator(nbins=15).tick_values(0, 20)
cmap = cmc.davos_r
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
cs[0, 1] = axs[0, 1].pcolormesh(rlon, rlat, da.sel(sim='ctrl').values[:, :], cmap=cmap, norm=norm, shading="auto")
cs[1, 1] = axs[1, 1].pcolormesh(rlon, rlat, da.sel(sim='topo1').values[:, :], cmap=cmap, norm=norm, shading="auto")
cmap = custom_div_cmap(27, cmc.vik_r)
cs[2, 1] = axs[2, 1].pcolormesh(rlon, rlat, da.sel(sim='diff').values[:, :], cmap=cmap, clim=(-120, 120), shading="auto")
del da
# add wind at 850 hPa
da_u = read_data("U")
da_v = read_data("V")

q[0, 0] = axs[0, 1].quiver(rlon[::30], rlat[::30], da_u.sel(sim='ctrl').values[::30, ::30],
                           da_v.sel(sim='ctrl').values[::30, ::30], color='black', scale=150)
axs[0, 1].quiverkey(q[0, 0], 0.93, 1.1, 10, r'$10\ m\ s^{-1}$', labelpos='S', transform=axs[0, 0].transAxes,
                     fontproperties={'size': 12})
q[1, 0] = axs[1, 1].quiver(rlon[::30], rlat[::30], da_u.sel(sim='topo1').values[::30, ::30],
                           da_v.sel(sim='topo1').values[::30, ::30], color='black', scale=150)
q[2, 0] = axs[2, 1].quiver(rlon[::30], rlat[::30], da_u.sel(sim='diff').values[::30, ::30],
                           da_v.sel(sim='diff').values[::30, ::30], color='black', scale=50)
axs[2, 1].quiverkey(q[2, 0], 0.95, 1.1, 2, r'$2\ m\ s^{-1}$', labelpos='S', transform=axs[0, 0].transAxes,
                     fontproperties={'size': 12})
del da_u, da_v

# -------------------------
# plot vertical velocity
da = read_data("W")
levels = MaxNLocator(nbins=14).tick_values(-3, 3)
cmap = cmc.broc_r
norm = BoundaryNorm(levels, ncolors=255, clip=True)
cs[0, 2] = axs[0, 2].pcolormesh(rlon, rlat, da.sel(sim='ctrl').values[:, :]*100, cmap=cmap, norm=norm, shading="auto")
cs[1, 2] = axs[1, 2].pcolormesh(rlon, rlat, da.sel(sim='topo1').values[:, :]*100, cmap=cmap, norm=norm, shading="auto")
cmap = custom_div_cmap(27, cmc.vik_r)
cs[2, 2] = axs[2, 2].pcolormesh(rlon, rlat, da.sel(sim='diff').values[:, :]*100, cmap=cmap, clim=(-2, 2), shading="auto")
del da
# add wind at 850 hPa
da_u = read_data("TWATFLXU")
da_v = read_data("TWATFLXV")

q[0, 1] = axs[0, 2].quiver(rlon[::30], rlat[::30], da_u.sel(sim='ctrl').values[::30, ::30],
                           da_v.sel(sim='ctrl').values[::30, ::30], color='black', scale=8000)
axs[0, 2].quiverkey(q[0, 1], 0.9, 1.1, 200, r'$200\ kg\ m^{-1}\ s^{-1}$', labelpos='S', transform=axs[0, 2].transAxes,
                     fontproperties={'size': 12})
q[1, 1] = axs[1, 2].quiver(rlon[::30], rlat[::30], da_u.sel(sim='topo1').values[::30, ::30],
                           da_v.sel(sim='topo1').values[::30, ::30], color='black', scale=8000)
q[2, 1] = axs[2, 2].quiver(rlon[::30], rlat[::30], da_u.sel(sim='diff').values[::30, ::30],
                           da_v.sel(sim='diff').values[::30, ::30], color='black', scale=3000)
axs[2, 2].quiverkey(q[2, 1], .9, 1.1, 100, r'$100\ kg\ m^{-1}\ s^{-1}$', labelpos='S', transform=axs[2, 2].transAxes,
                     fontproperties={'size': 12})
del da_u, da_v

# -------------------------
# plot summer temperature
da = read_data("T")
da_f = read_data("FI")
levels = MaxNLocator(nbins=24).tick_values(-20, 3)
cmap = plt.get_cmap('YlOrRd')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
cs[0, 3] = axs[0, 3].pcolormesh(rlon, rlat, da.sel(sim='ctrl').values[:, :]-273.15, cmap=cmap, norm=norm)
ct[0, 0] = axs[0, 3].contour(rlon, rlat, da_f.sel(sim='ctrl').values[:, :]/g, levels=np.linspace(5600, 5900, 13),
                             colors='k',
                             linewidths=.7)
trans = Transformer.from_proj(ccrs.PlateCarree(), rot_pole_crs, always_xy=True)
x = np.array([120, 146, 130, 125, 105, 115, 147, 110, 110, 160])
y = np.array([60, 53, 50, 43, 47, 40, 38, 35, 27, 18])
loc_lon, loc_lat = trans.transform(x, y)
manual_locations = [i for i in zip(loc_lon, loc_lat)]
axs[0, 3].clabel(ct[0, 0], ct[0, 0].levels[::1], inline=True, fontsize=8, manual=manual_locations)

cs[1, 3] = axs[1, 3].pcolormesh(rlon, rlat, da.sel(sim='topo1').values[:, :]-273.15, cmap=cmap, norm=norm)
ct[1, 0] = axs[1, 3].contour(rlon, rlat, da_f.sel(sim='topo1').values[:, :]/g, levels=np.linspace(5600, 5900, 13),
                             colors='k',
                             linewidths=.7)
axs[1, 3].clabel(ct[1, 0], ct[1, 0].levels[::1], inline=True, fontsize=8, manual=manual_locations)
cmap = custom_div_cmap(27, cmc.vik)
cs[2, 3] = axs[2, 3].pcolormesh(rlon, rlat, da.sel(sim='diff').values[:, :], cmap=cmap, clim=(-1.2, 1.2), shading="auto")
del da, da_f
# -------------------------
# add title
axs[0, 0].set_title("Topography", fontweight='bold', pad=12, fontsize=15)
axs[0, 1].set_title("Precipitation and UV850", fontweight='bold', pad=12, fontsize=15)
axs[0, 2].set_title("Vertical velocity and IVT", fontweight='bold', pad=12, fontsize=15)
axs[0, 3].set_title("Summer temperature at 500 hPa", fontweight='bold', pad=12, fontsize=15)

# -------------------------
# add label
axs[0, 0].text(-0.13, 0.55, 'Control', ha='center', va='center', rotation='vertical',
               transform=axs[0, 0].transAxes, fontsize=15, fontweight='bold')
axs[1, 0].text(-0.13, 0.55, 'Reduced Topography', ha='center', va='center', rotation='vertical',
               transform=axs[1, 0].transAxes, fontsize=15, fontweight='bold')
axs[2, 0].text(-0.13, 0.55, 'Difference', ha='center', va='center', rotation='vertical',
               transform=axs[2, 0].transAxes, fontsize=15, fontweight='bold')

# -------------------------
# adjust figure
xmin, xmax = axs[0, 0].get_xbound()
ymin, ymax = axs[0, 0].get_ybound()
y2x_ratio = (ymax - ymin) / (xmax - xmin) * nrow / ncol * 1.17
fig.set_figwidth(hi / y2x_ratio)

# -------------------------
# add colorbar
cax = colorbar(fig, axs[1, 0], 1)  # edit here
cb1 = fig.colorbar(cs[1, 0], cax=cax, orientation='horizontal')
cb1.ax.tick_params(labelsize=12)
cb1.set_label('m', fontsize=12)
cax = colorbar(fig, axs[2, 0], 1)  # edit here
cb1 = fig.colorbar(cs[2, 0], cax=cax, orientation='horizontal')
cb1.ax.tick_params(labelsize=12)
cb1.set_label('m', fontsize=12)

cax = colorbar(fig, axs[1, 1], 1)  # edit here
cb1 = fig.colorbar(cs[1, 1], cax=cax, orientation='horizontal', extend='max')
cb1.ax.tick_params(labelsize=12)
cb1.set_label('mm/day', fontsize=11)
cax = colorbar(fig, axs[2, 1], 1)  # edit here
cb1 = fig.colorbar(cs[2, 1], cax=cax, orientation='horizontal', extend='both')
cb1.ax.tick_params(labelsize=12)
cb1.set_label('%', fontsize=12)

cax = colorbar(fig, axs[1, 2], 1)  # edit here
cb1 = fig.colorbar(cs[1, 2], cax=cax, orientation='horizontal', extend='both')
cb1.set_label('$100^{-1} m s^{-1}$', fontsize=12)
cax = colorbar(fig, axs[2, 2], 1)  # edit here
cb1 = fig.colorbar(cs[2, 2], cax=cax, orientation='horizontal', extend='both')
cb1.ax.tick_params(labelsize=12)
cb1.set_label('$100^{-1} m s^{-1}$', fontsize=12)

cax = colorbar(fig, axs[1, 3], 1)  # edit here
cb1 = fig.colorbar(cs[1, 3], cax=cax, orientation='horizontal', extend='both')
cb1.ax.tick_params(labelsize=12)
cb1.set_label('$^{o}C$', fontsize=12)
cax = colorbar(fig, axs[2, 3], 1)  # edit here
cb1 = fig.colorbar(cs[2, 3], cax=cax, orientation='horizontal', extend='both')
cb1.ax.tick_params(labelsize=12)
cb1.set_label('$^{o}C$', fontsize=12)
# cax = colorbar(fig, axs[3, 1], 1)
# cb2 = fig.colorbar(cs[3, 1], cax=cax, orientation='horizontal', extend='both')
# # # cb1.set_ticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000])
# cb2.set_label('%')

plt.show()
# -------------------------
# save figure
plotpath = "/project/pr133/rxiang/figure/topo1/"
fig.savefig(plotpath + 'monsoon2.png', dpi=300)
plt.close(fig)
