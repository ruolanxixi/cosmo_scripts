# -------------------------------------------------------------------------------
# modules
#
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
import matplotlib
import matplotlib.colors as colors

font = {'size': 14}
matplotlib.rc('font', **font)

def drywet(numcolors, colormap):

    colors_blue = colormap(np.linspace(0.5, 1, 5))
    colors_white = np.array([1, 1, 1, 1])
    colors_brown = [[84, 48, 5, 255],
                    [140, 81, 10, 255],
                    [191, 129, 45, 255],
                    [223, 194, 125, 255],
                    [246, 232, 195, 255]]
    rgb = []
    for i in range(len(colors_brown)):
        z = [x / 255 for x in colors_brown[i]]
        rgb.append(z)
    colors = np.vstack((rgb, colors_white, colors_blue))

    cmap = LinearSegmentedColormap.from_list(name=colormap, colors=colors, N=numcolors)

    return cmap

# -------------------------------------------------------------------------------
# read data
# %%
sims = ['PI', 'PD', 'LGM', 'PLIO']
mdpath = "/scratch/snx3000/rxiang/echam5"
friac = {}
labels = {'PI': 'Pre-industrial', 'PD': 'Present day (1970-1995)', 'LGM': 'Last glacial maximum', 'PLIO': 'Mid-Pliocene', 'diff': 'PI-LGM'}

for s in range(len(sims)):
    sim = sims[s]
    friac[sim] = {}
    friac[sim]['label'] = labels[sim]
    data = xr.open_dataset(f'{mdpath}/{sim}/analysis/friac/mon/jan.nc')
    dt = data['friac'].values[0, :, :] * 100
    friac[sim]['friac'] = dt

friac['diff'] = {}
friac['diff']['friac'] = friac['PI']['friac'] - friac['LGM']['friac']
friac['diff']['label'] = labels['diff']
# %%
lat = xr.open_dataset(f'{mdpath}/LGM/analysis/friac/mon/jan.nc')['lat'].values[:]
lon = xr.open_dataset(f'{mdpath}/LGM/analysis/friac/mon/jan.nc')['lon'].values[:]
lat_, lon_ = np.meshgrid(lon, lat)
# -------------------------------------------------------------------------------
# plot
# %%
ar = 1.0  # initial aspect ratio for first trial
wi = 10  # width in inches #15
hi = 2.7  # height in inches #10
ncol = 3
nrow = 1
axs, cs, gl = np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object'), np.empty(shape=(nrow, ncol), dtype='object')

cmap1 = cmc.davos_r
levels1 = np.linspace(0, 100, 21, endpoint=True)
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

cmap2 = drywet(25, cmc.vik_r)
levels2 = np.linspace(-100, 100, 21, endpoint=True)
norm2 = colors.TwoSlopeNorm(vmin=-100., vcenter=0., vmax=100.)

# change here the lat and lon
map_ext = [-50, 50, 40, 90]

fig = plt.figure(figsize=(wi, hi))
left, bottom, right, top = 0.08, 0.23, 0.985, 0.99
gs = gridspec.GridSpec(nrows=1, ncols=3, left=left, bottom=bottom, right=right, top=top,
                       wspace=0.1, hspace=0.15)

sims = ['LGM', 'PI', 'diff']

cmaps = [cmap1, cmap1, cmap2]
norms = [norm1, norm1, norm2]
for i in range(3):
    sim = sims[i]
    cmap = cmaps[i]
    norm = norms[i]
    label = friac[sim]['label']
    axs[0, i] = fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree())
    axs[0, i].set_extent(map_ext, crs=ccrs.PlateCarree())
    axs[0, i].coastlines(zorder=3)
    axs[0, i].stock_img()
    gl[0, i] = axs[0, i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False, linewidth=1, color='grey', alpha=0.5, linestyle='--')
    gl[0, i].right_labels = False
    gl[0, i].top_labels = False
    gl[0, i].left_labels = False
    cs[0, i] = axs[0, i].pcolormesh(lon, lat, friac[sim]['friac'], cmap=cmap, norm=norm, shading="auto",
                                    transform=ccrs.PlateCarree())
    axs[0, i].set_title(f'{label}', fontweight='bold', pad=6, fontsize=14, loc='center')

gl[0, 0].left_labels = True

cax = fig.add_axes(
    [axs[0, 0].get_position().x0 + 0.16, axs[0, 1].get_position().y0 - 0.15, axs[0, 1].get_position().width, 0.05])
cbar = fig.colorbar(cs[0, 1], cax=cax, orientation='horizontal',
                    ticks=np.linspace(0, 100, 6, endpoint=True))
cbar.set_label('[%]')
cbar.ax.tick_params(labelsize=14)

cax = fig.add_axes(
    [axs[0, 2].get_position().x0, axs[0, 2].get_position().y0 - 0.15, axs[0, 2].get_position().width, 0.05])
cbar = fig.colorbar(cs[0, 2], cax=cax, orientation='horizontal',
                    ticks=np.linspace(-100, 100, 5, endpoint=True))
cbar.set_label('[%]')
cbar.ax.tick_params(labelsize=14)


axs[0, 0].text(-0.23, 0.5, 'Sea ice', ha='center', va='center', rotation='vertical',
               transform=axs[0, 0].transAxes, fontsize=14, fontweight='bold')
# axs[0, 2].text(1.07, 1.09, '[%]', ha='center', va='center', rotation='horizontal',
#                transform=axs[0, 2].transAxes, fontsize=12)

fig.show()
# plotpath = "/project/pr133/rxiang/figure/echam5/"
# fig.savefig(plotpath + 'friac' + f'{mon}.png', dpi=500)
plt.close(fig)






