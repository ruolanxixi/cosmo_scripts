# -------------------------------------------------------------------------------
# Plot atmospheric river
# -------------------------------------------------------------------------------
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotcosmomap import plotcosmo, pole, colorbar
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

# -------------------------------------------------------------------------------
# import data
#
mdvnames = ['TWATER', 'U', 'V']  # edit here
year = '2001-2005'
ctrlpath = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/10days/"
topo1path = "/project/pr133/rxiang/data/cosmo/EAS11_topo1/10days/"
sims = ['ctrl', 'topo1']
titles = ['Control', 'Reduced topography']
[pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()
dates = ['01 Apr. - 10 Apr.', '11 Apr. - 20 Apr.', '21 Apr - 30 Apr.',
         '01 May - 10 May', '11 May - 20 May', '21 May - 31 May',
         '01 June - 10 June', '11 June - 20 June', '21 Jun - 30 June',
         '01 July - 10 July', '11 July - 20 July', '21 July - 31 July',
         '01 Aug. - 10 Aug.', '11 Aug. - 20 Aug.', '21 Aug. - 31 Aug.',
         '01 Sep. - 10 Sep.', '11 Sep. - 20 Sep.', '21 Sep. - 30 Sep.']


# -------------------------------------------------------------------------------
def readdata(v, day, s):
    filename = f'{year}.{v}.{day}.nc'
    path = f'/project/pr133/rxiang/data/cosmo/EAS11_{s}/10days/'
    if v in ('U', 'V'):
        d = xr.open_dataset(f'{path}{v}/{filename}')[v].values[0, 0, :, :]
    else:
        d = xr.open_dataset(f'{path}{v}/{filename}')[v].values[0, :, :]
    return d


for i in range(len(sims)):
    sim = sims[i]
    title = titles[i]
    for j in np.arange(1, 19):
        jj = str(j).zfill(2)
        date = dates[j-1]
        # read data
        data = readdata("TWATER", f'{jj}', f'{sim}')
        data_u = readdata("U", f'{jj}', f'{sim}')
        data_v = readdata("V", f'{jj}', f'{sim}')
        w = np.sqrt(data_u**2+data_v**2)
        # plot
        fig, ax = plt.subplots(subplot_kw={'projection': rot_pole_crs})
        ax = plotcosmo(ax)
        levels = MaxNLocator(nbins=20).tick_values(0, 600)
        cmap = cmc.davos_r
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        cs = ax.pcolormesh(rlon, rlat, data, cmap=cmap, norm=norm, shading="auto")
        q = ax.quiver(rlon[::30], rlat[::30], data_u[::30, ::30], data_v[::30, ::30], color='black', scale=150)
        ax.quiverkey(q, 0.95, 1.12, 10, r'$10\ m\ s^{-1}$', labelpos='S', transform=ax.transAxes,
                     fontproperties={'size': 10})
        # if title == 'Control':
            # ax.text(0.055, 1.05, f'{title}', ha='center', va='center', transform=ax.transAxes, fontsize=11)
        # else:
            # ax.text(0.162, 1.05, f'{title}', ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.text(0.125, 1.05, f'{date}', ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title(f'{title}', fontweight='bold', pad=15, fontsize=13)
        cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 - 0.1, ax.get_position().width, 0.027])
        cb = fig.colorbar(cs, cax=cax, orientation='horizontal', extend='max')
        cb.set_label('Total water content ($kg/m^{-2}$/day)', fontsize=11)
        # adjust figure
        fig.show()
        # save figure
        plotpath = "/project/pr133/rxiang/figure/atmosriver2/"
        fig.savefig(plotpath + f'{sim}_{jj}.png', dpi=300)
        plt.close(fig)
