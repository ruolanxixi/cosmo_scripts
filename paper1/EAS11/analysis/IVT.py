# -------------------------------------------------------------------------------
# Plot atmospheric river
# -------------------------------------------------------------------------------
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from plotcosmomap import plotcosmo, pole, colorbar
import pandas as pd
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

ds = xr.open_dataset('/project/pr133/rxiang/data/cosmo/EAS11_ctrl/monsoon/IVT/01-05.IVT.cpm.nc')
iuq = ds['IUQ'].values[:, :, :]
ivq = ds['IVQ'].values[:, :, :]
TQF = np.sqrt(iuq ** 2 + ivq ** 2)
# %%
for i in range(73):
    [pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs] = pole()
    # plot
    fig, ax = plt.subplots(subplot_kw={'projection': rot_pole_crs})
    ax = plotcosmo(ax)
    levels = np.linspace(200, 500, 13, endpoint=True)
    scale = 8000
    date = ds.time[i].values
    date = pd.Timestamp(date)
    month = "{:02}".format(date.month)
    day = "{:02}".format(date.day)
    # cmap1 = plt.cm.get_cmap("Spectral")
    cmap = cmc.roma
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    q = ax.quiver(rlon[::15], rlat[::15], iuq[i, ::15, ::15], ivq[i, ::15, ::15], TQF[i, ::15, ::15],
                  cmap=cmap, norm=norm, scale=scale, headaxislength=3.5, headwidth=5, minshaft=0)
    ax.text(0.0, 1.05, f'{month}/{day}', ha='left', va='center', transform=ax.transAxes, fontsize=11)
    ax.set_title('Integrated Water Vapour Transport', fontweight='bold', pad=7, fontsize=12)
    cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 - 0.1, ax.get_position().width, 0.027])
    cb = fig.colorbar(q, cax=cax, orientation='horizontal',extend='both', ticks=[200, 250, 300, 350, 400, 450, 500])
    cb.set_label('IVT ($kg/m/s$)', fontsize=11)
    # adjust figure
    fig.show()
    # save figure
    plotpath = "/project/pr133/rxiang/figure/paper1/results/IVT/"
    fig.savefig(plotpath + f'{month}{day}.png', dpi=300)
    # plt.close(fig)
