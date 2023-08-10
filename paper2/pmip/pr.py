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
import matplotlib.ticker as mticker
import cartopy.feature as cfeature
from mycolor import custom_div_cmap, cbr_wet, cbr_drywet, drywet, custom_seq_cmap_

sim = 'midPliocene'

if sim == 'midPliocene':
    dir = 'midPlio'
    model = ['CESM2', 'EC-Earth3-LR', 'GISS-E2-1-G', 'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 'NorESM1-F']
elif sim == 'lgm':
    dir = 'lgm'
    model = ['AWI-ESM-1-1-LR', 'INM-CM4-8', 'MPI-ESM1-2-LR', 'CESM2-WACCM-FV2', 'MIROC-ES2L']

ds = xr.open_dataset(f'/project/pr133/rxiang/data/era5/pr/mo/era5.mo.1980-2022.nc')
era5 = np.nanmean(ds['tp'].values[4:9, :, :], axis=0) * 1000
lat_ = ds['latitude'].values
lon_ = ds['longitude'].values

for i in model:
    ds = xr.open_dataset(f'/project/pr133/rxiang/data/pmip/{dir}/pr_Amon_{i}_{sim}.nc')
    pr = np.nanmean(ds['pr'].values[4:9, :, :], axis=0) * 86400
    lat = ds['lat'].values
    lon = ds['lon'].values
    ds = xr.open_dataset(f'/project/pr133/rxiang/data/pmip/{dir}/pr_Amon_{i}_{sim}_remap.nc')
    pr_remap = np.nanmean(ds['pr'].values[4:9, :, :], axis=0) * 86400
    # %%
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([60, 173, 0, 65], crs=ccrs.PlateCarree())
    levels = np.linspace(0, 20, 21, endpoint=True)
    cmap = cmc.davos_r
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    q = ax.pcolormesh(lon, lat, pr, cmap=cmap, norm=norm, shading="auto", transform=ccrs.PlateCarree())
    ax.set_title(f'{i}', pad=8, fontsize=12)
    cax = fig.add_axes([ax.get_position().x1+0.002, ax.get_position().y0-0.065, 0.03, ax.get_position().height*1.07])
    cb = fig.colorbar(q, cax=cax, orientation='vertical', extend='max', ticks=np.linspace(0, 20, 11, endpoint=True))
    gl = ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, linewidth=0.7,
                          color='gray', alpha=0.7, linestyle='--')
    gl.xlocator = mticker.FixedLocator([60, 80, 100, 120, 140, 160, 180])
    gl.ylocator = mticker.FixedLocator([0, 10, 20, 30, 40, 50, 60])
    ax.add_feature(cfeature.COASTLINE)
    fig.tight_layout(rect=[0.01, 0.0, 0.90, 0.98])
    # cb.set_label('mm', fontsize=11)
    # adjust figure
    fig.show()
    # save figure
    plotpath = f"/project/pr133/rxiang/figure/paper2/pmip/{dir}/"
    fig.savefig(plotpath + f'pr_{i}.png', dpi=300, transparent='True')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([60, 173, 0, 65], crs=ccrs.PlateCarree())
    cmap = drywet(25, cmc.vik_r)
    norm = colors.TwoSlopeNorm(vmin=-5., vcenter=0., vmax=5.)
    q = ax.pcolormesh(lon_, lat_, pr_remap - era5, cmap=cmap, norm=norm, shading="auto", transform=ccrs.PlateCarree())
    ax.set_title(f'{i}', pad=8, fontsize=12)
    cax = fig.add_axes(
        [ax.get_position().x1 + 0.002, ax.get_position().y0 - 0.065, 0.03, ax.get_position().height * 1.07])
    cb = fig.colorbar(q, cax=cax, orientation='vertical', extend='both', ticks=[-4, -2, 0, 2, 4])
    gl = ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, linewidth=0.7,
                      color='gray', alpha=0.7, linestyle='--')
    gl.xlocator = mticker.FixedLocator([60, 80, 100, 120, 140, 160, 180])
    gl.ylocator = mticker.FixedLocator([0, 10, 20, 30, 40, 50, 60])
    ax.add_feature(cfeature.COASTLINE)
    fig.tight_layout(rect=[0.01, 0.0, 0.90, 0.98])
    # cb.set_label('mm', fontsize=11)
    # adjust figure
    fig.show()

    plotpath = f"/project/pr133/rxiang/figure/paper2/pmip/{dir}/"
    fig.savefig(plotpath + f'pr_{i}_diff.png', dpi=300, transparent='True')
    plt.close(fig)
