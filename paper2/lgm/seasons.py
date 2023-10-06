###############################################################################
# Modules
###############################################################################
import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import matplotlib
import cartopy.feature as feature
import matplotlib.ticker as mticker
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mycolor import drywet
from plotcosmomap import plotcosmo_notick, pole, plotcosmo_notick_lgm
import matplotlib.colors as colors
import scipy.ndimage as ndimage
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from mycolor import custom_div_cmap
import metpy.calc as mpcalc
from metpy.units import units

mpl.style.use("classic")
font = {'size': 15}
matplotlib.rc('font', **font)
plt.rcParams['hatch.linewidth'] = 0.5
plt.rcParams['hatch.color'] = 'dimgrey'

###############################################################################
# %% Data
###############################################################################
# PIMP
path = '/project/pr133/rxiang/data/pmip/'
models = ('INM-CM4-8', 'MIROC-ES2L', 'MPI-ESM1-2-LR', 'AWI-ESM-1-1-LR', 'CESM2-WACCM-FV2')

# Dictionaries to store the datasets
annual_mean_data = {}
seasonal_mean_data = {season: {} for season in ['DJF', 'MAM', 'JJA', 'SON']}
for model in models:
    ds_lgm = xr.open_dataset(f'{path}lgm/{model}/ts/ts_Amon_{model}_lgm.nc')
    ds_piControl = xr.open_dataset(f'{path}piControl/{model}/ts/ts_Amon_{model}_piControl.nc')
    subset_lgm = ds_lgm.sel(lat=slice(5, 50), lon=slice(70, 160))
    subset_piControl = ds_piControl.sel(lat=slice(5, 50), lon=slice(70, 160))
    # Annual mean
    annual_mean_data[model] = np.nanmean(subset_lgm['ts'].values) - np.nanmean(subset_piControl['ts'].values)
    # Seasonal mean
    seasonal_means = subset_lgm['ts'].groupby('time.season').mean(dim='time') - subset_piControl['ts'].groupby('time.season').mean(dim='time')
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        seasonal_mean_data[season][model] = np.nanmean(seasonal_means.sel(season=season).values)

# COSMO
ds_ctrl = xr.open_dataset('/project/pr133/rxiang/data/cosmo/EAS11_ctrl/monsoon/T_2M/01-05.T_2M.cpm.lonlat.nc')
ds_lgm = xr.open_dataset('/project/pr133/rxiang/data/cosmo/EAS11_lgm/monsoon/T_2M/01-05.T_2M.cpm.lonlat.nc')
subset_lgm = ds_lgm.sel(lat=slice(5, 50), lon=slice(70, 160))
subset_piControl = ds_ctrl.sel(lat=slice(5, 50), lon=slice(70, 160))
annual_mean_data['COSMO'] = np.nanmean(subset_lgm['T_2M'].values) - np.nanmean(subset_piControl['T_2M'].values)
# Seasonal mean
seasonal_means = subset_lgm['T_2M'].groupby('time.season').mean(dim='time') - subset_piControl['T_2M'].groupby('time.season').mean(dim='time')
for season in ['DJF', 'MAM', 'JJA', 'SON']:
    seasonal_mean_data[season]['COSMO'] = np.nanmean(seasonal_means.sel(season=season).values)

# ECHAM5
ds_ctrl = xr.open_dataset('/project/pr133/rxiang/data/pgw/deltas/native/day/ECHAM5/tas_piControl.nc')
ds_lgm = xr.open_dataset('/project/pr133/rxiang/data/pgw/deltas/native/day/ECHAM5/tas_lgm.nc')
subset_lgm = ds_lgm.sel(lat=slice(50, 5), lon=slice(70, 160))
subset_piControl = ds_ctrl.sel(lat=slice(50, 5), lon=slice(70, 160))
annual_mean_data['ECHAM5'] = np.nanmean(subset_lgm['tas'].values) - np.nanmean(subset_piControl['tas'].values)
# Seasonal mean
seasonal_means = subset_lgm['tas'].groupby('time.season').mean(dim='time') - subset_piControl['tas'].groupby('time.season').mean(dim='time')
for season in ['DJF', 'MAM', 'JJA', 'SON']:
    seasonal_mean_data[season]['ECHAM5'] = np.nanmean(seasonal_means.sel(season=season).values)

models = ('INM-CM4-8', 'MIROC-ES2L', 'MPI-ESM1-2-LR', 'AWI-ESM-1-1-LR', 'CESM2-WACCM-FV2', 'ECHAM5', 'COSMO')
all_data = {model: [
    seasonal_mean_data['DJF'][model],
    seasonal_mean_data['MAM'][model],
    seasonal_mean_data['JJA'][model],
    seasonal_mean_data['SON'][model],
    annual_mean_data[model]
] for model in models}

# Calculating median and standard deviation for the first five models
selected_models = ('INM-CM4-8', 'MIROC-ES2L', 'MPI-ESM1-2-LR', 'AWI-ESM-1-1-LR', 'CESM2-WACCM-FV2')
medians = []
std_devs = []

seasons = ['DJF', 'MAM', 'JJA', 'SON', 'Annual']
for idx, season in enumerate(seasons):
    values = [all_data[model][idx] for model in selected_models]
    medians.append(np.median(values))
    std_devs.append(np.std(values))

###############################################################################
# %% Plot
###############################################################################

bar_width = 0.15
index = np.arange(len(models)) + 1
colors = ['#4393c3', '#abdda4', '#f46d43', '#fee08b', '#9970ab']

fig, ax = plt.subplots(figsize=(12, 6))

# Plotting the data
index = np.arange(5)
for idx, season in enumerate(seasons):
    means = [all_data[model][idx] for model in selected_models]
    bars = ax.bar(index + idx*bar_width, means, bar_width, label=season, color=colors[idx])

# Add the medians and std_devs to the plot
for idx, season in enumerate(seasons):
    ax.bar(5 + idx*bar_width, medians[idx], bar_width, yerr=std_devs[idx], color=colors[idx])

for idx, season in enumerate(seasons):
    means = [all_data[model][idx] for model in ['ECHAM5', 'COSMO']]
    bars = ax.bar(np.arange(2)+6 + idx*bar_width, means, bar_width, color=colors[idx])

# Adjust x-ticks to account for the additional 'Median' bars
# ax.set_xticks(np.arange(len(models) + 2*bar_width, step=bar_width))
# ax.set_xticklabels(list(models) + ['Median'])
models = ('INM-CM4-8', 'MIROC-ES2L', 'MPI-ESM1-2-LR', 'AWI-ESM-1-1-LR', 'CESM2-WACCM-FV2', '5-model median', 'ECHAM5', 'COSMO')
index = np.arange(8)
# Configuring the plot
# ax.set_xlabel('Models')
ax.set_ylabel('2m Temperature')
ax.set_xticks(index + 2*bar_width)
ax.set_xticklabels(models, rotation=45)
ax.set_xlim(0-bar_width, max(index + 2*bar_width) + 3*bar_width)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('top')
ax.tick_params(direction='out')
ax.legend(loc='lower right', fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)      # this sets the figure background transparency
ax.patch.set_alpha(0.0)


fig.tight_layout()
plt.show()
plotpath = "/project/pr133/rxiang/figure/paper2/results/lgm/"
fig.savefig(plotpath + 'ts_4seasons.png', dpi=500, transparent='true')
plt.close(fig)
