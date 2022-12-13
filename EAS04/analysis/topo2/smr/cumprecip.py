import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def get_cumfreq(val, nvals=None):
    if nvals == None:
        nvals = len(val)
#     weights = np.ones_like(val)/float(len(val))
    weights = np.ones_like(val)/nvals
    max_int = int(np.max(val)) + 1
    num_bins = 10 * max_int
    freq, base = np.histogram(val, bins=num_bins, range=(0, max_int),
                              weights=weights)
    freq = freq[::-1]
    cumfreq = np.cumsum(freq)
    cumfreq = cumfreq[::-1]
    base = base[0:-1]
    return base, cumfreq

# Get histogram
# def get_freq(val):
#     weights = np.ones_like(val)/float(len(val))
#     max_int = int(np.max(val)) + 1
#     num_bins = 10 * max_int
#     freq, base = np.histogram(val, bins=num_bins, range=(0, max_int),
#                               weights=weights)
#     base = base[0:-1]
#     return base, freq

# read data
sims = ['ctrl', 'topo2']
pr = {}
row = []
# mdpath = "/project/pr133/rxiang/data/cosmo/"
# fname_pr = '01-05.W.50000.smr.yhourmean.nc'

mdpath = '/scratch/snx3000/rxiang/data/cosmo/'
fname_pr = '01-05.TOT_PREC.hr.smr.nc'

colors = {'ctrl': 'darkorange', 'topo2': 'steelblue'}
labels = {'ctrl': 'CTRL', 'topo2': 'TENV'}

# data = xr.open_dataset(f'{mdpath}EAS04_ctrl/diurnal/W/{fname_pr}')
data = xr.open_dataset(f'{mdpath}EAS04_ctrl/monsoon/TOT_PREC/{fname_pr}')
lon = data['lon'].values[:, :]
lat = data['lat'].values[:, :]

mask = (lat > 26) & (lat < 32) & (lon > 98) & (lon < 104)
mask3d = np.broadcast_to(mask, data['TOT_PREC'].values[:, :, :].shape)

for s in range(len(sims)):
    sim = sims[s]
    data = xr.open_dataset(f'{mdpath}EAS04_{sim}/monsoon/TOT_PREC/{fname_pr}')
    # data = np.nanmean(data['W'].values[:, 0, :, :], axis=0)
    data = data['TOT_PREC'].values[:, :, :]
    # row = data[(lat > 26) & (lat < 32) & (lon > 98) & (lon < 104)]
    row = data[mask3d].flatten()
    del data
    pr[sim] = {}
    pr[sim]['pr'] = row
    row = []

    prl = pr[sim]['pr'][pr[sim]['pr'] > 0.1]

    nvals_pr = len(pr[sim]['pr'])

    base_pr, cf_pr = get_cumfreq(prl, nvals_pr)

    pr[sim]['base_pr'] = base_pr
    pr[sim]['cf_pr'] = cf_pr

    pr[sim]['color'] = colors[sim]
    pr[sim]['label'] = labels[sim]

del mask, mask3d, lon, lat

fig = plt.figure()
fig.set_size_inches(10, 8)
ax = fig.add_subplot(111)

cf_lines = []
cf_labels = []

loc_x = 0.5
incr = .3

lw = 2.
textsize = 20.
labelsize = 24.
titlesize = 28.
handlelength=2.

for sim in sims:
    color = pr[sim]['color']
    ax.plot(pr[sim]['base_pr'], pr[sim]['cf_pr'], lw=lw, color=color, label=pr[sim]['label'])
    cf_labels.append(pr[sim]['label'])

    # ax.plot(loc_x, pr[sim]['cf_pr_pos_850'][0], '<', ms=7)
    # ax.plot(-loc_x, pr[sim]['cf_pr_neg_850'][0], '>', ms=7)
    # loc_x += incr

# Labels, legend, scale, limits
ax.set_yscale('log')
ax.set_ylim([10**(-8), 1])
ax.set_xlim([0, 50])
ax.legend(loc='best', frameon=False,  prop={'size': textsize}, handlelength=handlelength)
ax.tick_params(axis='both', which='major', labelsize=labelsize)
ax.set_xlabel('Precipitation [mm h$^{-1}$]', size=labelsize)
ax.set_ylabel('Cumulative Frequency', size=labelsize)

# Remove some lines
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(False)
plt.grid(False)

# plt.show()
# plotpath = "/project/pr133/rxiang/figure/analysis/EAS04/topo2/smr/"
# fig.savefig(plotpath + 'cumwind500.png', dpi=500)
plotpath = "/scratch/snx3000/rxiang/"
fig.savefig(plotpath + 'cumprecip.png', dpi=500)
plt.close(fig)
# Save and close
# with plt.rc_context({'savefig.format': 'pdf'}):
#     plt.savefig('plots/w_new/w_' + str(p_hPa) + '_' + str(res) + '.pdf', bbox_inches='tight')

