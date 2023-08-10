import xarray as xr
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import numpy as np

# sims = ['ctrl', 'topo1']
# path = "/scratch/snx3000/rxiang/data/cosmo/"
#
# data = xr.open_dataset(f'{path}' + 'EAS11_ctrl/monsoon/TOT_PREC/' + '01-05.TOT_PREC.smr.yearmean.nc')
# ctrl = data.variables['TOT_PREC'][...]
#
# data = xr.open_dataset(f'{path}' + 'EAS11_topo1/monsoon/TOT_PREC/' + '01-05.TOT_PREC.smr.yearmean.nc')
# topo1 = data.variables['TOT_PREC'][...]


# %%
# conduct Wilcoxon signed-rank test
def compute_pvalue(ctrl, topo):
    ctrl = np.array(ctrl)
    topo = np.array(topo)
    p = np.zeros((int(ctrl.shape[1]), int(ctrl.shape[2])))
    for i in range(int(ctrl.shape[1])):
        for j in range(int(ctrl.shape[2])):
            ii, jj = mannwhitneyu(ctrl[:, i, j], topo[:, i, j], alternative='two-sided')
            p[i, j] = jj
    p_values = multipletests(p.flatten(), alpha=0.05, method='fdr_bh')[1].reshape((int(ctrl.shape[1]), int(ctrl.shape[2])))
    return p, p_values
