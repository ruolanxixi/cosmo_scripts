import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

mpl.style.use("classic")

# Change latex fonts
mpl.rcParams["mathtext.fontset"] = "custom"
# custom mathtext font (set default to Bitstream Vera Sans)
mpl.rcParams["mathtext.default"] = "rm"
mpl.rcParams["mathtext.rm"] = "Bitstream Vera Sans"

prec_prod = {
    # -------------------------------------------------------------------------
    "PD": {"file": "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/mon/"
                       + "V/2001-2005.V.20000.75-80.nc",
               "var_name": "V",
               "units_in": "m/s"},
    "LGM": {"file": "/project/pr133/rxiang/data/cosmo/EAS11_lgm_ssu/mon/"
                       + "V/2001-2005.V.20000.75-80.nc",
               "var_name": "V",
               "units_in": "m/s"},
    # -------------------------------------------------------------------------
    }

prec_prod_reg = {}
for i in prec_prod.keys():

    print("Process product " + i)
    # Load precipitation data
    ds = xr.open_dataset(prec_prod[i]["file"])
    prec = ds[prec_prod[i]["var_name"]].values

    prec_prod_reg[i] = - prec

y = ds["lat"]
# %%
fig = plt.figure(figsize=(12.5, 5.0), dpi=300)
gs = gridspec.GridSpec(1, 7, left=0.1, bottom=0.1, right=0.9, top=0.9,
                       hspace=0.05, wspace=0.1,
                       width_ratios=[0.18, 0.18, 0.18,
                                     0.18, 0.18, 0.18, 0.18])

cols_prec = {"LGM": "royalblue", "PD": "orange"}
n = 3
for i in ["April", "May", "June", "July", "August", "September", "October"]:
    ax = plt.subplot(gs[0, n-3])
    for j in prec_prod.keys():
        plt.plot(prec_prod_reg[j][n, 0, :, 0], y, color=cols_prec[j], zorder=3)

    plt.text(x=1, y=61, s=i, fontweight="normal", fontsize=12)
    plt.axis([0.5, 3.5, 0.0, 19.5])
    plt.xticks(np.arange(0, 20, 4), np.arange(0, 20, 4), fontsize=9)
    plt.ylim(20, 60)
    # plt.xticks(np.arange(18, 34, 2), ['22$^{o}$N', '26$^{o}$N', '30$^{o}$N', '34$^{o}$N', '38$^{o}$N', '42$^{o}$N', '46$^{o}$N', '50$^{o}$N'], fontsize=9)
    if n == 3:
        plt.yticks(np.arange(20, 65, 5), ['20$^{o}$N', '25$^{o}$N', '30$^{o}$N', '35$^{o}$N', '40$^{o}$N', '45$^{o}$N', '50$^{o}$N', '55$^{o}$N', '60$^{o}$N'])
    else:
        plt.yticks(np.arange(20, 65, 5), [""] * 9)
    n += 1

plt.show()

plotpath = "/project/pr133/rxiang/figure/paper2/results/lgm/"
fig.savefig(plotpath + 'wj.png', dpi=500)
plt.close(fig)
