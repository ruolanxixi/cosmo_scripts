# Description: Compute Hengduan Mountains regions masks for different products
#
# Author: Christian R. Steger, April 2023

# Load modules
import os
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

# Path to folders
path_reg_masks = "/project/pr133/rxiang/data/region_masks/"
path_out = "/project/pr133/rxiang/figure/paper2/results/lgm/"
# output directory

###############################################################################
# Load and process temperature data
###############################################################################

temp_prod = {
    # -------------------------------------------------------------------------
    "COSMO CTRL": {"file": "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/mon/"
                       + "T_2M/2001-2005.T_2M.nc",
               "var_name": "T_2M",
               "units_in": "K"},
    "COSMO PGW": {"file": "/project/pr133/rxiang/data/cosmo/EAS11_lgm/mon/"
                       + "T_2M/2001-2005.T_2M.nc",
               "var_name": "T_2M",
               "units_in": "K"},
    # "ECHAM5-PI": {"file": "/project/pr133/rxiang/data/pgw/deltas/native/day/ECHAM5/"
    #                    + "tas_piControl_mon.nc",
    #            "var_name": "tas",
    #            "units_in": "K"},
    # "ECHAM5-LGM": {"file": "/project/pr133/rxiang/data/pgw/deltas/native/day/ECHAM5/"
    #                    + "tas_lgm_mon.nc",
    #            "var_name": "tas",
    #            "units_in": "K"},
    # # -------------------------------------------------------------------------
    }

# Load topographies
topo = {}
# -----------------------------------------------------------------------------
ds = xr.open_dataset("/project/pr133/rxiang/data/extpar/"
                     + "extpar_EAS_ext_12km_merit_unmod_topo.nc")
ds = ds.isel(rlon=slice(32, 1090), rlat=slice(28, 638))
topo["COSMO CTRL"] = ds["HSURF"].values
ds.close()
# -----------------------------------------------------------------------------
ds = xr.open_dataset("/project/pr133/rxiang/data/extpar/"
                     + "extpar_EAS_ext_12km_merit_LGM_consistent_TCL.nc")
ds = ds.isel(rlon=slice(32, 1090), rlat=slice(28, 638))
topo["COSMO PGW"] = ds["HSURF"].values
# print(ds["rlon"][0].values, ds["rlon"][-1].values)
# print(ds["rlat"][0].values, ds["rlat"][-1].values)
ds.close()
# -----------------------------------------------------------------------------
ds = xr.open_dataset("/project/pr133/rxiang/data/echam5_raw/PI/input/"
                     + "T159_jan_surf.nc")
#mds = ds.isel(rlon=slice(32, 1090), rlat=slice(28, 638))
topo["ECHAM5-PI"] = ds["OROMEA"].values
ds.close()
# -----------------------------------------------------------------------------
ds = xr.open_dataset("/project/pr133/rxiang/data/echam5_raw/LGM/input/"
                     + "T159_jan_surf.lgm.veg.nc")
# ds = ds.isel(rlon=slice(32, 1090), rlat=slice(28, 638))
topo["ECHAM5-LGM"] = ds["OROMEA"].values
ds.close()
# -----------------------------------------------------------------------------
regions = ["ET", "HM", "HMU", "HMC", "HMUS", "HMUN"]
months_num = np.arange(1, 13)
mask_MAM = (months_num >= 3) & (months_num <= 5)  # rainy season (MJJAS)
mask_JJA = (months_num >= 6) & (months_num <= 8)  # rainy season (MJJAS)
mask_SON = (months_num >= 9) & (months_num <= 11)  # rainy season (MJJAS)
mask_DJF = (months_num <= 2) | (months_num >= 12)  # dry season (NDJFM)

# Compute monthly temperature for products and regions
temp_prod_reg = {}
temp_prod_elev = {}
elev_bin_edges = np.arange(0.0, 7500.0, 500.0)
elev_bin = elev_bin_edges[:-1] + np.diff(elev_bin_edges) / 2.0
num_bin = len(elev_bin)
for i in temp_prod.keys():

    print("Process product " + i)
    temp_prod_reg[i] = {}

    # Load temperature data
    ds = xr.open_dataset(temp_prod[i]["file"])
    temp = ds[temp_prod[i]["var_name"]].values
    ds.close()

    # Convert units of temperature data (if necessary)
    if temp_prod[i]["units_in"] == "deg_C":
        pass
    elif temp_prod[i]["units_in"] == "K":
        temp -= 273.15
    else:
        raise ValueError("Unknown input units")

    # Load region masks
    if i == 'COSMO CTRL' or i == 'COSMO PGW':
        ds = xr.open_dataset(path_reg_masks + "CTRL11" + "_region_masks.nc")
        region_masks = {j: ds[j].values.astype(bool) for j in regions}
        ds.close()
    if i == 'ECHAM5-PI' or i == 'ECHAM5-LGM':
        ds = xr.open_dataset(path_reg_masks + "ECHAM5" + "_region_masks.nc")
        region_masks = {j: ds[j].values.astype(bool) for j in regions}
        ds.close()

    # Compute spatially integrated temperature for regions
    for j in regions:
        temp_seas = np.empty(12 + 4, dtype=np.float32)
        for k in range(12):
            temp_seas[k] = np.nanmean(temp[k, :, :][region_masks[j]])
        temp_seas[12] = np.nanmean(temp[mask_DJF, :, :].mean(axis=0)
                                       [region_masks[j]])  # dry season
        temp_seas[13] = np.nanmean(temp[mask_MAM, :, :].mean(axis=0)
                                   [region_masks[j]])  # rainy season (MJJAS)
        temp_seas[14] = np.nanmean(temp[mask_JJA, :, :].mean(axis=0)
                                   [region_masks[j]])  # dry season
        temp_seas[15] = np.nanmean(temp[mask_SON, :, :].mean(axis=0)
                                   [region_masks[j]])  # dry season

        # -> use grid cell area as weights for more accurate spatial averaging!
        temp_prod_reg[i][j] = temp_seas

    # Compute temperature lapse rates
    temp_grad_MAM = np.empty(num_bin, dtype=np.float32)  # rainy season
    temp_grad_MAM.fill(np.nan)
    temp_grad_JJA = np.empty(num_bin, dtype=np.float32)  # dry season
    temp_grad_JJA.fill(np.nan)
    temp_grad_SON = np.empty(num_bin, dtype=np.float32)  # rainy season
    temp_grad_SON.fill(np.nan)
    temp_grad_DJF = np.empty(num_bin, dtype=np.float32)  # dry season
    temp_grad_DJF.fill(np.nan)
    for j in range(num_bin):
        mask = (topo[i] >= elev_bin_edges[j]) \
               & (topo[i] < elev_bin_edges[j + 1]) & region_masks["HM"]
        if mask.sum() >= 15:
            temp_grad_DJF[j] = np.nanmean(temp[mask_DJF, :, :].mean(axis=0)
                                          [mask])
            temp_grad_MAM[j] = np.nanmean(temp[mask_MAM, :, :].mean(axis=0)
                                         [mask])
            temp_grad_JJA[j] = np.nanmean(temp[mask_JJA, :, :].mean(axis=0)
                                         [mask])
            temp_grad_SON[j] = np.nanmean(temp[mask_SON, :, :].mean(axis=0)
                                          [mask])
    temp_prod_elev[i] = {"DJF": temp_grad_DJF,
                         "MAM": temp_grad_MAM,
                         "JJA": temp_grad_JJA,
                         "SON": temp_grad_SON
                         }


###############################################################################
# %% Plot
###############################################################################
font = {'size': 14}
# Settings
cols = {"COSMO CTRL": "#f46d43", "COSMO PGW": "#4393c3", "ECHAM5-PI": "#f46d43", "ECHAM5-LGM": "#4393c3"}
markers = {"COSMO CTRL": "o", "COSMO PGW": "o", "ECHAM5-PI": "x", "ECHAM5-LGM": "x"}
s = 50
s_small = 30

# Plot
fig = plt.figure(figsize=(15, 4.3), dpi=300)
gs = gridspec.GridSpec(1, 8, left=0.04, bottom=0.14, right=0.99, top=0.93,
                       hspace=0.05, wspace=0.0,
                       width_ratios=[0.6, 0.12, 0.05, 0.04,
                                     0.2, 0.2, 0.2, 0.2])
# -----------------------------------------------------------------------------
months_char = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# -----------------------------------------------------------------------------
ax = plt.subplot(gs[0, 0])
for i in temp_prod.keys():
    plt.plot(months_num, temp_prod_reg[i]["HM"][:12], color=cols[i],
             zorder=3)
    plt.scatter(months_num, temp_prod_reg[i]["HM"][:12], s=s_small, marker=markers[i],
                color=cols[i], label=i, zorder=3)
# plt.fill_between(x=[4.5, 9.5], y1=-11.0, y2=22.0, color="black", alpha=0.1)
plt.xticks(months_num, months_char, fontsize=14)
# plt.text(x=6.0, y=-1.8, s="Rainy season", fontsize=14)
plt.ylabel("2m temperature [$^{\circ} C$]", labelpad=5, fontsize=14)
plt.yticks(np.arange(-2, 22, 2), np.arange(-2, 22, 2), fontsize=14)
plt.ylim([-4, 24])
plt.axis([0.7, 12.3, -4.0, 22.0])
plt.title("(a)", fontsize=14, fontweight="normal", y=1.01,
          loc="left")
plt.legend(loc="upper left", frameon=False, fontsize=13, ncol=1,
           scatterpoints=1)
# -----------------------------------------------------------------------------
ax = plt.subplot(gs[0, 1])
x_2 = np.arange(1, 5)
x_ticks_2 = ["DJF", "MAM", "JJA", "SON"]
for i in temp_prod.keys():
    plt.scatter(x_2, temp_prod_reg[i]["HM"][12:], s=s, color=cols[i], marker=markers[i],
                zorder=3)
plt.xticks(x_2, x_ticks_2, rotation=90, fontsize=13)
plt.yticks(np.arange(-2, 22, 2), [""] * 12, fontsize=14)
plt.axis([0.3, 4.7, -4.0, 22.0])
# -----------------------------------------------------------------------------
seas = ("DJF", "MAM", "JJA", "SON")
pos_x = (4, 5, 6, 7)
lims = ([-26, 19], [-17.5, 27.5], [-10, 35], [-17.5, 27.5])
for i in range(0, 4):
    lim = lims[i]
    ax = plt.subplot(gs[0, pos_x[i]])
    for j in temp_prod_elev.keys():
        k = seas[i].replace(" ", "_")
        plt.plot(temp_prod_elev[j][k], elev_bin, color=cols[j])
        plt.scatter(temp_prod_elev[j][k], elev_bin, s=s_small, marker=markers[j],
                    color=cols[j], label=j)
    plt.text(x=0.5, y=0.94, s=seas[i], fontsize=14,
             horizontalalignment="center", verticalalignment="center",
             transform=ax.transAxes)
    plt.xticks(np.arange(-20, 35, 10), fontsize=14)
    ax.set_xticks(np.arange(-25, 40, 10), minor=True)
    if i == 0:
        plt.yticks(np.arange(0, 7000, 500), np.arange(0, 7.0, 0.5), fontsize=14)
        plt.xlabel(" " * 105 + "2m temperature [$^{\circ} C$]", fontsize=14)
        plt.ylabel("Elevation [km a.s.l.]", labelpad=7.5, fontsize=14)
        plt.title("(b)", fontsize=14, fontweight="normal",
                  y=1.01,
                  loc="left")
    else:
        plt.yticks(np.arange(0, 7000, 500), "" * 14)
    plt.xlim(lim)
    plt.ylim([0.0, 6000.0])
# -----------------------------------------------------------------------------
plt.show()
fig.savefig(path_out + "lapse_rate.png", dpi=300,
           bbox_inches="tight")
plt.close(fig)
