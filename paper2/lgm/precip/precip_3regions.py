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
# Load and process precipitation data
###############################################################################

precip_prod = {
    # -------------------------------------------------------------------------
    "COSMO CTRL": {"file": "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/mon/"
                       + "TOT_PREC/2001-2005.TOT_PREC.nc",
               "var_name": "TOT_PREC",
               "units_in": "mm/day"},
    "COSMO PGW": {"file": "/project/pr133/rxiang/data/cosmo/EAS11_lgm/mon/"
                       + "TOT_PREC/2001-2005.TOT_PREC.nc",
               "var_name": "TOT_PREC",
               "units_in": "mm/day"}}

# -----------------------------------------------------------------------------
regions = ["HM", "SHM", "SEC"]
months_num = np.arange(1, 13)
mask_MAM = (months_num >= 3) & (months_num <= 5)  # rainy season (MJJAS)
mask_JJA = (months_num >= 6) & (months_num <= 8)  # rainy season (MJJAS)
mask_SON = (months_num >= 9) & (months_num <= 11)  # rainy season (MJJAS)
mask_DJF = (months_num <= 2) | (months_num >= 12)  # dry season (NDJFM)

precip_prod_reg = {}
precip_prod_elev = {}
for i in precip_prod.keys():

    print("Process product " + i)
    precip_prod_reg[i] = {}

    # Load temperature data
    ds = xr.open_dataset(precip_prod[i]["file"])
    temp = ds[precip_prod[i]["var_name"]].values
    ds.close()

    # Load region masks
    if i == 'COSMO CTRL' or i == 'COSMO PGW':
        ds = xr.open_dataset(path_reg_masks + "CTRL11" + "_region_masks.nc")
        region_masks = {j: ds[j].values.astype(bool) for j in regions}
        ds.close()

        # Compute spatially integrated temperature for regions
        for j in regions:
            precip_seas = np.empty(12 + 4, dtype=np.float32)
            for k in range(12):
                precip_seas[k] = np.nanmean(temp[k, :, :][region_masks[j]])
            precip_seas[12] = np.nanmean(temp[mask_DJF, :, :].mean(axis=0)
                                       [region_masks[j]])  # dry season
            precip_seas[13] = np.nanmean(temp[mask_MAM, :, :].mean(axis=0)
                                       [region_masks[j]])  # rainy season (MJJAS)
            precip_seas[14] = np.nanmean(temp[mask_JJA, :, :].mean(axis=0)
                                       [region_masks[j]])  # dry season
            precip_seas[15] = np.nanmean(temp[mask_SON, :, :].mean(axis=0)
                                       [region_masks[j]])  # dry season

            # -> use grid cell area as weights for more accurate spatial averaging!
            precip_prod_reg[i][j] = precip_seas

###############################################################################
# %% Plot
###############################################################################

# Settings
cols = {"COSMO CTRL": "#f46d43", "COSMO PGW": "#4393c3"}
markers = {"COSMO CTRL": "o", "COSMO PGW": "o"}
s = 50
s_small = 30

# Plot
fig = plt.figure(figsize=(20, 5), dpi=300)
gs = gridspec.GridSpec(1, 8, left=0.04, bottom=0.1, right=0.99, top=0.94,
                       hspace=0.05, wspace=0.0,
                       width_ratios=[0.5, 0.12, 0.06, 0.5, 0.12, 0.06, 0.5, 0.12])
# -----------------------------------------------------------------------------
months_char = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
# -----------------------------------------------------------------------------
ax = plt.subplot(gs[0, 3])
for i in precip_prod.keys():
    plt.plot(months_num, precip_prod_reg[i]["HM"][:12], color=cols[i],
             zorder=3)
    plt.scatter(months_num, precip_prod_reg[i]["HM"][:12], s=s_small, marker=markers[i],
                color=cols[i], label=i, zorder=3)
# plt.fill_between(x=[4.5, 9.5], y1=-11.0, y2=22.0, color="black", alpha=0.1)
plt.xticks(months_num, months_char)
# plt.text(x=6.0, y=-4.8, s="Rainy season", fontsize=12)
plt.ylabel("Precipitation [mm d$^{-1}$]", labelpad=5, fontsize=13)
plt.yticks(np.arange(0, 12, 1), np.arange(0, 12, 1))
plt.ylim([0, 12])
plt.axis([0.7, 12.3, 0, 12])
plt.title("(b) Hengduan Mountains", fontsize=14, fontweight="normal", y=1.01,
          loc="left")
plt.legend(loc="upper left", frameon=False, fontsize=12, ncol=1,
           scatterpoints=1)
# -----------------------------------------------------------------------------
ax = plt.subplot(gs[0, 4])
x_2 = np.arange(1, 5)
x_ticks_2 = ["DJF", "MAM", "JJA", "SON"]
for i in precip_prod.keys():
    plt.scatter(x_2, precip_prod_reg[i]["HM"][12:], s=s, color=cols[i], marker=markers[i],
                zorder=3)
plt.xticks(x_2, x_ticks_2, rotation=90, fontsize=11)
plt.yticks(np.arange(0, 12, 1), [""] * 12)
plt.axis([0.5, 4.5, 0, 12.0])
# -----------------------------------------------------------------------------
ax = plt.subplot(gs[0, 0])
for i in precip_prod.keys():
    plt.plot(months_num, precip_prod_reg[i]["SHM"][:12], color=cols[i],
             zorder=3)
    plt.scatter(months_num, precip_prod_reg[i]["SHM"][:12], s=s_small, marker=markers[i],
                color=cols[i], label=i, zorder=3)
# plt.fill_between(x=[4.5, 9.5], y1=-11.0, y2=22.0, color="black", alpha=0.1)
plt.xticks(months_num, months_char)
# plt.text(x=6.0, y=-4.8, s="Rainy season", fontsize=12)
# plt.ylabel("2m temperature [$^{\circ} C$]", labelpad=5)
plt.yticks(np.arange(0, 26, 2), np.arange(0, 26, 2))
plt.ylim([0, 26])
plt.axis([0.7, 12.3, 0, 26])
plt.title("(a) Southern Himalaya", fontsize=14, fontweight="normal", y=1.01,
          loc="left")
plt.legend(loc="upper left", frameon=False, fontsize=12, ncol=1,
           scatterpoints=1)
# -----------------------------------------------------------------------------
ax = plt.subplot(gs[0, 1])
x_2 = np.arange(1, 5)
x_ticks_2 = ["DJF", "MAM", "JJA", "SON"]
for i in precip_prod.keys():
    plt.scatter(x_2, precip_prod_reg[i]["SHM"][12:], s=s, color=cols[i], marker=markers[i],
                zorder=3)
plt.xticks(x_2, x_ticks_2, rotation=90, fontsize=11)
plt.yticks(np.arange(0, 26, 2), [""] * 13)
plt.axis([0.5, 4.5, 0, 26.0])
# -----------------------------------------------------------------------------
ax = plt.subplot(gs[0, 6])
for i in precip_prod.keys():
    plt.plot(months_num, precip_prod_reg[i]["SEC"][:12], color=cols[i],
             zorder=3)
    plt.scatter(months_num, precip_prod_reg[i]["SEC"][:12], s=s_small, marker=markers[i],
                color=cols[i], label=i, zorder=3)
# plt.fill_between(x=[4.5, 9.5], y1=-11.0, y2=22.0, color="black", alpha=0.1)
plt.xticks(months_num, months_char)
# plt.text(x=6.0, y=-4.8, s="Rainy season", fontsize=12)
# plt.ylabel("2m temperature [$^{\circ} C$]", labelpad=5)
plt.yticks(np.arange(0, 12, 1), np.arange(0, 12, 1))
plt.ylim([0, 12])
plt.axis([0.7, 12.3, 0, 12])
plt.title("(c) Southeastern China", fontsize=14, fontweight="normal", y=1.01,
          loc="left")
plt.legend(loc="upper left", frameon=False, fontsize=12, ncol=1,
           scatterpoints=1)
# -----------------------------------------------------------------------------
ax = plt.subplot(gs[0, 7])
x_2 = np.arange(1, 5)
x_ticks_2 = ["DJF", "MAM", "JJA", "SON"]
for i in precip_prod.keys():
    plt.scatter(x_2, precip_prod_reg[i]["SEC"][12:], s=s, color=cols[i], marker=markers[i],
                zorder=3)
plt.xticks(x_2, x_ticks_2, rotation=90, fontsize=11)
plt.yticks(np.arange(0, 12, 1), [""] * 12)
plt.axis([0.5, 4.5, 0, 12.0])

fig.savefig(path_out + "precipitation_regions.png", dpi=300,
           bbox_inches="tight", transparent='True')
plt.show()
