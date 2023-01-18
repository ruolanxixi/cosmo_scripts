# %% importing pandas
import numpy as np
import pandas as pd
import xarray as xr

# %%
file = '/users/rxiang/EAS04_indices_for_64station.txt'
df = pd.read_csv(f"{file}", sep="    ", header=None, names=["lat", "lon"])

# %%
f = xr.open_dataset('/scratch/snx3000/rxiang/ruolan/CTRL04/day/01-05_TOT_PREC_JJA.nc')
ds = f["TOT_PREC"].values[:, :, :]
time = f["time"].values[:]

# %%
df_ts = pd.DataFrame()
for i in range(len(df.index)):
    lat = df.iloc[i].loc['lat']
    lon = df.iloc[i].loc['lon']
    ts = ds[:, lat, lon]
    df_ts = df_ts.append(pd.DataFrame(ts.reshape(1, -1)))

df_ts.index = df.index.values
df_ts_transposed = df_ts.T

# df = pd.concat([df, df_ts], axis=1)

# %%
df_np = df_ts_transposed.to_numpy()
# df_ts_transposed.to_csv('/users/rxiang/EAS04-day.txt', sep=',', header=True, index=True)
np.savetxt('/users/rxiang/EAS04-day.txt', df_np, delimiter=',', fmt='%.7f')




