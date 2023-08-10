#%%
import numpy as np
import os
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import glob
#%%
# Directory containing .nc files
base_dir = "/store/c2sm/pr04/rxiang/data_lmp/"
end_dir = "/lm_coarse/24h/"
file_extension = "W_SO.nc"
var_name = 'W_SO'  # Variable name

# Placeholder list to store data
data_list = []

# Define the years and months you are interested in
years = range(1, 6)  # Replace with your actual range
months = range(1, 13)  # All months from 1 to 12

for year in years:
    for month in months:
        # Formatted month
        formatted_year = str(year).zfill(2)
        formatted_month = str(month).zfill(2)

        # Create file path
        file_dir = f'{base_dir}' + f'{formatted_year}{formatted_month}0100_EAS11_lgm' + f'{end_dir}'
        file_path = f'{file_dir}' + f'{file_extension}'

        # Check if the file exists
        if os.path.isfile(file_path):
            # Open the .nc file
            with Dataset(file_path, 'r') as nc_file:
                # Read the variable and append to the list
                a = nc_file.variables[var_name][...]
                data_list.append(nc_file.variables[var_name][...])

# Convert the list to a single numpy array
data_array = np.concatenate(data_list, axis=0)

# %%
plt.figure(figsize=(10, 6))
x = np.arange(1, 1887)
plt.plot(x, data_array[:, 0, 379, 54], linestyle='-', marker='')
plt.show()
