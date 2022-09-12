"""
A function to draw the basic map where the COSMO output can be plotted.
The output will be plotted on a rotated pole grid
"""
# ---------------------------------------------------------
# Load modules
# ---------------------------------------------------------
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)
import xarray as xr


def add_gridline_labels(ax, labels_set=None, side=None):  # 'top', 'bottom', 'left', 'right'

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, dms=True, x_inline=False, y_inline=False, linewidth=1,
                      color='grey', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'rotation': 0, 'rotation_mode': 'anchor'}
    gl.ylabel_style = {'rotation': 0, 'rotation_mode': 'anchor'}

    if side == 'top':
        # gl.xlocator = mticker.FixedLocator(labels_set)
        gl.ylines = False
        gl.top_labels = True
        gl.bottom_labels = False
        gl.left_labels = False
        gl.right_labels = False
        gl.geo_labels = False
    elif side == 'bottom':
        # gl.xlocator = mticker.FixedLocator(labels_set)
        gl.ylines = False
        gl.top_labels = False
        gl.bottom_labels = True
        gl.left_labels = False
        gl.right_labels = False
        gl.geo_labels = False
    elif side == 'left':
        # gl.ylocator = mticker.FixedLocator(labels_set)
        gl.xlines = False
        gl.top_labels = False
        gl.bottom_labels = False
        gl.left_labels = True
        gl.right_labels = False
        gl.geo_labels = False
    elif side == 'right':
        # gl.ylocator = mticker.FixedLocator(labels_set)
        gl.xlines = False
        gl.top_labels = False
        gl.bottom_labels = False
        gl.left_labels = False
        gl.right_labels = True
        gl.geo_labels = False

    # gl.xformatter = LongitudeFormatter()
    # gl.yformatter = LatitudeFormatter()

    return ax


def plotcosmo(ax):
    """
    A function to draw the background map for plotting CCLM output.
            The output-map will be ploted on a rotated pole grid.

            Args:
                    infile: an xarray data array or a structure that contains lat and lon data

                    ax: axes

                    plabels (optional): label definition for parallels

                    mlabels (optional): labels for meridians

                    additional optional arguments with default are resolution of the coastlines, linewidth for meridians and the fontsize of the labels. additional text and line **kwargs can also be passed

            Returns:
                    m: the basemap map projection (rotated pole) used in CCLM

            Example usage:
                    from plotcosmomap import plotcosmomap

                    mydata=xr.open_dataset('mypath')

                    m, xi, yi = plotcosmomap(mydata); m.pcolormesh(xi, yi, mydata, cmap='plasma')
    """

    ax.set_extent([65, 173, 7, 61], crs=ccrs.PlateCarree())  # for extended 12km domain
    # ax.add_feature(cfeature.LAND)
    ax.stock_img()
    # ax.add_feature(cfeature.OCEAN, zorder=100)
    ax.add_feature(cfeature.LAND, edgecolor='k')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

    # add_gridline_labels(ax, labels_set=[10, 20, 30, 40, 50, 60, 70], side='left')
    # add_gridline_labels(ax, labels_set=[0, 10, 20, 30, 40, 50, 60], side='right')
    # add_gridline_labels(ax, labels_set=[60, 100, 140, 180], side='top')
    # add_gridline_labels(ax, labels_set=[80, 100, 120, 140, 160], side='bottom')
    gl = ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, linewidth=1,
                      color='grey', alpha=0.7, linestyle='--')
    gl.xlocator = mticker.FixedLocator([80, 100, 120, 140, 160])
    gl.ylocator = mticker.FixedLocator([0, 10, 20, 30, 40, 50, 60])

    # add ticks manually
    ax.text(-0.05, 0.95, '50°N', ha='center', va='center', transform=ax.transAxes, fontsize=13)
    ax.text(-0.05, 0.77, '40°N', ha='center', va='center', transform=ax.transAxes, fontsize=13)
    ax.text(-0.05, 0.59, '30°N', ha='center', va='center', transform=ax.transAxes, fontsize=13)
    ax.text(-0.05, 0.41, '20°N', ha='center', va='center', transform=ax.transAxes, fontsize=13)
    ax.text(-0.05, 0.23, '10°N', ha='center', va='center', transform=ax.transAxes, fontsize=13)
    ax.text(-0.038, 0.05, '0°N', ha='center', va='center', transform=ax.transAxes, fontsize=13)

    ax.text(0.12, -0.05, '80°E', ha='center', va='center', transform=ax.transAxes, fontsize=13)
    ax.text(0.32, -0.05, '100°E', ha='center', va='center', transform=ax.transAxes, fontsize=13)
    ax.text(0.52, -0.05, '120°E', ha='center', va='center', transform=ax.transAxes, fontsize=13)
    ax.text(0.72, -0.05, '140°E', ha='center', va='center', transform=ax.transAxes, fontsize=13)
    ax.text(0.92, -0.05, '160°E', ha='center', va='center', transform=ax.transAxes, fontsize=13)

    # ax.set_xticks([80, 100, 120, 140, 160], crs=ccrs.PlateCarree())
    # ax.set_yticks([0, 10, 20, 30, 40, 50, 60], crs=ccrs.PlateCarree())
    # lon_formatter = LongitudeFormatter(number_format='.0f', degree_symbol='°')
    # lat_formatter = LatitudeFormatter(number_format='.0f', degree_symbol='°')
    # ax.xaxis.set_major_formatter(lon_formatter)
    # ax.yaxis.set_major_formatter(lat_formatter)

    return ax



def plotcosmo04(ax):
    """
    A function to draw the background map for plotting CCLM output.
            The output-map will be ploted on a rotated pole grid.

            Args:
                    infile: an xarray data array or a structure that contains lat and lon data

                    ax: axes

                    plabels (optional): label definition for parallels

                    mlabels (optional): labels for meridians

                    additional optional arguments with default are resolution of the coastlines, linewidth for meridians and the fontsize of the labels. additional text and line **kwargs can also be passed

            Returns:
                    m: the basemap map projection (rotated pole) used in CCLM

            Example usage:
                    from plotcosmomap import plotcosmomap

                    mydata=xr.open_dataset('mypath')

                    m, xi, yi = plotcosmomap(mydata); m.pcolormesh(xi, yi, mydata, cmap='plasma')
    """

    ax.set_extent([88, 112.5, 17, 39], crs=ccrs.PlateCarree())  # for extended 12km domain
    # ax.add_feature(cfeature.LAND)
    ax.stock_img()
    # ax.add_feature(cfeature.OCEAN, zorder=100)
    ax.add_feature(cfeature.LAND, edgecolor='k')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

    # add_gridline_labels(ax, labels_set=[10, 20, 30, 40, 50, 60, 70], side='left')
    # add_gridline_labels(ax, labels_set=[0, 10, 20, 30, 40, 50, 60], side='right')
    # add_gridline_labels(ax, labels_set=[60, 100, 140, 180], side='top')
    # add_gridline_labels(ax, labels_set=[80, 100, 120, 140, 160], side='bottom')
    gl = ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, linewidth=1,
                      color='grey', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([90, 100, 110, 120])
    gl.ylocator = mticker.FixedLocator([15, 20, 25, 30, 35, 40])

    # add ticks manually
    ax.text(-0.1, 0.88, '35°N', ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.text(-0.1, 0.67, '30°N', ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.text(-0.1, 0.46, '25°N', ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.text(-0.1, 0.25, '20°N', ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.text(-0.1, 0.04, '15°N', ha='center', va='center', transform=ax.transAxes, fontsize=12)

    ax.text(0.04, -0.05, '90°E', ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.text(0.46, -0.05, '100°E', ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.text(0.86, -0.05, '110°E', ha='center', va='center', transform=ax.transAxes, fontsize=12)

    # ax.set_xticks([80, 100, 120, 140, 160], crs=ccrs.PlateCarree())
    # ax.set_yticks([0, 10, 20, 30, 40, 50, 60], crs=ccrs.PlateCarree())
    # lon_formatter = LongitudeFormatter(number_format='.0f', degree_symbol='°')
    # lat_formatter = LatitudeFormatter(number_format='.0f', degree_symbol='°')
    # ax.xaxis.set_major_formatter(lon_formatter)
    # ax.yaxis.set_major_formatter(lat_formatter)

    return ax



def pole():
    file = "/project/pr133/rxiang/data/cosmo/EAS11_ctrl/szn/T_2M/01_T_2M_DJF.nc"
    ds = xr.open_dataset(f'{file}')
    pole_lat = ds["rotated_pole"].grid_north_pole_latitude
    pole_lon = ds["rotated_pole"].grid_north_pole_longitude
    lat = ds["lat"].values
    lon = ds["lon"].values
    rlat = ds["rlat"].values
    rlon = ds["rlon"].values
    rot_pole_crs = ccrs.RotatedPole(pole_latitude=pole_lat, pole_longitude=pole_lon)

    return pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs


def pole04():
    file = "/project/pr133/rxiang/data/cosmo/EAS04_ctrl/200106/T_2M.nc"
    ds = xr.open_dataset(f'{file}')
    pole_lat = ds["rotated_pole"].grid_north_pole_latitude
    pole_lon = ds["rotated_pole"].grid_north_pole_longitude
    lat = ds["lat"].values
    lon = ds["lon"].values
    rlat = ds["rlat"].values
    rlon = ds["rlon"].values
    rot_pole_crs = ccrs.RotatedPole(pole_latitude=pole_lat, pole_longitude=pole_lon)

    return pole_lat, pole_lon, lat, lon, rlat, rlon, rot_pole_crs



def colorbar(fig, ax, n, wspace):
    if n == 1:
        cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 - 0.04, ax.get_position().width, 0.012])
    elif n == 2:
        cax = fig.add_axes(
            [ax.get_position().x0, ax.get_position().y0 - 0.04, ax.get_position().width * 2 + wspace, 0.012])
    elif n == 3:
        cax = fig.add_axes(
            [ax.get_position().x0, ax.get_position().y0 - 0.04, ax.get_position().width * 3 + wspace*2, 0.012])
    elif n == 4:
        cax = fig.add_axes(
            [ax.get_position().x0, ax.get_position().y0 - 0.04, ax.get_position().width * 4 + wspace*3, 0.012])

    return cax



def custom_div_cmap(numcolors, colormap):
    """ Create a custom diverging colormap with three colors

    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """

    from matplotlib.colors import LinearSegmentedColormap

    colors_red = colormap(np.linspace(0, 0.5, 20))
    colors_blue = colormap(np.linspace(0.5, 1, 20))
    colors_white = np.array([1, 1, 1, 1])
    colors = np.vstack((colors_red, colors_white, colors_blue))

    cmap = LinearSegmentedColormap.from_list(name=colormap, colors=colors, N=numcolors)

    return cmap



def custom_seq_cmap(numcolors, colormap):
    """ Create a custom diverging colormap with three colors

    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """

    from matplotlib.colors import LinearSegmentedColormap

    colors_blue = colormap(np.linspace(0.5, 1, 20))
    colors_white = np.array([1, 1, 1, 1])
    colors = np.vstack((colors_white, colors_blue))

    cmap = LinearSegmentedColormap.from_list(name=colormap, colors=colors, N=numcolors)

    return cmap



def custom_white_cmap(numcolors, colormap):
    """ Create a custom diverging colormap with three colors

    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """

    from matplotlib.colors import LinearSegmentedColormap

    colors_blue = colormap(np.linspace(0, 1, 20))
    colors_white = np.array([1, 1, 1, 1])
    colors = np.vstack((colors_white, colors_blue))

    cmap = LinearSegmentedColormap.from_list(name=colormap, colors=colors, N=numcolors)

    return cmap


