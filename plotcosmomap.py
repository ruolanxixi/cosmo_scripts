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


def add_gridline_labels(ax, labels_set=None, side=None):  # 'top', 'bottom', 'left', 'right'

    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, linewidth=1, color='grey', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'rotation': 0, 'rotation_mode': 'anchor'}
    gl.ylabel_style = {'rotation': 0, 'rotation_mode': 'anchor'}

    if side == 'top':
        gl.xlocator = mticker.FixedLocator(labels_set)
        gl.ylines = False
        gl.top_labels = True
        gl.bottom_labels = False
        gl.left_labels = False
        gl.right_labels = False
        gl.geo_labels = False
    elif side == 'bottom':
        gl.xlocator = mticker.FixedLocator(labels_set)
        gl.ylines = False
        gl.top_labels = False
        gl.bottom_labels = True
        gl.left_labels = False
        gl.right_labels = False
        gl.geo_labels = False
    elif side == 'left':
        gl.ylocator = mticker.FixedLocator(labels_set)
        gl.xlines = False
        gl.top_labels = False
        gl.bottom_labels = False
        gl.left_labels = True
        gl.right_labels = False
        gl.geo_labels = False
    elif side == 'right':
        gl.ylocator = mticker.FixedLocator(labels_set)
        gl.xlines = False
        gl.top_labels = False
        gl.bottom_labels = False
        gl.left_labels = False
        gl.right_labels = True
        gl.geo_labels = False

    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    return ax


def plotcosmo(infile, ax):
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
    try:
        lon = infile.lon
        lat = infile.lat
    except:
        print("Cannot read lon or lat from specified file. Is it an xarray data array?")
        exit

    proj = ccrs.PlateCarree()
    ax.set_extent([65, 174, 10, 61], crs=proj)  # for extended 12km domain
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)
    # ax.set_yticks([10, 20, 30, 40, 50], crs=proj)
    add_gridline_labels(ax, labels_set=[10, 20, 30, 40, 50, 60], side='left')
    add_gridline_labels(ax, labels_set=[0, 10, 20, 30, 40, 50, 60], side='right')
    add_gridline_labels(ax, labels_set=[60, 100, 140, 180], side='top')
    add_gridline_labels(ax, labels_set=[80, 100, 120, 140, 160], side='bottom')

    return ax





