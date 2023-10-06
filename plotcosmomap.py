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
import pickle


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
    # ax.stock_img()
    # ax.add_feature(cfeature.OCEAN, zorder=100)
    # ax.add_feature(cfeature.LAND, edgecolor='k')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    # ax.add_feature(cfeature.RIVERS)

    # add_gridline_labels(ax, labels_set=[10, 20, 30, 40, 50, 60, 70], side='left')
    # add_gridline_labels(ax, labels_set=[0, 10, 20, 30, 40, 50, 60], side='right')
    # add_gridline_labels(ax, labels_set=[60, 100, 140, 180], side='top')
    # add_gridline_labels(ax, labels_set=[80, 100, 120, 140, 160], side='bottom')
    gl = ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, linewidth=1,
                      color='grey', alpha=0.7, linestyle='--')
    gl.xlocator = mticker.FixedLocator([60, 80, 100, 120, 140, 160, 180])
    gl.ylocator = mticker.FixedLocator([0, 10, 20, 30, 40, 50, 60])

    # add ticks manually
    ax.text(-0.008, 0.95, '50°N', ha='right', va='center', transform=ax.transAxes, fontsize=13)
    ax.text(-0.008, 0.77, '40°N', ha='right', va='center', transform=ax.transAxes, fontsize=13)
    ax.text(-0.008, 0.59, '30°N', ha='right', va='center', transform=ax.transAxes, fontsize=13)
    ax.text(-0.008, 0.41, '20°N', ha='right', va='center', transform=ax.transAxes, fontsize=13)
    ax.text(-0.008, 0.23, '10°N', ha='right', va='center', transform=ax.transAxes, fontsize=13)
    ax.text(-0.008, 0.05, '0°N', ha='right', va='center', transform=ax.transAxes, fontsize=13)

    ax.text(0.12, -0.02, '80°E', ha='center', va='top', transform=ax.transAxes, fontsize=13)
    ax.text(0.32, -0.02, '100°E', ha='center', va='top', transform=ax.transAxes, fontsize=13)
    ax.text(0.52, -0.02, '120°E', ha='center', va='top', transform=ax.transAxes, fontsize=13)
    ax.text(0.72, -0.02, '140°E', ha='center', va='top', transform=ax.transAxes, fontsize=13)
    ax.text(0.92, -0.02, '160°E', ha='center', va='top', transform=ax.transAxes, fontsize=13)

    # ax.set_xticks([80, 100, 120, 140, 160], crs=ccrs.PlateCarree())
    # ax.set_yticks([0, 10, 20, 30, 40, 50, 60], crs=ccrs.PlateCarree())
    # lon_formatter = LongitudeFormatter(number_format='.0f', degree_symbol='°')
    # lat_formatter = LatitudeFormatter(number_format='.0f', degree_symbol='°')
    # ax.xaxis.set_major_formatter(lon_formatter)
    # ax.yaxis.set_major_formatter(lat_formatter)

    return ax


def plotcosmo_notick(ax):

    ax.set_extent([65, 173, 7, 61], crs=ccrs.PlateCarree())  # for extended 12km domain
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)

    gl = ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, linewidth=0.7,
                      color='gray', alpha=0.7, linestyle='--')
    gl.xlocator = mticker.FixedLocator([60, 80, 100, 120, 140, 160, 180])
    gl.ylocator = mticker.FixedLocator([0, 10, 20, 30, 40, 50, 60])

    return ax


def plotcosmo_notick_lgm(ax, diff=False):

    ax.set_extent([65, 173, 7, 61], crs=ccrs.PlateCarree())  # for extended 12km domain
    ax.add_feature(cfeature.LAKES, alpha=0.5)

    with open('/project/pr133/rxiang/data/extpar/lgm_contour.pkl', 'rb') as file:
        contours_rlatrlon = pickle.load(file)

    # loaded_contours_rlatrlon now contains the list

    if diff:
        ax.add_feature(cfeature.COASTLINE, linestyle='--')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        for contour in contours_rlatrlon:
            ax.plot(contour[:, 0], contour[:, 1], c='black', linewidth=1)
    else:
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        for contour in contours_rlatrlon:
            ax.plot(contour[:, 0], contour[:, 1], c='black', linewidth=1)

    gl = ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, linewidth=0.7,
                      color='gray', alpha=0.7, linestyle='--')
    gl.xlocator = mticker.FixedLocator([60, 80, 100, 120, 140, 160, 180])
    gl.ylocator = mticker.FixedLocator([0, 10, 20, 30, 40, 50, 60])

    return ax


def plotcosmo_notick_nogrid(ax):

    ax.set_extent([65, 173, 7, 61], crs=ccrs.PlateCarree())  # for extended 12km domain
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    return ax


def plotcosmo04(ax):
    ax.set_extent([89, 112.5, 22.2, 39], crs=ccrs.PlateCarree())  # for extended 12km domain
    # ax.add_feature(cfeature.LAND)
    # ax.stock_img()
    # ax.add_feature(cfeature.OCEAN, zorder=100)
    # ax.add_feature(cfeature.LAND, edgecolor='k')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    # ax.add_feature(cfeature.RIVERS)

    # add_gridline_labels(ax, labels_set=[10, 20, 30, 40, 50, 60, 70], side='left')
    # add_gridline_labels(ax, labels_set=[0, 10, 20, 30, 40, 50, 60], side='right')
    # add_gridline_labels(ax, labels_set=[60, 100, 140, 180], side='top')
    # add_gridline_labels(ax, labels_set=[80, 100, 120, 140, 160], side='bottom')
    gl = ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, linewidth=1,
                      color='grey', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([90, 100, 110, 120])
    gl.ylocator = mticker.FixedLocator([20, 25, 30, 35, 40])

    # add ticks manually
    ax.text(-0.01, 0.83, '35°N', ha='right', va='center', transform=ax.transAxes, fontsize=14)
    ax.text(-0.01, 0.57, '30°N', ha='right', va='center', transform=ax.transAxes, fontsize=14)
    ax.text(-0.01, 0.31, '25°N', ha='right', va='center', transform=ax.transAxes, fontsize=14)
    ax.text(-0.01, 0.05, '20°N', ha='right', va='center', transform=ax.transAxes, fontsize=14)
    # ax.text(-0.01, 0.04, '15°N', ha='right', va='center', transform=ax.transAxes, fontsize=14)

    ax.text(0.04, -0.02, '90°E', ha='center', va='top', transform=ax.transAxes, fontsize=14)
    ax.text(0.46, -0.02, '100°E', ha='center', va='top', transform=ax.transAxes, fontsize=14)
    ax.text(0.86, -0.02, '110°E', ha='center', va='top', transform=ax.transAxes, fontsize=14)

    # ax.set_xticks([80, 100, 120, 140, 160], crs=ccrs.PlateCarree())
    # ax.set_yticks([0, 10, 20, 30, 40, 50, 60], crs=ccrs.PlateCarree())
    # lon_formatter = LongitudeFormatter(number_format='.0f', degree_symbol='°')
    # lat_formatter = LatitudeFormatter(number_format='.0f', degree_symbol='°')
    # ax.xaxis.set_major_formatter(lon_formatter)
    # ax.yaxis.set_major_formatter(lat_formatter)

    return ax


def plotcosmo04_notick(ax):

    ax.set_extent([89, 112.5, 22.2, 39], crs=ccrs.PlateCarree())  # for extended 12km domain
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)

    gl = ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, linewidth=1,
                      color='grey', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([90, 100, 110, 120])
    gl.ylocator = mticker.FixedLocator([20, 25, 30, 35, 40])

    return ax


def plotcosmo04_notick_lgm(ax, diff=False):
    ax.set_extent([89, 112.5, 22.2, 39], crs=ccrs.PlateCarree())  # for extended 12km domain
    ax.add_feature(cfeature.LAKES, alpha=0.5)

    with open('/project/pr133/rxiang/data/extpar/lgm_contour.pkl', 'rb') as file:
        contours_rlatrlon = pickle.load(file)

    # loaded_contours_rlatrlon now contains the list

    if diff:
        ax.add_feature(cfeature.COASTLINE, linestyle='--')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        for contour in contours_rlatrlon:
            ax.plot(contour[:, 0], contour[:, 1], c='black', linewidth=1)
    else:
        for contour in contours_rlatrlon:
            ax.plot(contour[:, 0], contour[:, 1], c='black', linewidth=1)

    gl = ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, linewidth=1,
                      color='grey', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([90, 100, 110, 120])
    gl.ylocator = mticker.FixedLocator([20, 25, 30, 35, 40])

    return ax


def plotcosmo04sm(ax):
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

    ax.set_extent([95, 106, 21.9, 31.1], crs=ccrs.PlateCarree())  # for extended 12km domain
    # ax.add_feature(cfeature.LAND)
    # ax.stock_img()
    # ax.add_feature(cfeature.OCEAN, zorder=100)
    # ax.add_feature(cfeature.LAND, edgecolor='k')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    # ax.add_feature(cfeature.RIVERS)

    # add_gridline_labels(ax, labels_set=[10, 20, 30, 40, 50, 60, 70], side='left')
    # add_gridline_labels(ax, labels_set=[0, 10, 20, 30, 40, 50, 60], side='right')
    # add_gridline_labels(ax, labels_set=[60, 100, 140, 180], side='top')
    # add_gridline_labels(ax, labels_set=[80, 100, 120, 140, 160], side='bottom')
    gl = ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, linewidth=1,
                      color='grey', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([96, 98, 100, 102, 104])
    gl.ylocator = mticker.FixedLocator([20, 24, 26, 28, 30])

    # add ticks manually
    ax.text(-0.01, 0.9, '30°N', ha='right', va='center', transform=ax.transAxes, fontsize=14)
    ax.text(-0.01, 0.39, '25°N', ha='right', va='center', transform=ax.transAxes, fontsize=14)
    # ax.text(-0.01, 0.05, '20°N', ha='right', va='center', transform=ax.transAxes, fontsize=14)
    # ax.text(-0.01, 0.04, '15°N', ha='right', va='center', transform=ax.transAxes, fontsize=14)

    ax.text(0.04, -0.02, '95°E', ha='center', va='top', transform=ax.transAxes, fontsize=14)
    ax.text(0.46, -0.02, '100°E', ha='center', va='top', transform=ax.transAxes, fontsize=14)
    ax.text(0.86, -0.02, '105°E', ha='center', va='top', transform=ax.transAxes, fontsize=14)

    # ax.set_xticks([80, 100, 120, 140, 160], crs=ccrs.PlateCarree())
    # ax.set_yticks([0, 10, 20, 30, 40, 50, 60], crs=ccrs.PlateCarree())
    # lon_formatter = LongitudeFormatter(number_format='.0f', degree_symbol='°')
    # lat_formatter = LatitudeFormatter(number_format='.0f', degree_symbol='°')
    # ax.xaxis.set_major_formatter(lon_formatter)
    # ax.yaxis.set_major_formatter(lat_formatter)

    return ax

def plotcosmo04sm_notick(ax):
    ax.set_extent([95, 106, 21.9, 31.1], crs=ccrs.PlateCarree())  # for extended 12km domain
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)

    gl = ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False, linewidth=1,
                      color='grey', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([95, 100, 105])
    gl.ylocator = mticker.FixedLocator([20, 25, 30])

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
    file = "/project/pr133/rxiang/data/cosmo/EAS04_ctrl/1h/TOT_PREC/01_TOT_PREC.nc"
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






