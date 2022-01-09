#-------------------------------------------------------------------------------
# Definition of functions used for COSMO output analysis
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Modules
#
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# List files
#
def listFiles(absDir, wc=None, anyAll=any):
    """
    List files in a directory.
    Optionally filter by (any or all) wildcard(s)
    """
    if wc is None:
        # no wildcard
        files = [f.path for f in os.scandir(absDir) \
                if f.is_file()]
    elif type(wc) is str:
        # single wildcard
        files = [f.path for f in os.scandir(absDir) \
                if (f.is_file() and wc in f.name)]
    elif type(wc) is list:
        # list of wildcards
        files = [f.path for f in os.scandir(absDir) \
                if (f.is_file() and anyAll([w in f.name for w in wc]))]
    else:
        raise NotImplementedError('Wildcard type not implemented.')
    return files
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# List directories
#
def listDirectories(absDir, wc=None, anyAll=any):
    """
    List directories in a directory.
    Optionally filter by (any or all) wildcard(s)
    """
    if wc is None:
        # no wildcard
        directories = [f.path for f in os.scandir(absDir) \
                      if f.is_dir()]
    elif type(wc) is str:
        # single wildcard
        directories = [f.path for f in os.scandir(absDir) \
                      if (f.is_dir() and wc in f.name)]
    elif type(wc) is list:
        # list of wildcards
        directories = [f.path for f in os.scandir(absDir) \
                      if (f.is_dir() and anyAll([w in f.name for w in wc]))]
    else:
        raise NotImplementedError('Wildcard type not implemented.')
    return directories
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Haversine formula
#
def haversine(latlon1, latlon2):
    """
    Compute the distance in meters between two points on a sphere
    """
    r = 6371.0e3
    lat1 = latlon1[0]*np.pi/180.; lon1 = latlon1[1]*np.pi/180.
    lat2 = latlon2[0]*np.pi/180.; lon2 = latlon2[1]*np.pi/180.
    distance = 2*r*np.arcsin(np.sqrt(
        np.sin((lat2-lat1)/2.)**2
        + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2.)**2
        ))
    return distance
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Haversine formula
#
def soilSaturation(soiltyp, w_so):
    """
    Compute the soil saturation
    given the soil type (soiltyp) and soil moisture (w_so)
    """
    soil1 = w_so.coords['soil1'].values
    nlayers = soil1.shape[0]

    wso_vol = w_so.copy()      # initialize
    wso_vol.values[:,:,:] = 0. # init
    soil1_bnds = np.zeros((nlayers, 2))
    soil1_dz   = np.zeros(nlayers)

    for ilayer in range(nlayers):
        soil1_dz[ilayer] = (soil1[ilayer] - soil1_bnds[ilayer, 0])*2
        soil1_bnds[ilayer,   1] = soil1_bnds[ilayer, 0] + soil1_dz[ilayer]
        if ilayer+1 < nlayers:
            soil1_bnds[ilayer+1, 0] = soil1_bnds[ilayer, 1]
        #
        # convert to volumetric soil water [m**3/m**3]
        wso_vol[ilayer, :,:] = w_so.values[ilayer, :,:] / soil1_dz[ilayer]

    # cosmo-org definitions for porosity volume (cporv) and field capacity (cfcap) from sfc_terra_data.f90
    # soil type:   ice       rock     sand    sandy     loam     clay     clay     peat        sea         sea
    # (by index)                               loam              loam                        water         ice
    cporv = {1:1.0e-10, 2:1.0e-10, 3:0.364, 4:0.445, 5:0.455, 6:0.475, 7:0.507, 8:0.863, 9:1.0e-10, 10:1.0e-10}
    cfcap = {1:1.0e-10, 2:1.0e-10, 3:0.196, 4:0.260, 5:0.340, 6:0.370, 7:0.463, 8:0.763, 9:1.0e-10, 10:1.0e-10}

    # pore volume [m**3/m**3]
    # this is equivalent to a dict lookup for loop over the 2d field, but faster
    porevolume_2d = np.vectorize(cporv.get)(soiltyp.values[0,:,:])

    soilSaturation = wso_vol / porevolume_2d

    return soilSaturation
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Plot comparison
#
def plotComparison(case, comp, dataName, show=False, imgsDir=None, zlev=None, slev=None):
    """
    Plot the comparison between two (or more) different cases
    """
    #
    print('\tPlotting ' + dataName)
    #
    fig = plt.figure(313, figsize=(len(case)*10, 7)); fig.clf()
    if show:
        plt.show(block=False)
    #
    if (zlev is not None) and (zlev not in case[0].postproc[dataName].data.altitude.values):
        raise RuntimeError('zlev {0:f} not in range: '.format(zlev) + str(case[0].postproc[dataName].data.altitude.values))
    #
    if (slev is not None) and (slev not in case[0].postproc[dataName].data.soil1.values):
        raise RuntimeError('slev {0:f} not in range: '.format(slev) + str(case[0].postproc[dataName].data.soil1.values))
    #
    ax = (len(case)+1)*[None]
    im = (len(case)+1)*[None]
    for iplot in range(len(case)+1):
        ax[iplot] = plt.subplot(1, len(case)+1, iplot+1, projection=ccrs.PlateCarree())
        #
        if iplot != len(case):
            if zlev is not None:
                toPlot = case[iplot].postproc[dataName].data.sel(altitude=zlev)
                dataMin = comp.postproc[dataName].min[zlev]
                dataMax = comp.postproc[dataName].max[zlev]
                dataTot = case[iplot].postproc[dataName].tot[zlev]
            elif slev is not None:
                toPlot = case[iplot].postproc[dataName].data.sel(soil1=slev)
                dataMin = comp.postproc[dataName].min[slev]
                dataMax = comp.postproc[dataName].max[slev]
                dataTot = case[iplot].postproc[dataName].tot[slev]
            else:
                toPlot = case[iplot].postproc[dataName].data
                dataMin = comp.postproc[dataName].min[0]
                dataMax = comp.postproc[dataName].max[0]
                dataTot = case[iplot].postproc[dataName].tot[0]
            ##dataMean = dataTot / (case[iplot].postproc[dataName].data.shape[0] * case[iplot].postproc[dataName].data.shape[1])
            ##if dataMax > 10*dataMean:
            ##    dataMax = 10*dataMean
            #
            ax[iplot].coastlines(resolution='50m', color='white')
            im[iplot] = toPlot.plot(ax=ax[iplot], add_colorbar=False, add_labels=False, \
                                     vmin=dataMin, vmax=dataMax)
            #ax[iplot].set_title(case[iplot].name + ' - land mean: {0:.2e}'.format(dataTot/case[iplot].data['nlandpoints']))
            ax[iplot].set_title(case[iplot].name + ' - mean: {0:.2e} / m$^2$'.format(dataTot/case[iplot].data['area']))
            #ax[iplot].set_title(case[iplot].name + ' - total: {0:.2e}'.format(dataTot))
        else:
            if zlev is not None:
                toPlot = comp.postproc[dataName].data.sel(altitude=zlev)
                dataTot = comp.postproc[dataName].tot[zlev]
            elif slev is not None:
                toPlot = comp.postproc[dataName].data.sel(soil1=slev)
                dataTot = comp.postproc[dataName].tot[slev]
            else:
                toPlot = comp.postproc[dataName].data
                dataTot = comp.postproc[dataName].tot[0]
            ax[iplot].coastlines(resolution='50m', color='black')
            im[iplot] = toPlot.plot(ax=ax[iplot], add_colorbar=False, add_labels=False)
            #ax[iplot].set_title(comp.name + ' - land mean: {0:.2e}'.format(dataTot/case[0].data['nlandpoints']))
            ax[iplot].set_title(comp.name + ' - mean: {0:.2e} / m$^2$'.format(dataTot/case[0].data['area']))
            #ax[iplot].set_title(comp.name + ' - total: {0:.2e}'.format(dataTot))
        #
        ax[iplot].set_xticks(np.arange(-180,180,2))
        ax[iplot].set_yticks(np.arange( -90, 90,2))
        #
        if iplot != 0:
            ax[iplot].set_yticklabels('')
        ax[iplot].set_xlabel('')
        ax[iplot].set_ylabel('')
        ax[iplot].set_xlim(comp.data['lonMM'])
        ax[iplot].set_ylim(comp.data['latMM'])
    #
    suptitle = dataName.replace('_', ' ')
    if zlev is not None:
        suptitle += ' - altitude: {0:d} m'.format(int(zlev))
    elif slev is not None:
        suptitle += ' - soil depth: {0:.3f} m'.format(slev)
    fig.suptitle(suptitle)
    #
    axS = ax[0].get_position().x0
    axB = ax[0].get_position().y0
    axW = (ax[-2].get_position().x1 - ax[0].get_position().x0)
    axS += axW*0.25
    axW  = axW*0.5
    axH = 0.02
    cbaxes0 = fig.add_axes([axS, axB-4*axH, axW, axH])
    cb0 = plt.colorbar(im[0], cax=cbaxes0, orientation="horizontal")
    #cb.set_label(label=dataName, y=0.5, ha='right')
    #
    axS = ax[-1].get_position().x0
    axB = ax[-1].get_position().y0
    axW = ax[-1].get_position().x1 - ax[-1].get_position().x0
    axS += axW/10
    axW  = axW/10*8
    axH = 0.02
    cbaxes1 = fig.add_axes([axS, axB-4*axH, axW, axH])
    cb1 = plt.colorbar(im[-1], cax=cbaxes1, orientation="horizontal")
    #
    if imgsDir is not None:
        #plt.tight_layout()
        if zlev is not None:
            plt.savefig(os.path.join(imgsDir, dataName + '_A{0:06d}.png'.format(int(zlev))), bbox_inches='tight')
        elif slev is not None:
            plt.savefig(os.path.join(imgsDir, dataName + '_D{0:06d}.png'.format(int(slev*1000))), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(imgsDir, dataName + '.png'), bbox_inches='tight')
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Plot comparison of 1D height-dependent data
#
def plotComparison1D(case, dataName, show=False, imgsDir=None, xlims=None, ylims=None):
    """
    Plot the comparison between two (or more) different cases
    """
    #
    print('\tPlotting ' + dataName)
    #
    fig = plt.figure(314); fig.clf()
    if show:
        plt.show(block=False)
    #
    for icase in range(len(case)):
        #
        plt.plot(case[icase].postproc[dataName].data, case[icase].postproc[dataName].data.altitude, label=case[icase].name)
    #
    if xlims is not None:
        plt.xlim(xlims)
    if ylims is not None:
        plt.ylim(ylims)
    else:
        plt.ylim([0, case[0].postproc[dataName].data.altitude[-1]])
    plt.xlabel(case[0].postproc[dataName].data.name)
    plt.ylabel('Altitude [m]')
    #plt.title(dataName.replace('_', ' '))
    plt.legend()
    #
    if imgsDir is not None:
        #plt.tight_layout()
        plt.savefig(os.path.join(imgsDir, dataName + '.png'), bbox_inches='tight')
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Plot comparison matrix of 2D data
#
def plotComparisonMatrix(case, comp, dataName, meanType='mean', zlev=None, slev=None, show=False, imgsDir=None):
    """
    Plot the comparison between (more than) two different cases
    """
    #
    print('\tPlotting ' + dataName)
    #
    num_plots = len(case)
    fig_dpi=100
    fig = plt.figure(figsize=((num_plots+1)*5, num_plots*5)); fig.clf()
    if show:
        plt.show(block=False)
    #
    if (zlev is not None) and (zlev not in case[0].postproc[dataName].data.altitude.values):
        raise RuntimeError('zlev {0:f} not in range: '.format(zlev) + str(case[0].postproc[dataName].data.altitude.values))
    #
    if (slev is not None) and (slev not in case[0].postproc[dataName].data.soil1.values):
        raise RuntimeError('slev {0:f} not in range: '.format(slev) + str(case[0].postproc[dataName].data.soil1.values))
    #
    fig, ax = plt.subplots(num_plots, num_plots+1, num=fig.number, subplot_kw={'projection': ccrs.PlateCarree()})
    for row in range(num_plots):
        icase=row
        #
        for col in range(num_plots+1):
            #
            if (col <= row):
                # below diagonal
                #
                ax[row][col].set_xlim([0,1])
                ax[row][col].set_ylim([0,1])
                ax[row][col].outline_patch.set_visible(False)
            #
            elif (col == row+1):
                # diagonal
                #
                if zlev is not None:
                    toPlot  = case[icase].postproc[dataName].data.sel(altitude=zlev)
                    dataMin = comp.postproc[dataName].min[zlev]
                    dataMax = comp.postproc[dataName].max[zlev]
                    dataTot = case[icase].postproc[dataName].tot[zlev]
                elif slev is not None:
                    toPlot  = case[icase].postproc[dataName].data.sel(soil1=slev)
                    dataMin = comp.postproc[dataName].min[slev]
                    dataMax = comp.postproc[dataName].max[slev]
                    dataTot = case[icase].postproc[dataName].tot[slev]
                else:
                    toPlot  = case[icase].postproc[dataName].data
                    dataMin = comp.postproc[dataName].min[0]
                    dataMax = comp.postproc[dataName].max[0]
                    dataTot = case[icase].postproc[dataName].tot[0]
                #
                ax[row][col].coastlines(resolution='50m', color='white')
                im = toPlot.plot(ax=ax[row][col], add_colorbar=False, add_labels=False, \
                                 vmin=dataMin, vmax=dataMax)
                #
                ax[row][col].set_xticks(np.arange(-180,180,5))
                ax[row][col].set_yticks(np.arange( -90, 90,5))
                ax[row][col].set_xlabel('Lon')
                ax[row][col].set_ylabel('Lat')
                ax[row][col].set_xlim(comp.data['lonMM'])
                ax[row][col].set_ylim(comp.data['latMM'])
                #
                # Colorbar
                axPos = ax[row][col].get_position()
                axW = axPos.x1 - axPos.x0
                axH = axPos.y1 - axPos.y0
                cbPos = [axPos.x0+axW/20., axPos.y0+axH/10, axW/10*4., axH/20.]
                cbaxes = fig.add_axes(cbPos)
                cb = plt.colorbar(im, cax=cbaxes, orientation='horizontal', extend='both')
                cbxtick_obj = plt.getp(cb.ax.axes, 'xticklabels')
                plt.setp(cbxtick_obj, color='w')
                #
                # Text on left
                name      = case[icase].name
                if meanType == 'mean':
                    meanValue = 'mean: {0:.2e}'.format(dataTot / case[icase].data['numpoints'])
                elif meanType == 'mean_land':
                    meanValue = 'land mean: {0:.2e}'.format(dataTot / case[icase].data['numpoints_land'])
                elif meanType == 'mean_sea':
                    meanValue = 'sea mean: {0:.2e}'.format(dataTot / case[icase].data['numpoints_sea'])
                elif meanType == 'total':
                    meanValue = 'total: {0:.2e}'.format(dataTot)
                textBox = name + '\n' + meanValue
                #
                ax[row][col-1].text(0.2,0.5, textBox)
            #
            else:
                # above the diagonal
                if zlev is not None:
                    toPlot  = comp.postproc[dataName].M[row][col-1].data.sel(altitude=zlev)
                    dataTot = comp.postproc[dataName].M[row][col-1].tot[zlev]
                elif slev is not None:
                    toPlot  = comp.postproc[dataName].M[row][col-1].data.sel(soil1=slev)
                    dataTot = comp.postproc[dataName].M[row][col-1].tot[slev]
                else:
                    toPlot  = comp.postproc[dataName].M[row][col-1].data
                    dataTot = comp.postproc[dataName].M[row][col-1].tot[0]
                ax[row][col].coastlines(resolution='50m', color='black')
                im = toPlot.plot(ax=ax[row][col], add_colorbar=False, add_labels=False)
                #
                label     = comp.postproc[dataName].Ml[row][col-1]
                if meanType == 'mean':
                    meanValue = 'mean: {0:.2e}'.format(dataTot / case[icase].data['numpoints'])
                elif meanType == 'mean_land':
                    meanValue = 'land mean: {0:.2e}'.format(dataTot / case[icase].data['numpoints_land'])
                elif meanType == 'mean_sea':
                    meanValue = 'sea mean: {0:.2e}'.format(dataTot / case[icase].data['numpoints_sea'])
                elif meanType == 'total':
                    meanValue = 'total: {0:.2e}'.format(dataTot)
                textBox = label + ' ' + meanValue
                #
                ax[row][col].set_xlabel(textBox)
                ax[row][col].set_xticks(np.arange(-180,180,5))
                ax[row][col].set_yticks(np.arange( -90, 90,5))
                ax[row][col].set_xticklabels('')
                ax[row][col].set_yticklabels('')
                ax[row][col].set_ylabel('')
                ax[row][col].set_xlim(comp.data['lonMM'])
                ax[row][col].set_ylim(comp.data['latMM'])
                #
                # Colorbar
                axPos = ax[row][col].get_position()
                axW = axPos.x1 - axPos.x0
                axH = axPos.y1 - axPos.y0
                cbPos = [axPos.x0+axW/20., axPos.y0+axH/10, axW/10*4., axH/20.]
                cbaxes = fig.add_axes(cbPos)
                cb = plt.colorbar(im, cax=cbaxes, orientation='horizontal')

    #
    if imgsDir is not None:
        #plt.tight_layout()
        if zlev is not None:
            plt.savefig(os.path.join(imgsDir, 'm_' + dataName + '_A{0:06d}.png'.format(int(zlev))), bbox_inches='tight', dpi=fig_dpi)
        elif slev is not None:
            plt.savefig(os.path.join(imgsDir, 'm_' + dataName + '_D{0:06d}.png'.format(int(slev*1000))), bbox_inches='tight', dpi=fig_dpi)
        else:
            plt.savefig(os.path.join(imgsDir, 'm_' + dataName + '.png'), bbox_inches='tight', dpi=fig_dpi)
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Calculate counts frequency and cumulative frequency
#
def get_count(val, bins=100):
    counts, base = np.histogram(val, bins=bins)
    return base, counts

def get_freq(val):
    weights = np.ones_like(val)/float(len(val))
    max_int = int(np.max(val)) + 1 
    num_bins = 10 * max_int
    freq, base = np.histogram(val, bins=num_bins, range=(0, max_int),
                              weights=weights)
    base = base[0:-1]
    return base, freq

def get_cumfreq(val, nvals=None):
    if nvals == None:
        nvals = len(val)
    weights = np.ones_like(val)/nvals
    max_int = int(np.max(val)) + 1 
    num_bins = 20 * max_int
    freq, base = np.histogram(val, bins=num_bins, range=(0, max_int),
                              weights=weights)
    freq = freq[::-1]
    cumfreq = np.cumsum(freq)
    cumfreq = cumfreq[::-1]
    base = base[0:-1]
    return base, cumfreq
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Re-gridding
#
def xrDataset(var, lat, lon, time=None):
    #
    import xarray as xr
    # create a clean xarray dataset
    # given a variable and its coordinates
    #
    varName   = var[0]
    varValue  = var[1]
    varCoords = ['y', 'x']
    #
    gwb = gridWithBoundaries(lat,lon)
    #
    coords = {
            'lat':   (['y',  'x'  ], lat  ),
            'lon':   (['y',  'x'  ], lon  ),
            'lat_b': (['y_b','x_b'], gwb['lat_b'] ),
            'lon_b': (['y_b','x_b'], gwb['lon_b'] )
            }
    #
    if time is not None:
        varCoords = ['time'] + varCoords
        coords['time'] = ('time', time)
    #
    dset = xr.Dataset({varName: (varCoords, varValue)}, coords=coords)

    return dset

def gridWithBoundaries_old(lat, lon):
    #
    # compute cell boundaries needed for conservative regridding
    lat_b = (lat[:,0][1:]+lat[:,0][:-1])/2
    lat_b = np.insert(lat_b, 0, lat[:,0][ 0] - (lat[:,0][ 1] - lat[:,0][ 0])/2, axis=0)
    lat_b = np.append(lat_b,    lat[:,0][-1] + (lat[:,0][-1] - lat[:,0][-2])/2)
    lat_b = np.outer(lat_b, np.ones((lat.shape[1]+1,1)))
    lon_b = (lon[0,:][1:]+lon[0,:][:-1])/2
    lon_b = np.insert(lon_b, 0, lon[0,:][ 0] - (lon[0,:][ 1] - lon[0,:][ 0])/2, axis=0)
    lon_b = np.append(lon_b,    lon[0,:][-1] + (lon[0,:][-1] - lon[0,:][-2])/2)
    lon_b = np.outer(np.ones((lat.shape[0]+1,1)), lon_b)
    grid = {
            'lat': lat,
            'lon': lon,
            'lat_b': lat_b,
            'lon_b': lon_b
            }
    return grid

def gridWithBoundaries(lat, lon):
    # compute cell boundaries needed for conservative regridding
    # improved by Christian Zeman for non-regular grids
    lat_sy = 0.5 * (lat[1:,:]+lat[:-1,:])
    lat_s = 0.5 * (lat_sy[:,1:] + lat_sy[:,:-1])
    ny, nx = lat.shape
    lat_b = np.zeros((ny+1, nx+1))
    lat_b[1:-1,1:-1] = lat_s
    lat_b[0,:] = lat_b[1,:] - (lat_b[2,:] - lat_b[1,:])
    lat_b[-1,:] = lat_b[-2,:] + (lat_b[-2,:] - lat_b[-3,:])
    lat_b[:,0] = lat_b[:,1] - (lat_b[:,2] - lat_b[:,1])
    lat_b[:,-1] = lat_b[:,-2] + (lat_b[:,-2] - lat_b[:,-3])
    lon_sy = 0.5 * (lon[1:,:]+lon[:-1,:])
    lon_s = 0.5 * (lon_sy[:,1:] + lon_sy[:,:-1])
    ny, nx = lon.shape
    lon_b = np.zeros((ny+1, nx+1))
    lon_b[1:-1,1:-1] = lon_s
    lon_b[0,:] = lon_b[1,:] - (lon_b[2,:] - lon_b[1,:])
    lon_b[-1,:] = lon_b[-2,:] + (lon_b[-2,:] - lon_b[-3,:])
    lon_b[:,0] = lon_b[:,1] - (lon_b[:,2] - lon_b[:,1])
    lon_b[:,-1] = lon_b[:,-2] + (lon_b[:,-2] - lon_b[:,-3])
    grid = {
            'lat': lat,
            'lon': lon,
            'lat_b': lat_b,
            'lon_b': lon_b
            }
    return grid
#-------------------------------------------------------------------------------
