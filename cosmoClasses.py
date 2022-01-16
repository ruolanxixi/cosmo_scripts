# -------------------------------------------------------------------------------
# Definition of classes used for COSMO output analysis
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# Modules
#
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from cosmoFunctions import listFiles, listDirectories, haversine


# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# Simulation class
#
class Simulation():

    def __init__(self, name=None, rootDir=None):
        self.name = name
        self.rootDir = rootDir
        self.ncHandles = [['lbfd'], ['lffd', '0.nc']]
        self.dataNames = [
            'TQC', 'TQV', 'TQI',
            'RELHUM_2M', 'T_2M',
            'TOT_PREC', 'TQI', 'CLCL',
            'ATHB_T', 'ASOD_T', 'ASOB_T',
            'ATHB_S', 'ATHD_S', 'ASOB_S', 'ASWDIFD_S', 'ASWDIFU_S', 'ASWDIR_S',
            'ALHFL_S', 'ASHFL_S',
            'U', 'V', 'W', 'T', 'P', 'QV', 'QC',
            'W_SO', 'SOILTYP',
            'CAPE_ML', 'CIN_ML'
        ]
        self.data = {}
        self.postproc = {}
        self.lat = None
        self.lon = None
        self.Nx = 0
        self.Ny = 0
        self.dx = 0
        self.dy = 0
        self.getAvailData()

    def getAvailData(self):
        self.dirs = {}
        #
        # Travel through the directory tree
        for dirPath, subDirs, fileList in os.walk(self.rootDir, followlinks=True):
            #
            # List any raw .nc files that respect the ncHandles criteria
            rawNCfiles = [os.path.join(dirPath, f) for f in fileList
                          if any([all([handle in f for handle in handles]) for handles in self.ncHandles])]
            # List any extracted .nc files that are listed in dataNames
            extNCfiles = [os.path.join(dirPath, f) for f in fileList
                          if any([name + '.nc' == f for name in self.dataNames])]
            if (rawNCfiles != []) or (extNCfiles != []):
                #
                # dirPath contains .nc files we are interested in.
                # Add it to the directory dictionary.
                dirName = os.path.relpath(dirPath, self.rootDir)
                # fileInit = [handles[0] for handles in self.ncHandles if handles[0] in rawNCfiles[0]][0]
                self.dirs[dirName] = OutputData(name=dirName, absPath=dirPath, rawNCfiles=rawNCfiles,
                                                extNCfiles=extNCfiles)

    def load(self, which='ext'):
        if which == 'ext':
            self.loadExt()
        elif which == 'raw':
            self.loadRaw()
        else:
            raise RuntimeError('Key not implemented')
        #
        self.data['latMM'] = [self.lat.min().values, self.lat.max().values]
        self.data['lonMM'] = [self.lon.min().values, self.lon.max().values]
        self.data['area'] = haversine([self.data['latMM'][0], self.data['lonMM'][0]],
                                      [self.data['latMM'][0], self.data['lonMM'][1]]) \
                            * haversine([self.data['latMM'][0], self.data['lonMM'][0]],
                                        [self.data['latMM'][1], self.data['lonMM'][0]])
        #
        if 'SOILTYP' in self.data:
            # exclude water and sea ice
            self.data['nlandpoints'] = np.sum(self.data['SOILTYP'].values != 9) + np.sum(
                self.data['SOILTYP'].values != 10)

    def loadExt(self):
        #
        # Load extracted files from all subdirectories
        for dirname, subDir in self.dirs.items():
            #
            subDir.loadExt()
            #
            if subDir.loaded():
                # Add data variables to dict
                for dataName in subDir.extData:
                    self.data[dataName] = subDir.extData[dataName]
                #
                # Add latitude and longitude (overwriting)
                self.lat = subDir.extData[dataName].coords['lat']
                self.lon = subDir.extData[dataName].coords['lon']
                self.Nx = len(self.lon)
                self.Ny = len(self.lat)
                self.data['numpoints'] = self.Nx * self.Ny
                #
                self.dy = (self.lat[1, 0] - self.lat[0, 0]).values * 110 * 1e3
                self.dx = (self.lon[0, 1] - self.lon[0, 0]).values * 110 * 1e3

    def loadRaw(self):
        #
        # Load raw files from selected subdirectories
        for dirname, subDir in self.dirs.items():
            #
            subDir.loadRaw()
            #
            if subDir.loaded():
                # Add selected data variables to dict
                for dataName in subDir.rawData.data_vars:
                    if dataName in self.dataNames:
                        self.data[dataName] = subDir.rawData[dataName]
                #
                # Add latitude and longitude (overwriting)
                self.lat = subDir.rawData.coords['lat']
                self.lon = subDir.rawData.coords['lon']
                #
                self.dy = (self.lat[1, 0] - self.lat[0, 0]) * 110 * 1e3
                self.dx = (self.lon[0, 1] - self.lon[0, 0]) * 110 * 1e3


#
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# Output data class
#
class OutputData():

    def __init__(self, name=None, absPath=None, rawNCfiles=None, extNCfiles=None):
        self.name = name
        self.absPath = absPath
        self.fileInit = 'lffd'
        self.dateStart = None
        self.dateEnd = None
        self.dateFreq = None
        self.timeStamps = None
        self.zlevs = None
        self.loadStatus = 0
        self.extNCfiles = extNCfiles
        self.extData = {}
        if rawNCfiles is None:
            self.rawNCfiles = listFiles(absPath, wc='.nc')
        else:
            self.rawNCfiles = rawNCfiles

    def toLoad(self):
        #
        # set load status
        self.loadStatus = 1

    def loaded(self):
        #
        # return true if loaded
        return self.loadStatus == 2

    def setOptions(self, dates=None, zlevs=None):
        if dates is not None:
            self.setDates(dates)
        if zlevs is not None:
            self.setZlevs(zlevs)

    def setDates(self, dates):
        self.dateStart = dates[0]
        self.dateEnd = dates[1]
        if len(dates) > 2:
            self.dateFreq = dates[2]
            self.timeStamps = self.makeTimestamps()
        self.toLoad()

    def setZlevs(self, zlevs):
        self.zlevs = zlevs

    def makeTimestamps(self, dateStart=None, dateEnd=None, dateFreq=None):
        if dateStart is None:
            dateStart = self.dateStart
        if dateEnd is None:
            dateEnd = self.dateEnd
        if dateFreq is None:
            dateFreq = self.dateFreq
        #
        timeStamps = np.arange(dateStart, \
                               dateEnd, \
                               dateFreq).tolist()
        return timeStamps

    def loadExt(self):
        # load files containing variables extracted with ncrcat
        #
        if self.loadStatus == 1:
            for extFile in self.extNCfiles:
                #
                # load dataset
                # dataIn = xr.open_dataset(extFile, chunks={'time': 16})[os.path.basename(extFile)[0:-3]]
                # dataIn = xr.open_dataset(extFile, chunks={'time': 16, 'latitude': 16, 'longitude': 16})[os.path.basename(extFile)[0:-3]]
                dataIn = xr.open_dataset(extFile)[os.path.basename(extFile)[0:-3]]
                #
                # cut possibly larger than wanted dataset
                if (self.dateStart is not None) and (self.dateEnd is not None):
                    dataIn = dataIn.sel(time=slice(self.dateStart, self.dateEnd))
                if self.zlevs is not None:
                    dataIn = dataIn.sel(altitude=self.zlevs)
                #
                # load the dataset
                self.extData[os.path.basename(extFile)[0:-3]] = dataIn
            self.loadStatus = 2
            print('Loaded ext data from ' + self.name)

    def loadRaw(self):
        # load raw lffd files
        #
        if self.loadStatus == 1:
            timeStamps = self.timeStamps
            if (timeStamps is None):
                # load all the raw files
                rawNCfiles = self.rawNCfiles
            else:
                # load only the raw files with requested time stamp
                rawNCfiles = []
                for timeStamp in timeStamps:
                    fileName = self.fileInit + timeStamp.strftime('%Y%m%d%H%M%S') + '.nc'
                    filePath = os.path.join(self.absPath, fileName)
                    if filePath in self.rawNCfiles:
                        rawNCfiles.append(filePath)
                    else:
                        raise RuntimeError('File not found {0:s}'.format(filePath))
            #
            # self.rawData = xr.open_mfdataset(rawNCfiles, combine='by_coords', parallel=True, chunks={'time': 16})
            # self.rawData = xr.open_mfdataset(rawNCfiles, combine='by_coords', parallel=True, chunks={'time': 16, 'latitude': 16, 'longitude': 16})
            self.rawData = xr.open_mfdataset(rawNCfiles, combine='by_coords', parallel=True)
            self.loadStatus = 2
            print('Loaded raw data from ' + self.name)


#
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# Comparison matrix data class
#
class ComparisonMatrix():

    def __init__(self, case=None, comp='diff', dataName=None):
        self.dataName = dataName
        self.comp = comp
        self.min = 0
        self.max = 0
        self.M = None
        self.Ml = None
        if (case is not None):
            self.buildMatrix(case)
            self.computMinMax(case)

    def buildMatrix(self, case):
        num_cases = len(case)
        self.M = [[None for j in range(num_cases)] for i in range(num_cases)]
        self.Ml = [[None for j in range(num_cases)] for i in range(num_cases)]
        for row in range(num_cases - 1):
            for col in range(row + 1, num_cases):
                #
                self.Ml[row][col] = '[{0:d}-{1:d}]'.format(row + 1, col + 1)
                if self.comp == 'diff':
                    self.M[row][col] = \
                        PostprocessData(
                            data=case[row].postproc[self.dataName].data - case[col].postproc[self.dataName].data)

    def computMinMax(self, case):
        # min max of the data fields (NOT of the comparison)
        # NOT INVERTED
        if hasattr(case[0].postproc[self.dataName].data, 'altitude'):
            self.min = {}
            self.max = {}
            zlevs = case[0].postproc[self.dataName].data.altitude.values
            for zlev in zlevs:
                self.min[zlev] = np.min([c.postproc[self.dataName].min[zlev] for c in case], 0)
                self.max[zlev] = np.max([c.postproc[self.dataName].max[zlev] for c in case], 0)
        elif hasattr(case[0].postproc[self.dataName].data, 'soil1'):
            self.min = {}
            self.max = {}
            slevs = case[0].postproc[self.dataName].data.soil1.values
            for slev in slevs:
                self.min[slev] = np.min([c.postproc[self.dataName].min[slev] for c in case], 0)
                self.max[slev] = np.max([c.postproc[self.dataName].max[slev] for c in case], 0)
        else:
            self.min = {0: np.min([c.postproc[self.dataName].min[0] for c in case], 0)}
            self.max = {0: np.max([c.postproc[self.dataName].max[0] for c in case], 0)}


#
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# Postprocess data class
#
class PostprocessData():

    def __init__(self, name=None, data=None):
        self.name = name
        self.data = data
        self.best = False
        self.worst = False
        if (data is not None):
            self.computMin()
            self.computMax()
            self.computTot()

    def computMin(self):
        if hasattr(self.data, 'altitude'):
            self.min = {}
            for zlev in self.data.altitude.values:
                self.min[zlev] = self.data.sel(altitude=zlev).min().values
        elif hasattr(self.data, 'soil1'):
            self.min = {}
            for slev in self.data.soil1.values:
                self.min[slev] = self.data.sel(soil1=slev).min().values
        else:
            data = self.data.min()
            if hasattr(data, 'values'):
                self.min = {0: data.values}
            else:
                self.min = {0: data}

    def computMax(self):
        if hasattr(self.data, 'altitude'):
            self.max = {}
            for zlev in self.data.altitude.values:
                self.max[zlev] = self.data.sel(altitude=zlev).max().values
        elif hasattr(self.data, 'soil1'):
            self.max = {}
            for slev in self.data.soil1.values:
                self.max[slev] = self.data.sel(soil1=slev).max().values
        else:
            data = self.data.max()
            if hasattr(data, 'values'):
                self.max = {0: data.values}
            else:
                self.max = {0: data}

    def computTot(self):
        if hasattr(self.data, 'altitude'):
            self.tot = {}
            for zlev in self.data.altitude.values:
                self.tot[zlev] = self.data.sel(altitude=zlev).sum().values
        elif hasattr(self.data, 'soil1'):
            self.tot = {}
            for slev in self.data.soil1.values:
                self.tot[slev] = self.data.sel(soil1=slev).sum().values
        else:
            data = self.data.sum()
            if hasattr(data, 'values'):
                self.tot = {0: data.values}
            else:
                self.tot = {0: data}
#
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# Memos
##
# for name, params in self.standardData.items():
#    directory = params['dir']
#    filesWC   = params['files']
#    dataDir = os.path.join(self.rootDir, directory)
#    if os.path.isdir(dataDir):
#        # Data directory exists
#        #
#        ncFiles = listFiles(dataDir, wc=filesWC, anyAll=all)
#        subDirs = listDirectories(dataDir)
#        #
#        if (ncFiles != [] or subDirs != []):
#            # It contains either .nc files or subdirectories or both.
#            #
#            # Add one element to the 'dirs' dictionary...
#            self.dirs[name] = OutputData(name=name, absDir=dataDir, ncFiles=ncFiles, fileInit=filesWC[0])
#            # ...and add one key to the class to allow access in both
#            # ways.
#            self.__dict__['d'+name] = self.dirs[name]
#        #
#        # Now check the (possible) subdirectories (**ONLY** one level)
#        self.dirs[name].dirs = {}
#        for subDir in subDirs:
#            ncFiles = listFiles(subDir, wc=filesWC, anyAll=all)
#            if ncFiles != []:
#                # It contains .nc files
#                #
#                # Add one element to the 'dirs' dictionary...
#                subDirName = os.path.basename(subDir)
#                self.dirs[name].dirs[subDirName] = OutputData(name=subDirName, absDir=subDir, ncFiles=ncFiles, fileInit=filesWC[0])
#                # ...and add one key to the class to allow access in
#                # both ways.
#                self.__dict__['d'+name].__dict__['d'+subDirName] = self.dirs[name].dirs[subDirName]
