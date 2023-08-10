import numpy as np
from osgeo import gdal
from gdalconst import *

def writeTiff(im_data,im_width, im_height,im_bands,im_geotrans,im_proj,path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset

def climate_Velocity(filenameP,filenameF,distance_unit,time_interval,path):
    # ####################################################################
    # ####################################################################
    # 1 read tiff
    # present tiff
    gdal.AllRegister()
    dataset_p = gdal.Open(filenameP, GA_ReadOnly)
    nx,ny = dataset_p.RasterXSize,dataset_p.RasterYSize  # 栅格矩阵的行数？
    present = dataset_p.ReadAsArray(0, 0, nx, ny)  # present data

    # information for writing tiff files
    im_bands = dataset_p.RasterCount  # band numbers
    im_proj = dataset_p.GetProjection()  # projection
    im_geotrans = dataset_p.GetGeoTransform()  # location
    im_width, im_height = present.shape[1], present.shape[0]


    # future tiff
    gdal.AllRegister()
    dataset_f = gdal.Open(filenameF, GA_ReadOnly)
    future = dataset_f.ReadAsArray(0, 0, nx, ny)  # present data

    # ####################################################################
    # ####################################################################
    # 2 user-defined threshold
    t=5 # you can change this parameter, it means you assume the range of x ± t will be the same. x is your climate raster
    t=1/(t*2)
    present=np.round(present*t)/t
    future=np.round(future*t)/t

    # ####################################################################
    # ####################################################################
    # 3 calculate climate velocity
    distance = np.full(present.shape, np.nan)  # this is the climate velocity raster, the unit is the number of cells
    northDifference = np.full(present.shape, np.nan)  # the north vector value of climate velocity
    eastDifference = np.full(present.shape, np.nan)  # the east vector value of climate velocity

    uniqueP=np.unique(present)
    Findex=[]
    for i_uniqueP in uniqueP:
        Findex.append(np.where(future == i_uniqueP))

    for col in range(present.shape[1]):
        for row in range(present.shape[0]):
            try:
                temindex=np.where(uniqueP==present[row][col])[0][0]

                Findexrow=np.array(Findex[temindex][0])
                Findexcol=np.array(Findex[temindex][1])

                temdistance=np.square(Findexrow-row)+np.square(Findexcol-col)
                temmindistance=np.sqrt(np.min(temdistance))
                temminindex=np.argmin(temdistance)

                distance[row][col]=temmindistance
                northDifference[row][col]=(Findexrow[temminindex]-row)/temmindistance
                eastDifference[row][col]=(Findexcol[temminindex]-col)/temmindistance
            except:
                continue

    # ####################################################################
    # ####################################################################
    # 4 write climate velocity to tiff file
    climateV=distance*distance_unit/time_interval
    northV=northDifference*distance_unit/time_interval
    eastV=eastDifference*distance_unit/time_interval

    writeTiff(climateV, im_width, im_height, im_bands, im_geotrans, im_proj, path+'climateV.tif')
    writeTiff(northV, im_width, im_height, im_bands, im_geotrans, im_proj, path+'northV.tif')
    writeTiff(eastV, im_width, im_height, im_bands, im_geotrans, im_proj, path+'eastV.tif')


if __name__ == '__main__':

    filenameP='/Users/fxj/Documents/fresearch/F202105hengduanMatlab/data_files/chelsa/bio12cut/CHELSA_TraCE21k_bio12_-190_V1.09510224.031.tif' # present tiff file
    filenameF='/Users/fxj/Documents/fresearch/F202105hengduanMatlab/data_files/chelsa/bio12cut/CHELSA_TraCE21k_bio12_20_V1.09510224.031.tif' # future tiff file
    path = '/Users/fxj/Documents/fresearch/F202105hengduanMatlab/data_files/chelsa/bio12cut/'  # where you want to write your climate velocity files

    distance_unit=1 # km the distance unit of your tiff files
    time_interval=21 # ka the time difference from present to future

    climate_Velocity(filenameP, filenameF, distance_unit, time_interval,path)

