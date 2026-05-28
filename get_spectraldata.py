import os
import time
import csv
import json
import tifffile
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from scipy import interpolate, signal
from scipy.optimize import least_squares
import pandas as pd
import seaborn as sns
import subprocess
#from osgeo import gdal, osr, gdalconst, gdal_array

# read radiometric file (csv)
def read_bfile(file):
    fileb = os.path.splitext(file)[0] + "_B.csv"
    df = pd.read_csv(fileb)
    param = np.zeros((185, 5), dtype=float)
    for i in range(min(185, len(df))):
        for j in range(5):
            param[i, j] = float(df.iloc[i, j + 1])
    #‘CenterWavelengthNanometer’, ‘FullWidthAtHalfMaximumNanometer’,
    #‘SolarIrradianceWatt/Meter2/Micron’, ‘ReflectanceMulti’, ‘ReflectanceAdd’
    return param

# read meta data (txt)
def read_tfile(file):
    fileb = os.path.splitext(file)[0]  
    fileb = fileb + ".txt" 
    csv_file = open(fileb, "r")  
    record_list = []  
    record = csv_file.readline() 
    while record : 
        record_list.append(record.rstrip().split("=")) 
        record = csv_file.readline() 
    for record in record_list:  
        if(record[0]=="RadianceMultiVNIR                                                      "):
            radiancemultivnir = float(record[1])
        if(record[0]=="RadianceAddVNIR                                                        "):
            radianceaddvnir = float(record[1])
        if(record[0]=="RadianceMultiSWIR                                                      "):
            radiancemultiswir = float(record[1])
        if(record[0]=="RadianceAddSWIR                                                        "):
            radianceaddswir = float(record[1])
    return radiancemultivnir, radiancemultiswir, radianceaddvnir, radianceaddswir

def apply_radiometric(img, radmultivnir, radmultiswir, radaddvnir, radaddswir):
    im = np.ones([img.shape[0], img.shape[1]])  
    # no data area
    im[img[:,:,10] == 0] = 0  
    # change to float
    img = 1.0 * img  
    # apply radiometric vnir
    for j in range(58): 
        img[:,:,j] = img[:,:,j] * radmultivnir + radaddvnir  
    # apply radiometric swir
    for j in range(58, 185): 
        img[:,:,j] = img[:,:,j] * radmultiswir + radaddswir  
    img[im == 0] = 0 
    return img

def show_xy(src, x, y):
    width = src.RasterXSize 
    height = src.RasterYSize 
    gt = src.GetGeoTransform() 
    minx = gt[0]
    miny = gt[3] + width * gt[4] + height * gt[5]
    maxx = gt[0] + width * gt[1] + height * gt[2]
    maxy = gt[3]
    X = gt[0] + x * gt[1] + y * gt[2]
    Y = gt[3] + x * gt[4] + y * gt[5]
    return X, Y

#def show_latlon(src, x, y):
    old_cs= osr.SpatialReference() 
    old_cs.ImportFromWkt(src.GetProjectionRef()) 
    wgs84_wkt = """
        GEOGCS["WGS 84",
            DATUM["WGS_1984",
                SPHEROID["WGS 84",6378137,298.257223563,
                    AUTHORITY["EPSG","7030"]],
                AUTHORITY["EPSG","6326"]],
            PRIMEM["Greenwich",0,
                AUTHORITY["EPSG","8901"]],
            UNIT["degree",0.01745329251994328,
                AUTHORITY["EPSG","9122"]],
            AUTHORITY["EPSG","4326"]]"""
    new_cs = osr.SpatialReference() 
    new_cs .ImportFromWkt(wgs84_wkt) 
    transform = osr.CoordinateTransformation(old_cs,new_cs)
    X, Y = show_xy(src, x, y) 
    latlong = transform.TransformPoint(X, Y)
    return latlong

def get_rgb(img, b=8, g=18, r=28):
    ims = np.zeros([img.shape[0], img.shape[1], 3])  
    ims[:,:,0] = img[:,:,r]    #R
    ims[:,:,1] = img[:,:,g]    #G
    ims[:,:,2] = img[:,:,b]    #B
    max = np.max(ims)/3 
    ims /= max  
    ims = np.clip(ims, 0.0, 1.0) 
    return ims
    
def show_img(img):
    fig, ax = plt.subplots()  
    im = ax.imshow(img) 
    plt.show()
    
def get_radiance(img, param, y, x):
    wave = param[58:185,0]
    rad = img[y, x, 58:185]
    list_data = [wave, rad]
    list_data_T = np.array(list_data).T
    return list_data_T

file = r"E:\メタン\2025_HISUI_72_The Permian Basin-論文照合用\HSHL1G_N320W1032_20221030160051_20231127193053\HSHL1G_N320W1032_20221030160051_20231127193053.tif"
img = tifffile.imread(file) 
param = read_bfile(file)   
radmultivnir, radmultiswir, radaddvnir, radaddswir = read_tfile(file)  
img = apply_radiometric(img, radmultivnir, radmultiswir, radaddvnir, radaddswir)  
ims = get_rgb(img, b=8, g=18, r=28)
center = np.array([1066, 1463]) 
img_slice = img[center[0] - 100 : center[0] + 100, center[1] - 100 : center[1] + 100, :] 
ims_slice = ims[center[0] - 100 : center[0] + 100, center[1] - 100 : center[1] + 100 , :] 
show_img(ims)
show_img(ims_slice)

def save_roi_spectra_csv(img_slice, param, output_csv="all_roi_spectra200x200.csv"):
    h, w, bands = img_slice.shape

    wavelengths = param[:bands, 0]

    columns = ["y", "x"] + [f"wave_{wl:.2f}nm" for wl in wavelengths]

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    # (h, w, bands) -> (h*w, bands)
    spectra = img_slice.reshape(-1, bands)

    data = np.column_stack([
        yy.reshape(-1),
        xx.reshape(-1),
        spectra
    ])

    df = pd.DataFrame(data, columns=columns)

    df["y"] = df["y"].astype(int)
    df["x"] = df["x"].astype(int)

    df.to_csv(output_csv, index=False)

    return df

roi_df = save_roi_spectra_csv(
    img_slice,
    param,
    output_csv="all_roi_spectra200x200.csv"
)

roi_df.head()
