import numpy as np
import pickle
import sys
import copy
import scipy.sparse as sp
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata
from pathlib import Path
import pandas as pd
import geopandas as gpd
import math
from shapely.geometry import shape, Point
import matplotlib.pyplot as plt
import skimage as ski
import numpy as np
from scipy.interpolate import griddata
import math
from scipy.signal import fftconvolve as scipy_fft_conv
from astropy.convolution import convolve, convolve_fft
import rasterio
import rasterio.features
import rasterio.warp
from skimage import img_as_float
from skimage import transform
import warnings
import cv2
from sklearn.metrics import log_loss
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import datetime
from matplotlib.gridspec import GridSpec
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from osgeo import gdal, ogr
import plotly.express as px
import plotly.io as pio
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import math
from scipy.ndimage import generic_filter
import datetime as dt
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import plotly.express as px
import plotly.io as pio
from hdbscan import HDBSCAN, approximate_predict
import math
import rasterio
import rasterio.features
import rasterio.warp
import scipy.stats
import sys
import warnings
from osgeo import gdal, ogr
from pathlib import Path
from scipy import ndimage as ndi
from skimage import measure, segmentation, morphology
from skimage.segmentation import watershed
from skimage import transform
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import SpectralClustering
from scipy.spatial import distance as d
from sklearn.cluster import KMeans
from skimage import io, color, filters, measure, morphology
from scipy import ndimage as ndi
from skimage.morphology import disk
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte
import scipy.ndimage as ndimage
import logging
import cv2
import geopandas as gpd
import pandas as pd
import xarray as xr

def haversine(p1, p2, unit = 'kilometer'):
    import math
    # Coordinates in decimal degrees (e.g. 2.89078, 12.79797)
    lon1 = float(p1.x)
    lat1 = float(p1.y)
    lon2 = float(p2.x)
    lat2 = float(p2.y)

    R = 6371000  # radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    meters = R * c  # output distance in meters
    km = meters / 1000.0  # output distance in kilometers
    
    meters = round(meters)
    km = round(km, 3)

    if unit == 'kilometer':
        return km
    elif unit == 'meters':
        return meters
    else:
        return math.inf
    
def distanceWithAltitude(p1, p2, a1, a2):
    haverDista = haversine(p1, p2, 'meters')
    diffAltitude = a1 - a2
    res = math.sqrt((diffAltitude * diffAltitude) + (haverDista * haverDista))
    return res

def get_top1_closest_cluster(x, a, dept, ind, locations, col):
    mindistance = math.inf
    minIndex = 0
    if dept in ['Yvelines', 'Doubs']:
        locs = locations[locations['departement'] == dept]
    else:
        locs = locations[locations['departement'].isin(['Ain', 'Rhone'])]
    for _, row in locs.iterrows():
        if row[col] in ind:
            continue
        #distance = haversine(x, row['centroid'])
        distance = distanceWithAltitude(x, row['centroid'], a, row['altitude'])
        if distance < mindistance:
            minIndex = row[col]
            mindistance = distance
    return minIndex

def get_distance_cluster(x, locations, col, colGeo):
    res = np.full((len(locations), 2), math.inf)
    index = 0
    ignore = [x]
    for ind, row in locations.iterrows():
        if row[col] in ignore:
            continue
        #distance = haversine(x, row['centroid'])
        distance = haversine(x, row[colGeo])
        res[index, 0] = row[col]
        res[index, 1] = distance
        #print(x, row[colGeo], row[col], distance)
        ignore.append(row[col])
        index += 1
    return res

def quantile_prediction_error(ytrue, ypred, reduction='mean'):
    maxie = np.nanmax(ypred)
    minie = np.nanmin(ypred)
    print(maxie)
    X = []
    Y = []

    quantiles = [(0.0 ,0.2),
                 (0.2, 0.4),
                 (0.4, 0.6),
                 (0.6, 0.8),
                 (0.8, 1.0)]
    
    ytrue = (ytrue > 0).astype(int)
    propor = 0.6
    print(ypred)
    for (minB, maxB) in quantiles:
        pred_quantile = ypred[(ypred >= minB * maxie) & (ypred < maxB * maxie)]
        if pred_quantile.shape[0] != 0:
            pred_quantile = np.mean(pred_quantile)
            number_of_fire = np.mean(ytrue[(ypred >= minB * maxie) & (ypred < maxB * maxie)])
            fire = ytrue[(ypred >= minB * maxie) & (ypred < maxB * maxie) & (ytrue == 1)]
            nf = ytrue[(ypred >= minB * maxie) & (ypred < maxB * maxie) & (ytrue == 0)]
            if fire.shape[0] != 0:
                m = np.random.choice(np.arange(fire.shape[0]), int(fire.shape[0] * (number_of_fire + 0.01)))
                X += list(ypred[(ypred >= minB * maxie) & (ypred < maxB * maxie) & (ytrue == 1)][m])
                Y += list(ytrue[(ypred >= minB * maxie) & (ypred < maxB * maxie) & (ytrue == 1)][m])
            if nf.shape[0] != 0:
                m = np.random.choice(np.arange(nf.shape[0]), int(nf.shape[0] * (number_of_fire + 0.01)))
                X += list(ypred[(ypred >= minB * maxie) & (ypred < maxB * maxie) & (ytrue == 0)][m])
                Y += list(ytrue[(ypred >= minB * maxie) & (ypred < maxB * maxie) & (ytrue == 0)][m])
            print((minB, maxB), pred_quantile, number_of_fire)

    return np.asarray(X), np.asarray(Y)

def get_dist1_closest_cluster(x, a, dept, ind, locations):
    mindistance = math.inf
    if dept.isin(['Yvelines', 'Doubs']):
        locs = locations[locations['departement'] == dept]
    else:
        locs = locations[locations['departement'].isin(['Ain', 'Rhone'])]
    for _, row in locs.iterrows():
        if row['cluster'] in ind:
            continue
        #distance = haversine(x, row['centroid'])
        distance = distanceWithAltitude(x, row['centroid'], a, row['altitude'])
        if distance < mindistance:
            mindistance = distance
    return mindistance

def show_pcs(pca, size, components, dir_output):
    labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    fig = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(size)
    )
    fig.update_traces(diagonal_visible=False)
    #fig.show()
    pio.write_image(fig, dir_output / "pca.png") 


def np_function(image):
    """
    Create a image with each pixel be the distance bewteen the pixel coordinate and the center
    """
    center = np.array([(image.shape[0])/2, (image.shape[1])/2])

    distances = np.linalg.norm(np.indices(image.shape) - center[:,None,None], axis = 0, ord=math.inf)

    return distances

def myFunction(outputShape):
    centerY, centerX = np.indices(outputShape)
    centerY = centerY - outputShape[0] // 2
    centerX = centerX - outputShape[1] // 2

    distances = np.sqrt(centerX**2 + centerY**2) + 1
    return distances

def myFunctionDistanceDugrandCercle(outputShape, earth_radius=6371.0, resolution_lon=0.0002694945852352859, resolution_lat=0.0002694945852326214):
    half_rows = outputShape[0] // 2
    half_cols = outputShape[1] // 2

    # Créer une grille de coordonnées géographiques avec les résolutions souhaitées
    latitudes = np.linspace(-half_rows * resolution_lat, half_rows * resolution_lat, outputShape[0])
    longitudes = np.linspace(-half_cols * resolution_lon, half_cols * resolution_lon, outputShape[1])
    latitudes, longitudes = np.meshgrid(latitudes, longitudes, indexing='ij')

    # Coordonnées du point central
    center_lat = latitudes[outputShape[0] // 2, outputShape[1] // 2]
    center_lon = longitudes[outputShape[0] // 2, outputShape[1] // 2]
    #print(center_lat, center_lon)
    
    # Convertir les coordonnées géographiques en radians
    latitudes_rad = np.radians(latitudes)
    longitudes_rad = np.radians(longitudes)

    # Calculer la distance du grand cercle entre chaque point et le point central
    delta_lon = longitudes_rad - np.radians(center_lon)
    delta_lat = latitudes_rad - np.radians(center_lat)
    a = np.sin(delta_lat/2)**2 + np.cos(latitudes_rad) * np.cos(np.radians(center_lat)) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distances = earth_radius * c

    return distances

def myFunctionDistanceDugrandCercle3D(outputShape, earth_radius=6371.0, resolution_lon=0.0002694945852352859, resolution_lat=0.0002694945852326214, resolution_altitude=10):
    half_rows = outputShape[0] // 2
    half_cols = outputShape[1] // 2
    half_altitude = outputShape[2] // 2

    # Créer une grille de coordonnées géographiques avec les résolutions souhaitées
    latitudes = np.linspace(-half_rows * resolution_lat, half_rows * resolution_lat, outputShape[0])
    longitudes = np.linspace(-half_cols * resolution_lon, half_cols * resolution_lon, outputShape[1])
    altitudes = np.linspace(-half_altitude * resolution_altitude, half_altitude * resolution_altitude, outputShape[2])
    
    latitudes, longitudes, altitudes = np.meshgrid(latitudes, longitudes, altitudes, indexing='ij')

    # Coordonnées du point central
    center_lat = latitudes[outputShape[0] // 2, outputShape[1] // 2, outputShape[2] // 2]
    center_lon = longitudes[outputShape[0] // 2, outputShape[1] // 2, outputShape[2] // 2]
    center_altitude = altitudes[outputShape[0] // 2, outputShape[1] // 2, outputShape[2] // 2]
    
    # Convertir les coordonnées géographiques en radians
    latitudes_rad = np.radians(latitudes)
    longitudes_rad = np.radians(longitudes)

    # Calculer la distance du grand cercle entre chaque point et le point central
    delta_lon = (longitudes_rad - np.radians(center_lon))
    delta_lat = (latitudes_rad - np.radians(center_lat))
    #delta_altitude = abs(altitudes - center_altitude)

    delta_altitude_ = abs(altitudes)
    delta_altitude = np.copy(delta_altitude_)
    #for i in range(delta_altitude_.shape[0]):
    #    for j in range(delta_altitude_.shape[1]):
    #        delta_altitude[i, j, :delta_altitude_.shape[2] // 2 + 1] = np.flip(delta_altitude_[i, j, :delta_altitude_.shape[2] // 2 + 1])
    #        delta_altitude[i, j, delta_altitude_.shape[2] // 2:] = np.flip(delta_altitude_[i, j, delta_altitude_.shape[2] // 2:])

    a = np.sin(delta_lat/2)**2 + np.cos(latitudes_rad) * np.cos(np.radians(center_lat)) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distances = earth_radius * c * np.cos(np.radians(center_lat)) + delta_altitude
    
    return distances

def read_tif(name):
    """
    Open a satellite images and return bands, latitude and longitude of each pixel.
    """
    with rasterio.open(name) as src:
        dt = src.read()
        height = dt.shape[1]
        width = dt.shape[2]
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        lons = np.array(xs)
        lats = np.array(ys)
        src.close()
    return dt, lons, lats

def find_pixel(lats, lons, lat, lon):

    lonIndex = (np.abs(lons - lon)).argmin()
    latIndex = (np.abs(lats - lat)).argmin()

    lonValues = lons.reshape(-1,1)[lonIndex]
    latValues = lats.reshape(-1,1)[latIndex]
    #print(lonValues, latValues)
    return np.where((lons == lonValues) & (lats == latValues))

from sklearn.preprocessing import MinMaxScaler, normalize

def influence_index(raster, mask, dimS, mode, dim=(90,150)):
    dimX, dimY = dimS
    if mode == 'laplace':
        #kernel = myFunctionDistanceDugrandCercle(dim, resolution_lon=dimY, resolution_lat=dimX) + 1
        kernel = 1 / kernel
    else:
        kernel = np.full(dim, 1/(dim[0]*dim[1]), dtype=float)

    res = convolve_fft(array=raster, kernel=kernel, normalize_kernel=False, mask=mask)
    #res = scipy_fft_conv(raster, kernel)
    return res

def laplacien3D(outputShape):
    centerY, centerX, centerZ = np.indices(outputShape)
    centerY = centerY - outputShape[0] // 2
    centerX = centerX - outputShape[1] // 2
    centerZ = centerZ - outputShape[2] // 2

    distances = np.sqrt(centerX**2 + centerY**2 + centerZ**2) + 1.0
    distances = 1 / distances
    return distances

def find_n_component(thresh, pca):
    nb_component = 0
    sumi = 0.0
    for i in range(pca.explained_variance_ratio_.shape[0]):
        sumi += pca.explained_variance_ratio_[i]
        if sumi >= thresh:
            nb_component = i + 1
            break
    return nb_component

def influence_index3D(raster, mask, dimS, mode, dim=(90, 150, 3), semi=False, semi2=False):
    dimX, dimY, dimZ = dimS
    if dimX == 0 or dimY == 0 or dimZ == 0:
        return res
    if dim[-1] == 1:
        dimZ = 1
    else:
        dimZ = np.linspace(dim[-1]/2, 0, num=(dim[-1] // 2) + 1)[0] - np.linspace(dim[-1]//2, 0, num=(dim[-1] // 2) + 1)[1]
    if mode == "laplace":
        kernel = myFunctionDistanceDugrandCercle3D(dim, resolution_lon=dimX, resolution_lat=dimY, resolution_altitude=dimZ) + dimZ
        kernel = dimZ / kernel
    elif mode == 'inverse_laplace':
        kernel = myFunctionDistanceDugrandCercle3D(dim, resolution_lon=dimX, resolution_lat=dimY, resolution_altitude=dimZ) + dimZ
        kernel = dimZ / kernel
        kernel = (np.max(kernel) - kernel) + np.min(kernel)
    else:
        kernel = np.full(dim, 1/(dim[0]*dim[1]*dim[2]), dtype=float)

    if semi:
        if dim[2] != 1:
            kernel[:,:,:(dim[2]//2)] = 0.0
    if semi2:
        if dim[2] != 1:
            kernel[:,:,(dim[2]//2):] = 0.0
    res = convolve_fft(raster, kernel, normalize_kernel=False, mask=mask)
    return res

def rasterization(ori : gpd.GeoDataFrame,
                  lats,
                  longs,
                  column : str,
                  dir_output: Path,
                  outputname='ori'):
    
    check_and_create_path(dir_output / 'mask/geo/' )
    check_and_create_path(dir_output / 'mask/tif/' )
    ori.to_file(dir_output.as_posix() + '/mask/geo/'+outputname+'.geojson', driver="GeoJSON")
    
    # paths du geojson d'entree et du raster tif de sortie
    input_geojson = dir_output.as_posix() + '/mask/geo/'+outputname+'.geojson'
    output_raster = dir_output.as_posix() + '/mask/tif/'+outputname+'.tif'

    # Si on veut rasteriser en fonction de la valeur d'un attribut du vecteur, mettre son nom ici
    attribute_name = column

    # Taille des pixels
    if isinstance(lats, float):
        pixel_size_y = lats
        pixel_size_x = longs
    else:
        pixel_size_x = abs(longs[0][0] - longs[0][1])
        pixel_size_y = abs(lats[0][0] - lats[1][0])
        print(f'px {pixel_size_x}, py {pixel_size_y}')
    
    #pixel_size_x = res[dim][0]
    #pixel_size_y = res[dim][1]

    source_ds = ogr.Open(input_geojson)
 
    source_layer = source_ds.GetLayer()

    # On obtient l'étendue du raster
    x_min, x_max, y_min, y_max = source_layer.GetExtent()

    # On calcule le nombre de pixels
    width = int((x_max - x_min) / pixel_size_x)
    height = int((y_max - y_min) / pixel_size_y)
    
    # On crée un nouveau raster dataset et on passe de "coordonnées image" (pixels) à des coordonnées goréférencées
    driver = gdal.GetDriverByName('GTiff')
    output_ds = driver.Create(output_raster, width, height, 1, gdal.GDT_Float32)
    output_ds.GetRasterBand(1).Fill(np.nan)
    output_ds.SetGeoTransform([x_min, pixel_size_x, 0, y_max, 0, -pixel_size_y])
    output_ds.SetProjection(source_layer.GetSpatialRef().ExportToWkt())

    if attribute_name != '' :
        # On  rasterise en fonction de l'attribut donné
        gdal.RasterizeLayer(output_ds, [1], source_layer, options=["ATTRIBUTE=" + attribute_name])
    else :
        # On  rasterise. Le raster prend la valeur 1 là où il y a un vecteur
        gdal.RasterizeLayer(output_ds, [1], source_layer)

    output_ds = None
    source_ds = None
    return read_tif(output_raster)

def find_nearest(array, value, nanmaks=None):
    if nanmaks is None:
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
    else:
        idx = (np.abs(array[nanmaks[:,0], nanmaks[:,1]] - value)).argmin()
    return array[idx]

def myRasterization(geo, tif, maskNan, sh, column, maskCol, val=np.nan):
    assert maskCol != ''
    res = np.full(sh, val, dtype=float)
    
    if maskNan is not None:
        res[maskNan[:,0], maskNan[:,1]] = np.nan

    for index, row in geo.iterrows():
        #inds = indss[row['hex_id']]
        """ mask = np.zeros((sh[0], sh[1]))
        cv2.fillConvexPoly(mask, inds, 1)
        mask = mask > 0"""
        mask = tif == row[maskCol]
        if column != '':
            res[mask] = row[column]
            print(row[column])
        else:
            res[mask] = 1
    return res

def create_spatio_temporal_sinister_image(firepoints: pd.DataFrame,
                                          regions: gpd.GeoDataFrame,
                                          dates: list,
                                          mask: np.array,
                                          sinisterType: str,
                                          sinister_encoding: str,
                                          n_pixel_y: float, 
                                          n_pixel_x: float,
                                          dir_output: Path,
                                          dept: str):
    # Create a deep copy of firepoints to avoid modifying the original DataFrame
    firepoints = firepoints.copy(deep=True)

    # Get coordinates of valid (non-NaN) positions in the spatial mask
    nonNanMask = np.argwhere(~np.isnan(mask))

    # Unique dates where incidents are reported
    fdate = firepoints['date'].unique()

    # Number of total dates for temporal axis
    lenDates = len(dates)

    # Initialize a 3D raster to store spatio-temporal values: (rows, cols, time)
    spatioTemporalRaster = np.full((mask.shape[0], mask.shape[1], lenDates), np.nan)

    # Iterate over each date in the full list of time steps
    for i, date in enumerate(dates):

        # Optional progress print every 200 iterations
        if i % 200 == 0:
            print(date)

        # Check if the date has incident data; if not, fill with 0
        if date in fdate:
            fdataset = firepoints[firepoints['date'] == date]
        else:
            spatioTemporalRaster[nonNanMask[:, 0], nonNanMask[:, 1], i] = 0
            continue

        # Create a deep copy of the spatial region data
        hexaFire = regions.copy(deep=True)

        # ---------- ENCODING: INCIDENT OCCURRENCE ----------
        if sinister_encoding == 'occurence':
            # Initialize columns to track presence and count of incidents
            hexaFire['is' + sinisterType] = 0
            hexaFire['nb' + sinisterType] = 0

            # For each incident, update matching spatial region
            for _, row in fdataset.iterrows():
                matched_idx = hexaFire[hexaFire['scale0'] == row['scale0']].index
                hexaFire.loc[matched_idx, 'is' + sinisterType] = 1
                hexaFire.loc[matched_idx, 'nb' + sinisterType] += 1

            # Rasterize the variable to 2D spatial image
            rasterVar, lon, lat = rasterization(
                hexaFire, n_pixel_y, n_pixel_x, 'nb' + sinisterType, dir_output, dept + '_bin0'
            )

            # Handle edge case: raster is empty (only zeros or NaNs)
            if np.all(rasterVar[~np.isnan(rasterVar)] == 0):
                # Try again with higher resolution to find small features
                rasterVar2, _, _ = rasterization(
                    hexaFire, 0.0002694945852326214, 0.0002694945852352859, 'nb' + sinisterType, dir_output, dept + '_bin0'
                )
                rasterVar2 = rasterVar2[0]
                rasterVar2[np.isnan(rasterVar2)] = 0

                # Resize to match original raster dimensions
                val_res = resize_no_dim(rasterVar2, rasterVar.shape[1], rasterVar.shape[2])

                # Pad array to help detect local maxima
                val_res = np.pad(val_res, [1, 1])

                # Detect peaks (potential hotspots)
                coordinates = peak_local_max(val_res, min_distance=1)

                # Prepare to update raster with enhanced peak values
                val_res_2 = np.copy(val_res)
                val_res = np.zeros_like(val_res).astype(int)

                # Copy peak values to final raster
                for coord in coordinates:
                    val_res[coord[0], coord[1]] = math.ceil(val_res_2[coord[0], coord[1]])

                # Insert peak-enhanced raster values
                mask_nan = np.isnan(rasterVar)
                rasterVar[0] = val_res[1:rasterVar[0].shape[0] + 1, 1:rasterVar[0].shape[1] + 1]

                # Restore NaNs where appropriate
                mask_nan = mask_nan & (rasterVar[0] == 0)
                rasterVar[mask_nan] = np.nan

                # Log issue if raster is still entirely 0
                if np.all(rasterVar[~np.isnan(rasterVar)] == 0):
                    print(f'Unique values of scale0 {fdataset.scale0.unique()}')
                    if "Département" in np.unique(fdataset.columns):
                        log_message = f'{date} Unique values of scale0 {len(fdataset)} {fdataset["Département"].unique()} {fdataset.scale0.unique()}\n'
                    elif 'departement' in np.unique(fdataset.columns):
                        log_message = f'{date} Unique values of scale0 {len(fdataset)} {fdataset["departement"].unique()} {fdataset.scale0.unique()}\n'
                    else:
                        log_message = f'{date} Unique values of scale0 {len(fdataset)} {fdataset.scale0.unique()}\n'
                    print(log_message.strip())

        # ---------- ENCODING: BURNED AREA ----------
        elif sinister_encoding == 'burned_area':
            # Initialize area field
            hexaFire['Surface parcourue (m2)'] = 0

            # Sum surface area by region
            for _, row in fdataset.iterrows():
                matched_idx = hexaFire[hexaFire['scale0'] == row['scale0']].index
                hexaFire.loc[matched_idx, 'Surface parcourue (m2)'] += row['Surface parcourue (m2)']

            # Convert to hectares
            hexaFire['Surface parcourue (h)'] = hexaFire['Surface parcourue (m2)'] * 0.0001

            # Rasterize burned area
            rasterVar, _, _ = rasterization(hexaFire, n_pixel_y, n_pixel_x, 'Surface parcourue (m2)', dir_output, dept + '_bin0')

        # ---------- ENCODING: TIME TO INTERVENTION ----------
        elif sinister_encoding == 'time_intervention':
            # Initialize time difference field
            hexaFire['time_intervention'] = 0.0

            # Sum intervention time by region
            for _, row in fdataset.iterrows():
                matched_idx = hexaFire[hexaFire['scale0'] == row['scale0']].index
                hexaFire.loc[matched_idx, 'time_intervention'] += row['hours_difference']

            # Rasterize intervention time
            rasterVar, _, _ = rasterization(hexaFire, n_pixel_y, n_pixel_x, 'time_intervention', dir_output, dept + '_bin0')

        # Store raster values at non-NaN mask locations for this date
        spatioTemporalRaster[nonNanMask[:, 0], nonNanMask[:, 1], i] = rasterVar[0][nonNanMask[:, 0], nonNanMask[:, 1]]

    # Return the full 3D spatio-temporal raster
    return spatioTemporalRaster

def myRasterization3D(geo, indss, maskNan, dicoSat, column):

    departements = geo.departement.unique()

    res = {}

    for dept in departements:
        sh = dicoSat[dept]['sat'].shape
     
        res[dept] = np.full(sh, 0, dtype=int)
        
        if maskNan is not None:
            res[dicoSat[dept]['nanmask'][:,0], dicoSat[dept]['nanmask'][:,1], dicoSat[dept]['nanmask'][:,2]] = np.nan

        geoDep = geo[geo['departement'] == dept]
        for index, row in geoDep.iterrows():
            inds = indss[row['hex_id']]
            mask = np.zeros((sh[0], sh[1], sh[2]))
            minZ = np.min(inds[:,2])
            maxZ = np.max(inds[:,2])
            for i in range(minZ, maxZ+1):
                mask2 = np.zeros((sh[0], sh[1]))
                cv2.fillConvexPoly(mask2, inds[:, 0:2], 1)
                mask[:,:,i] = mask2
            mask[dicoSat[dept]['nanmask'][:,0], dicoSat[dept]['nanmask'][:,1], dicoSat[dept]['nanmask'][:,2]] = 0
            mask = mask > 0
            res[dept][mask] += row[column]

    return res

def get_indices(geo, dicoSat):

    indss = {}

    i = 0
    leni = len(geo)
    
    for index, row in geo.iterrows():
        print(i, '/', leni)
        ulatsMask = ~np.isnan(dicoSat[row['departement']]['ulats'])
        ulonsMaks = ~np.isnan(dicoSat[row['departement']]['ulongs'])
        
        nodes = []
        for j in list(row['geometry'].exterior.coords): 
            nodes.append(gpd.GeoDataFrame({'geometry':[Point(j)], 'latitude':[float(Point(j).y)], 'longitude':[float(Point(j).x)]}))

        nodes = pd.concat(nodes)
        inds = []

        for _, row2 in nodes.iterrows():

            closestLong = find_nearest(dicoSat[row['departement']]['ulongs'][ulonsMaks], row2['longitude'])
            closestLat = find_nearest(dicoSat[row['departement']]['ulats'][ulatsMask],  row2['latitude'])

            ind = (np.where((dicoSat[row['departement']]['lats'] == closestLat) & (dicoSat[row['departement']]['longs'] == closestLong)))
            if np.array(ind).shape[1] == 1:
                #print('Found')
                inds.append([ind[1], ind[0]])
            elif np.array(ind).shape[1] == 0:
                print('Not found')
                pass
            else:
                #print('Found')
                for indx in ind:
                    inds.append([indx[1], indx[0]])
        
        i += 1
        inds = np.array(inds)
        indss[row['hex_id']] = inds
    return indss

def get_indices3D(geo, dicoSat):

    indss = {}

    i = 0
    leni = len(geo)
    
    for index, row in geo.iterrows():
        if i % 100 == 0:
            print(i, '/', leni)
        ulatsMask = ~np.isnan(dicoSat[row['departement']]['ulats'])
        ulonsMask = ~np.isnan(dicoSat[row['departement']]['ulongs'])
        ualtMask = ~np.isnan(dicoSat[row['departement']]['ualt'])
        
        nodes = []
        for j in list(row['geometry'].exterior.coords): 
            nodes.append(gpd.GeoDataFrame({'geometry':[Point(j)], 'latitude':[float(Point(j).y)], 'longitude':[float(Point(j).x)],
                                          'altitude' : [float(Point(j).z)]}))

        nodes = pd.concat(nodes)
        inds = []

        for _, row2 in nodes.iterrows():

            closestLong = find_nearest(dicoSat[row['departement']]['ulongs'][ulonsMask], row2['longitude'])
            closestLat = find_nearest(dicoSat[row['departement']]['ulats'][ulatsMask],  row2['latitude'])
            closestAlt = find_nearest(dicoSat[row['departement']]['ualt'][ualtMask],  row2['altitude'])
            
            ind = (np.where((dicoSat[row['departement']]['lats'] == closestLat) & (dicoSat[row['departement']]['longs'] == closestLong) & 
                            (dicoSat[row['departement']]['alt'] == closestAlt)))
            
            if np.array(ind).shape[1] == 1:
                #print('Found')
                inds.append([ind[1], ind[0], ind[2]])
            elif np.array(ind).shape[1] == 0:
                print('Not found')
                pass
            else:
                #print('Found')
                for indx in ind:
                    inds.append([indx[1], indx[0], indx[2]])
        
        i += 1
        inds = np.array(inds)
        indss[row['hex_id']] = inds
    return indss

def meter2h3(geo, sh, image, indss):

    res = []
    for index, row in geo.iterrows():
        inds = indss[row['hex_id']]
        mask = np.zeros((sh[0], sh[1]))
        cv2.fillConvexPoly(mask, inds, 1)
        mask = mask > 0
        res.append(np.nanmean(image[mask]))
    
    return res

def meter2h33D(geo, dicoSat, image, indss):

    departements = geo.departement.unique()
    geo['severity'] = 0
    
    for departement in departements:

        geodept = geo[geo['departement'] == departement]

        for index, row in geodept.iterrows():
            
            inds = indss[row['hex_id']]
            sh = dicoSat[departement]['sat'].shape
            mask = np.zeros((sh[0], sh[1], sh[2]))
            minZ = np.min(inds[:,2])
            maxZ = np.max(inds[:,2])
            for i in range(minZ, maxZ+1):
                mask2 = np.zeros((sh[0], sh[1]))
                cv2.fillConvexPoly(mask2, inds[:, 0:2], 1)
                mask[:,:,i] = mask2
            mask[dicoSat[departement]['nanmask'][:,0], dicoSat[departement]['nanmask'][:,1], dicoSat[departement]['nanmask'][:,2]] = 0
            mask = mask > 0
            geo.loc[index, 'severity'] = np.nanmean(image[departement][mask])

def resize(input_image, height, width, dim):
    """
    Resize the input_image into heigh, with, dim
    """
    img = img_as_float(input_image)
    img = transform.resize(img, (dim, height, width))
    return np.asarray(img)

def resize_no_dim(input_image, height, width):
    """
    Resize the input_image into heigh, with, dim
    """
    img = img_as_float(input_image)
    img = transform.resize(img, (height, width))
    return np.asarray(img)

def getWeekImage(departements):
    sh = np.array([0,0])

    ainShape = np.array([0,0])
    dousShape = np.array([0,0])
    rhoneShape = np.array([0,0])
    yvelinesShape = np.array([0,0])

    isAin = False
    isDoubs = False
    isYvelines = False
    isRhone = False

    if 'Ain' in departements:
        ainShape = ainSat[0].shape
        sh += ainSat[0].shape
        isAin = True

    if 'Doubs' in departements:
        dousShape = sh.copy()
        sh += doubsSat[0].shape
        isDoubs = True
    else:
        dousShape = sh.copy()

    if 'Rhone' in departements:
        rhoneShape = sh.copy()
        sh += rhoneSat[0].shape
        isRhone = True
    else:
        rhoneShape = sh.copy()

    if 'Yvelines' in departements:
        yvelinesShape = sh.copy()
        sh += yvelinesSat[0].shape
        isYvelines = True
    else:
        yvelinesShape = sh.copy()

    lons = np.full((sh[0], sh[1]), np.nan)
    lats = np.full((sh[0], sh[1]), np.nan)
    val = np.full((sh[0], sh[1]), np.nan)

    if isAin:
        lons[:ainShape[0], :ainShape[1]] = ainLons
        lats[:ainShape[0], :ainShape[1]] = ainLats
        val[:ainShape[0], :ainShape[1]] = ainSat[0]

    if isDoubs:
        lons[dousShape[0]: rhoneShape[0]:, dousShape[1]: rhoneShape[1]] = doubsLons
        lats[dousShape[0]: rhoneShape[0]:, dousShape[1]: rhoneShape[1]] = doubsLats
        val[dousShape[0]: rhoneShape[0]:, dousShape[1]: rhoneShape[1]] = doubsSat[0]

    if isRhone:
        lons[rhoneShape[0]: yvelinesShape[0], rhoneShape[1]:yvelinesShape[1]] = rhoneLons
        lats[rhoneShape[0]: yvelinesShape[0], rhoneShape[1]:yvelinesShape[1]] = rhoneLats
        val[rhoneShape[0]: yvelinesShape[0], rhoneShape[1]:yvelinesShape[1]] = rhoneSat[0]

    if isYvelines:
        lons[yvelinesShape[0]:, yvelinesShape[1]:] = yvelinesLons
        lats[yvelinesShape[0]:, yvelinesShape[1]:] = yvelinesLats
        val[yvelinesShape[0]:, yvelinesShape[1]:] = yvelinesSat[0]

    return lats, lons, val

def createMask(mask):
    index = np.argwhere(mask > 0)
    res = np.zeros(mask.shape, dtype=int)
    indexXmin = np.min(index[:,0])
    indexXmax = np.max(index[:,0])
    indexYmin = np.min(index[:,1])
    indexYmax = np.max(index[:,1])
    res[indexXmin : indexXmax, indexYmin: indexYmax] = 1
    return indexXmin, indexXmax, indexYmin, indexYmax

def concatImage(img1, img2, lats1, lons1, lats2, lons2):

    maskLat = np.intersect1d(lats1, lats2)
    maskLon = np.intersect1d(lons1, lons2)
    print(maskLon.shape, maskLat.shape)
    if maskLat.shape[0] == 0 and maskLon.shape[0] == 0:
        return img1, img2

    img1Copy = img1.copy()
    img2Copy = img2.copy()

    noneNanMaskimg1 = np.argwhere(~np.isnan(img1))
    noneNanMaskimg2 = np.argwhere(~np.isnan(img2))

    indexXminimg1, indexXmaximg1, indexYminimg1, indexYmaximg1 = createMask((np.isin(lats1, maskLat) & np.isin(lons1, maskLon)) > 0)
    indexXminimg2, indexXmaximg2, indexYminimg2, indexYmaximg2 = createMask((np.isin(lats2, maskLat) & np.isin(lons2, maskLon)) > 0)

    img1Copy[indexXminimg1 : indexXmaximg1, indexYminimg1: indexYmaximg1] = img2[indexXminimg2 : indexXmaximg2, indexYminimg2: indexYmaximg2]
    img2Copy[indexXminimg2 : indexXmaximg2, indexYminimg2: indexYmaximg2] = img1[indexXminimg1: indexXmaximg1, indexYminimg1: indexYmaximg1 ]

    img1Copy[noneNanMaskimg1[:,0], noneNanMaskimg1[:,1]] = img1[noneNanMaskimg1[:,0], noneNanMaskimg1[:,1]]
    img2Copy[noneNanMaskimg2[:,0], noneNanMaskimg2[:,1]] = img2[noneNanMaskimg2[:,0], noneNanMaskimg2[:,1]]

    return img1Copy, img2Copy

def find_dates_between(start, end):
    start_date = datetime.datetime.strptime(start, '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(end, '%Y-%m-%d').date()

    delta = datetime.timedelta(days=1)
    date = start_date
    res = []
    while date < end_date:
            res.append(date.strftime("%Y-%m-%d"))
            date += delta
    return res

def expected_calibration_error(true_labels, samples, bins=5):

    bin_count, bin_edges = np.histogram(samples, bins = bins)
    n_bins = len(bin_count)
    # uniform binning approach with M number of bins
    bin_boundaries = bin_edges
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = samples

    # get a boolean list of correct/false predictions
    accuracies = true_labels

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return n_bins, ece[0] * 100


def calibration_plot(values, proba, bins, dir_output, prefix=''):
    fig = plt.figure(figsize=(10,10))
    gs = GridSpec(4, 2)
    colors = plt.cm.get_cmap("Dark2")

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    ax_calibration_curve.plot([0, 1], [0, 1], linestyle = '--', label = 'Perfect calibration')
    
    n_bins, ece = expected_calibration_error(values, proba, bins=bins)
    y_means, proba_means = calibration_curve(values, proba, n_bins=n_bins, strategy='quantile')
    ax_calibration_curve.plot(proba_means, y_means, label=', nb bins: ' + str(n_bins) + ", ECE: " + str(round(ece, 3)))

    ax_calibration_curve.set(xlabel="Mean predicted probability of positive class", ylabel="Fraction of positive class")
    ax_calibration_curve.grid()
    ax_calibration_curve.legend()
    name = prefix + 'calibration.png'
    plt.savefig(dir_output / name)

def check_and_create_path(path: Path):
    """
    Creer un dossier s'il n'existe pas
    """
    path_way = path.parent if path.is_file() else path

    path_way.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.touch()


def contains(x, geometries, col):
    for index, row in geometries.iterrows():
        if row['geometry'].contains(x):
            return row[col]
        
def create_dicoSat3D(self):
    print('Create dico sat 3D')
    res = {}

    for departement in self.departements:
        code = int(departement.split('-')[1])
        z_max = np.max(self.dicoSat[code]['alt'])
        z_min = np.min(self.dicoSat[code]['alt'])

        pixel_size_z = 10

        length = int((z_max - z_min) / pixel_size_z) + 1

        rasterVal = np.full((self.dicoSat[code]['sat'].shape[0], self.dicoSat[code]['sat'].shape[1], length), np.nan)
        rasterlats = np.full((self.dicoSat[code]['sat'].shape[0], self.dicoSat[code]['sat'].shape[1], length), np.nan)
        rasterlons = np.full((self.dicoSat[code]['sat'].shape[0], self.dicoSat[code]['sat'].shape[1], length), np.nan)
        rasterEle = np.full((self.dicoSat[code]['sat'].shape[0], self.dicoSat[code]['sat'].shape[1], length), np.nan)

        for i in range(length):
            valI = i * 10
            values = np.full((self.dicoSat[code]['sat'].shape[0], self.dicoSat[code]['sat'].shape[1]), np.nan)
            values[self.dicoSat[code]['alt'] == valI] = self.dicoSat[code]['sat'][self.dicoSat[code]['alt'] == valI]
            rasterVal[:,:, i] = values

            rasterlats[:,:,i] = self.dicoSat[code]['lats']
            rasterlons[:,:,i] = self.dicoSat[code]['longs']
            rasterEle[:,:,i] = valI

        res[code] = {'sat' : rasterVal, "lats": rasterlats, 'longs': rasterlons, 'alt': rasterEle, 'ulats' :np.unique(self.dicoSat[code]['lats']),
            'ulongs': np.unique(self.dicoSat[code]['longs']), 'ualt': np.unique(self.dicoSat[code]['alt']),
                    'nanmask' : np.argwhere(np.isnan(rasterVal)), 'noneNanmask' : np.argwhere(~np.isnan(rasterVal))}

    self.dicoDat3D = res

def interpolate_gridd(var, grid, newx, newy):
    x = grid['longitude'].values
    y = grid["latitude"].values
    points = np.zeros((y.shape[0], 2))
    points[:,0] = x
    points[:,1] = y
    return griddata(points, grid[var].values, (newx, newy), method='linear')

def create_geocube(df, variables, reslons, reslats):
    """
    Create a image representing variables with the corresponding resolution from df
    """
    geo_grid = make_geocube(
        vector_data=df,
        measurements=variables,
        resolution=(reslons, reslats),
        rasterize_function=rasterize_points_griddata,
        fill = 0
    )
    return geo_grid

def save_object(obj, filename: str, path : Path):
    check_and_create_path(path)
    with open(path / filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def read_object(filename: str, path : Path):
    return pickle.load(open(path / filename, 'rb'))

def find_dates_between(start, end):
    start_date = dt.datetime.strptime(start, '%Y-%m-%d').date()
    end_date = dt.datetime.strptime(end, '%Y-%m-%d').date()

    delta = dt.timedelta(days=1)
    date = start_date
    res = []
    while date < end_date:
            res.append(date.strftime("%Y-%m-%d"))
            date += delta
    return res

allDates = find_dates_between('2017-06-12', '2024-06-29')

resolutions = {'2x2' : {'x' : 0.02875215641173088,'y' :  0.020721094073767096}}

def save_object(obj, filename: str, path : Path):
    check_and_create_path(path)
    with open(path / filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def read_object(filename: str, path : Path):
    if not (path / filename).is_file():
        return None
    return pickle.load(open(path / filename, 'rb'))

def create_larger_scale_bin(input, bin, influence, raster):
    binImageScale = np.full(bin.shape, np.nan)
    influenceImageScale = np.full(influence.shape, np.nan)
    timeScale = np.full(influence.shape, np.nan)

    clusterID = np.unique(input)
    for di in range(bin.shape[-1]):
        for id in clusterID:
            mask = (input == id)
            if np.any(influence[mask, di] > 0):
                binImageScale[mask, di] = np.nansum(bin[mask, di])
                influenceImageScale[mask, di] = np.nansum(influence[mask, di])
            else:
                binImageScale[mask, di] = 0
                influenceImageScale[mask, di] = 0
                timeScale[mask, di] = 0

    return binImageScale, influenceImageScale

def order_class(predictor, pred, min_values=0):
    res = np.zeros(pred[~np.isnan(pred)].shape[0], dtype=int)
    cc = predictor.cluster_centers_.reshape(-1)
    classes = np.arange(cc.shape[0])
    ind = np.lexsort([cc])
    cc = cc[ind]
    classes = classes[ind]
    for c in range(cc.shape[0]):
        mask = np.argwhere(pred == classes[c])
        res[mask] = c
    return res + min_values

def check_and_create_path(path: Path):
    """
    Creer un dossier s'il n'existe pas
    """
    path_way = path.parent if path.is_file() else path

    path_way.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.touch()

def merge_adjacent_clusters(image, nb_attempt=3, mode='size', min_cluster_size=0, max_cluster_size=math.inf, exclude_label=None, background=-1):
    """
    Fusionne les clusters adjacents dans une image en fonction de critères définis.
    
    Paramètres :
    - image : Image labellisée contenant des clusters.
    - mode : Critère de fusion ('size', 'time_series_similarity', 'time_series_similarity_fast').
    - min_cluster_size : Taille minimale d'un cluster avant fusion.
    - max_cluster_size : Taille maximale autorisée après fusion.
    - oridata : Données supplémentaires utilisées pour la fusion basée sur des séries temporelles (facultatif).
    - exclude_label : Label à exclure de la fusion.
    - background : Label représentant le fond (par défaut -1).
    """

    # Copie de l'image d'entrée pour éviter de la modifier directement
    labeled_image = np.copy(image)

    # Obtenir les propriétés des régions labellisées
    regions = measure.regionprops(labeled_image)
    # Trier les régions par taille croissante
    regions = sorted(regions, key=lambda r: r.area)

    # Masque pour stocker les labels mis à jour après fusion
    res = np.copy(labeled_image)

    # Liste des labels qui ont été modifiés
    changed_labels = []

    # Nombre d'essais pour dilater un cluster avant abandon
    nb_attempt = 3

    # Longueur initiale des régions
    len_regions = len(regions)
    i = 0

    # Boucle pour traiter chaque région
    while i < len_regions:
        region = regions[i]

        # Vérifier si le cluster est à exclure ou est un fond
        if region.label == exclude_label or region.label == background:
            # On conserve ces clusters tels quels
            res[labeled_image == region.label] = region.label
            i += 1
            continue

        label = region.label

        # Si le label a déjà été modifié, passer au suivant
        if label in changed_labels:
            i += 1
            continue

        # Vérifier la taille du cluster actuel
        ones = np.argwhere(res == label).shape[0]
        if ones < min_cluster_size:
            # Si la taille est inférieure au minimum, essayer de fusionner avec un voisin
            nb_test = 0
            find_neighbor = False
            dilated_image = np.copy(res)
            while nb_test < nb_attempt and not find_neighbor:

                # Trouver les voisins du cluster actuel
                mask_label = dilated_image == label
                mask_label_ori = res == label
                neighbors = segmentation.find_boundaries(mask_label, connectivity=1, mode='outer', background=background)
                neighbor_labels = np.unique(dilated_image[neighbors])
                # Exclure les labels indésirables
                neighbor_labels = neighbor_labels[(neighbor_labels != exclude_label) & (neighbor_labels != background) & (neighbor_labels != label)]
                dilate = True
                changed_labels.append(label)

                if len(neighbor_labels) > 0:
                    # Trier les voisins par taille
                    neighbors_size = np.sort([[neighbor_label, np.sum(res == neighbor_label)] for neighbor_label in neighbor_labels])
                    best_neighbor = None

                    if mode == 'size':
                        # Mode basé sur la taille des clusters
                        max_neighbor_size = -math.inf
                        for nei, neighbor in enumerate(neighbors_size):
                            if neighbor[0] == label:
                                continue
                            neighbor_size = neighbor[1] + np.sum(res == label)

                            # Vérifier si le voisin satisfait min_cluster_size
                            if neighbor_size > min_cluster_size:
                                # Vérifier si la taille reste sous max_cluster_size
                                if neighbor_size < max_cluster_size:
                                    dilate = False
                                    res[mask_label_ori] = neighbor[0]
                                    dilated_image[mask_label] = neighbor[0]
                                    print(f'Use neighbord label {label} -> {neighbor[0]}')
                                    label = neighbor[0]
                                    find_neighbor = True
                                    break
                                
                                best_neighbor = neighbor[0]
                                max_neighbor_size = neighbor_size
                                break

                            # Enregistrer le plus grand voisin si min_cluster_size n'est pas atteint
                            if neighbor_size > max_neighbor_size:
                                best_neighbor = neighbor[0]
                                max_neighbor_size = neighbor_size

                        # Si aucun voisin ne satisfait les critères, utiliser le plus grand
                        if not find_neighbor and best_neighbor is not None:
                            if max_neighbor_size < max_cluster_size:
                                res[mask_label] = best_neighbor
                                dilated_image[mask_label] = best_neighbor
                                dilate = False
                                print(f'Use biggest neighbord label {label} -> {best_neighbor}')
                                label = best_neighbor
                                find_neighbor = True
                                # Si la taille après fusion dépasse la taille maximal, appliquer l'érosion (peut être ne pas fusionner)
                                if max_neighbor_size < max_cluster_size:
                                    mask_label = dilated_image == label
                                    ones = np.argwhere(mask_label == 1).shape[0]
                                    while ones > max_cluster_size:
                                        mask_label = morphology.erosion(mask_label, morphology.disk(3))
                                        ones = np.argwhere(mask_label == 1).shape[0]

                # Si aucun voisin trouvé, dilater la région
                if dilate:
                    mask_label = morphology.dilation(mask_label, morphology.square(3))
                    dilated_image[(mask_label)] = label
                    nb_test += 1

                if not dilate:
                    break
                
            # Si aucun voisin trouvé après nb_attempt, supprimer ou conserver la région
            if not find_neighbor:
                if ones < min_cluster_size:
                    mask_label = dilated_image == label
                    ones = np.argwhere(mask_label == 1).shape[0] 
                    # Si l'objet dilaté ne vérifie pas la condition minimum
                    if ones < min_cluster_size:
                        res[mask_label] = 0
                        print(f'Remove label {region.label}')
                    else:
                        # Si l'objet dilaté ne vérifie pas la condition maximum
                        while ones > max_cluster_size:
                            mask_label = morphology.erosion(mask_label, morphology.square(3))
                            ones = np.argwhere(mask_label == 1).shape[0]
                        
                        res[mask_label] = region.label
                        print(f'Keep label dilated {region.label}')

            # Mettre à jour les régions pour tenir compte des changements
            regions = measure.regionprops(res)
            regions = sorted(regions, key=lambda r: r.area)
            len_regions = len(regions)
            i = 0
            continue
        else:
            mask_label = res == region.label
            mask_before_erosion = np.copy(mask_label)
            while ones > max_cluster_size:
                mask_label = morphology.erosion(mask_label, morphology.square(3))
                ones = np.argwhere(mask_label == 1).shape[0]

            res[mask_before_erosion & ~mask_label] = background

            # Si le cluster est assez grand, on le conserve tel quel
            print(f'Keep label {region.label}')
            
        i += 1

    return res

def find_clusters(image, threshold, clusters_to_ignore=None, background=0):
    """
    Traverse the clusters in an image and return the clusters whose size is greater than a given threshold.
    
    :param image: np.array, 2D image with values representing the clusters
    :param threshold: int, minimum size of the cluster to be considered
    :param background: int, value representing the background (default: 0)
    :param clusters_to_ignore: list, list of clusters to ignore (default: None)
    :return: list, list of cluster IDs whose size is greater than the threshold
    """
    # Initialize the list of valid clusters to return
    valid_clusters = []
    
    # If no clusters to ignore are provided, initialize with an empty list
    if clusters_to_ignore is None:
        clusters_to_ignore = []
    
    # Create a mask where the background is ignored
    mask = image != background
    
    # Label the clusters in the image
    cluster_ids = np.unique(image[mask])
    cluster_ids = cluster_ids[~np.isnan(cluster_ids)]
    
    # Traverse each cluster and check its size
    for cluster_id in cluster_ids:
        # Skip the cluster if it's in the ignore list
        if cluster_id == clusters_to_ignore:
            continue
        
        # Calculate the size of the cluster
        cluster_size = np.sum(image == cluster_id)
        
        # If the cluster size exceeds the threshold, add it to the list
        if cluster_size > threshold:
            valid_clusters.append(cluster_id)
    
    return valid_clusters

def split_large_clusters(image, size_threshold, min_cluster_size, background):
    labeled_image = np.copy(image)
    
    # Obtenir les propriétés des régions labellisées
    regions = measure.regionprops(labeled_image)
    
    # Initialiser une image pour les nouveaux labels après division
    new_labeled_image = np.copy(labeled_image)
    changes_made = False

    for region in regions:

        if region.label in background:
            continue
        
        if region.area > size_threshold:
            # Si la région est plus grande que le seuil, la diviser
            
            # Extraire le sous-image du cluster
            minr, minc, maxr, maxc = region.bbox
            region_mask = (labeled_image[minr:maxr, minc:maxc] == region.label)
            
            # Obtenir les coordonnées des pixels du cluster
            coords = np.column_stack(np.nonzero(region_mask))
            # Appliquer K-means pour diviser en 2 clusters
            if len(coords) > 1:  # Assurez-vous qu'il y a suffisamment de points pour appliquer K-means
                clusterer = KMeans(n_clusters=2, random_state=42, n_init=10).fit(coords)
                #clusterer = HDBSCAN(min_cluster_size=size_threshold).fit(coords)
                labels = clusterer.labels_
                
                # Créer deux nouveaux labels
                new_label_1 = new_labeled_image.max() + 1
                new_label_2 = new_labeled_image.max() + 2
                
                # Assigner les nouveaux labels aux pixels correspondants
                new_labeled_image[minr:maxr, minc:maxc][region_mask] = np.where(labels == 0, new_label_1, new_label_2)
                
                changes_made = True
    
    # Si des changements ont été effectués, vérifier s'il y a des clusters à fusionner
    if changes_made:
        new_labeled_image = split_large_clusters(new_labeled_image, size_threshold, min_cluster_size, background)
    
    return new_labeled_image

def relabel_clusters(cluster_labels, started):
    """
    Réorganise les labels des clusters pour qu'ils soient séquentiels et croissants.
    """
    unique_labels = np.sort(np.unique(cluster_labels[~np.isnan(cluster_labels)]))

    relabeled_image = np.copy(cluster_labels)
    for ncl, cl in enumerate(unique_labels):
        relabeled_image[cluster_labels == cl] = ncl + started
    return relabeled_image

def load_raster_targets(dir_raster: Path, dates: list, lat, lon, dept, resolution) -> xr.Dataset:
    """Load groundwater level rasters into an xarray."""
    data_vars = {}
    for var in ["occurence", "burned_area", "time_intervention"]:
        file = dir_raster / var / 'bin' / resolution / f"{dept}binScale0.pkl"
        if not file.is_file():
            continue
        values = pickle.load(open(file, "rb"))
        data_vars[var] = (("latitude", "longitude", "date"), values)

    coords = {"latitude": lat, "longitude": lon, "date": dates}
    return xr.Dataset(data_vars, coords=coords)

def concat_xarrays(dir_raster: Path, dates: list, dept, path_to_latitude, resolution) -> xr.Dataset:
    """Concatenate all available rasters into a single xarray dataset.
    
    Parameters
    ----------
    dir_raster : Path
        Directory containing the pickled rasters.
    dates : list
        Dates associated with the rasters.

    Returns
    -------
    xr.Dataset
        A merged dataset containing every raster that could be loaded.
    """

    print(f'Save into a datacube')

    check_and_create_path(dir_raster / 'datacube' / dept / resolution)

    latitude = read_object(f'latitude.pkl', path_to_latitude / dept / 'raster' / resolution)
    longitude = read_object(f'longitude.pkl', path_to_latitude  / dept / 'raster' / resolution)

    latitude = latitude[:, 0]
    longitude = longitude[0]

    datasets = load_raster_targets(dir_raster, dates, latitude, longitude, dept, resolution)

    if not datasets:
        raise ValueError("No raster data found in the provided directory")
    
    f = open(dir_raster / 'datacube' / dept / resolution / f'datacube.pkl',"wb")
    pickle.dump(datasets,f)