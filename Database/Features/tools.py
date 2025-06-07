import firedanger
import copy
import meteostat
import datetime as dt
from pathlib import Path
import random

from sympy import EX, true
random.seed(0)
import pandas as pd
import geopandas as gpd
import os
from sklearn.cluster import KMeans
import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata
from sklearn.preprocessing import MinMaxScaler
from shapely.geometry import shape, Point
import math
from osgeo import gdal, ogr
from scipy.signal import fftconvolve as scipy_fft_conv
from astropy.convolution import convolve_fft
import rasterio
import rasterio.features
import rasterio.warp
from skimage import img_as_float
from skimage import transform
from sklearn.linear_model import LinearRegression
import pickle
import json
import warnings
from skimage import morphology
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, normalize
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.geometry import box
from skimage import morphology
from skimage.segmentation import watershed
from scipy.interpolate import interp1d
from scipy import ndimage as ndi
from skimage.filters import rank
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from scipy.interpolate import griddata
import sys
from dico_departements import *
import time
import requests
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from shapely.geometry import Point
import rasterio
from rasterio.mask import mask

##################################################################################################
#                                       Meteostat
##################################################################################################

def compute_fire_indices(point, date_debut, date_fin, saison_feux):
    """
    Compute a comprehensive set of wildfire risk indices (e.g., FFMC, DMC, DC, ISI, FWI, KBDI) from Meteostat hourly data.
    It performs time-windowed aggregation, correction for missing values, and prepares weather-derived fire indicators.
    """
    meteostat.Point.radius = 200000
    meteostat.Point.alt_range = 1000
    meteostat.Point.max_count = 5
    location = meteostat.Point(point[0], point[1])
    # logger.info(f"Calcul des indices incendie pour le point de coordonnées {point}")
    df = meteostat.Hourly(location, date_debut-dt.timedelta(hours=24), date_fin)
    df = df.normalize()
    df = df.fetch()
    assert len(df)>0
    df.drop(['tsun', 'coco', 'wpgt'], axis=1, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df['snow'] = df['snow'].fillna(0)
    df['prcp'] = df['prcp'].fillna(0)
    df['rhum'] = df['rhum'].apply(lambda x : min(x, 100))
    df.reset_index(inplace=True)
    df.rename({'time': 'creneau'}, axis=1, inplace=True)
    df['wspd'] =  df['wspd'] * 1000 / 3600
    df.sort_values(by='creneau', inplace=True)
    # Calculer la somme des précipitations des 24 heures précédentes
    df['prec24h'] = df['prcp'].rolling(window=24, min_periods=1).sum()
    df['snow24h'] = df['snow'].rolling(window=24, min_periods=1).sum()
    df['hour'] = df['creneau'].dt.hour
    df['prec24h12'] = np.where(df['hour'] == 12, df['prec24h'], np.nan)
    df['snow24h12'] = np.where(df['hour'] == 12, df['snow24h'], np.nan)
    df['prec24h12'].ffill(inplace=True)
    df['snow24h12'].ffill(inplace=True)
    for col in ['temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd']:
        df[f'{col}16'] = df[col].copy()
        df[f'{col}16'] = np.where(df['hour'] == 16, df[f'{col}16'], np.nan)
        df[f'{col}16'].ffill(inplace=True)
    df['prec24h16'] = np.where(df['hour'] == 16, df['prec24h'], np.nan)
    df['snow24h16'] = np.where(df['hour'] == 16, df['snow24h'], np.nan)
    df['prec24h16'].ffill(inplace=True)
    df['snow24h16'].ffill(inplace=True)

    for col in ['temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd']:
        df[f'{col}12'] = df[col].copy()
        df[f'{col}12'] = np.where(df['hour'] == 12, df[f'{col}12'], np.nan)
        df[f'{col}12'].ffill(inplace=True)

    df['temp15h'] = df['temp'].copy()
    df['temp15h'] = np.where(df['hour'] == 15, df['temp15h'], np.nan)
    df['temp15h'].ffill(inplace=True)
    df['rhum15h'] = df['rhum'].copy()
    df['rhum15h'] = np.where(df['hour'] == 15, df['rhum15h'], np.nan)
    df['rhum15h'].ffill(inplace=True)
    df['temp12h'] = df['temp'].copy()
    df['temp12h'] = np.where(df['hour'] == 12, df['temp12h'], np.nan)
    df['temp12h'].ffill(inplace=True)
    df['rhum12h'] = df['rhum'].copy()
    df['rhum12h'] = np.where(df['hour'] == 12, df['rhum12h'], np.nan)
    df['rhum12h'].ffill(inplace=True)
    df.drop('hour', axis=1, inplace=True)

    df.set_index(df['creneau'], inplace=True)
    df.drop('creneau', axis=1, inplace=True)
    daily_max_temp = df.resample('D').max()
    daily_max_temp['temp24max'] = daily_max_temp['temp'].shift(1)
    df = df.merge(daily_max_temp['temp24max'].asfreq('h', method='ffill'), left_index=True, right_index=True, how='left')    
    
    daily_prec = df.resample('D').sum()
    daily_prec['prec24veille'] = daily_max_temp['prcp'].shift(1)
    df = df.merge(daily_prec['prec24veille'].asfreq('h', method='ffill'), left_index=True, right_index=True, how='left')    
    # Somme des précipitations de la semaine écoulée, pour le KBDI
    df['sum_rain_last_7_days'] = df['prcp'].rolling('7D').sum()
    df['sum_snow_last_7_days'] = df['snow'].rolling('7D').sum()
    
    df.reset_index(inplace=True)    
    # Somme des précipitations consécutives, toujours pour le KBDI
    df['no_rain'] = df['prcp'] < 1.8 # Identifier les jours sans précipitations
    df['consecutive_rain_group'] = (df['no_rain']).cumsum() # Calculer les groupes de jours consécutifs avec précipitations
    df['sum_consecutive_rainfall'] = df.groupby('consecutive_rain_group')['prcp'].transform('sum') # Calculer la somme des précipitations pour chaque groupe de jours consécutifs
    df.loc[df['no_rain'], 'sum_consecutive_rainfall'] = 0 # Réinitialiser la somme à 0 pour les jours sans pluie
    df.drop(['no_rain', 'consecutive_rain_group'], axis=1, inplace=True)
    # On peut maintenant calculer les indices
    df = df.loc[df.creneau>=date_debut]
    df.reset_index(inplace=True)

    t = time.time()
    df = df.loc[df.creneau.dt.hour == 12]
    temps = df['temp'].to_numpy()
    temps12 = df['temp12h'].to_numpy()
    temps15 = df['temp15h'].to_numpy()
    temps24max = df['temp24max'].to_numpy()
    wspds = df['wspd'].to_numpy()
    rhums = df['rhum'].to_numpy()
    rhums12 = df['rhum12h'].to_numpy()
    rhums15 = df['rhum15h'].to_numpy()
    months = df['creneau'].dt.month
    months += 1
    months = months.to_numpy()
    prec24h12s = df['prec24h12'].to_numpy()
    prec24hs = df['prec24h'].to_numpy()
    prec24veilles = df['prec24veille'].to_numpy()
    snow24 = df['snow24h'].to_numpy()
    sum_rain_last_7_days = df['sum_rain_last_7_days'].to_numpy()
    sum_snow_last_7_days = df['sum_snow_last_7_days'].to_numpy()
    sum_consecutive_rainfall = df['sum_consecutive_rainfall'].to_numpy()    
    months = df['creneau'].dt.month.to_numpy() + 1
    latitudes = np.full_like(temps, point[0])

    # Days since last significant rain
    treshPrec24 = 1.8
    dsr = np.empty_like(prec24h12s)
    dsr[0] = int(prec24hs[0] > treshPrec24)
    for i in range(1, len(prec24hs)):
        dsr[i] = dsr[i - 1] + 1 if prec24h12s[i] < treshPrec24 else 0
    df['days_since_rain'] = dsr

    # Drought Code (DC) using numpy
    """
    The Drought Code (DC) is part of the Canadian Forest Fire Weather Index (FWI) System, 
    which is a comprehensive system used to estimate forest fire risk. DC specifically measures 
    the long-term drying effects of dry weather on deep forest floor organic layers. In other words, 
    it evaluates moisture loss in deep and compact organic matter that can support forest fire 
    development even after prolonged periods without rain.
    """
    dc = np.empty_like(temps)
    dc[0] = 0
    consecutive = 0
    for i in range(1, len(temps)):
        if temps[i] > 12 and snow24[i] < 1:
            consecutive += 1
        elif consecutive < 3:
            consecutive = 0

        if consecutive < 3:
            dc[i] = 0
        elif consecutive == 3:
            dc[i] = 15
            consecutive += 1
        else:
            dc[i] = firedanger.indices.dc(temps[i], prec24h12s[i], months[i], latitudes[i], dc[i-1])

    df['dc'] = dc

    # Fine Fuel Moisture Code (FFMC) using numpy
    """
    The Fine Fuel Moisture Code (FFMC) is another component of the Canadian Forest Fire Weather Index (FWI) System, 
    designed to estimate the moisture content of fine surface fuels that ignite easily and contribute to fire spread. 
    These fuels include dead leaves, twigs, grasses, and small branches under 6 mm in diameter.
    """
    ffmc = np.empty_like(temps)
    ffmc[0] = 0
    consecutive = 0
    for i in range(1, len(temps)):
        if temps[i] > 12 and snow24[i] < 1:
            consecutive += 1
        elif consecutive < 3:
            consecutive = 0

        if consecutive < 3:
            ffmc[i] = 0
        elif consecutive == 3:
            ffmc[i] = 6
            consecutive += 1
        else:
            ffmc[i] = firedanger.indices.ffmc(temps[i], prec24h12s[i], wspds[i], rhums[i], ffmc[i-1])

    df['ffmc'] = ffmc

    # Duff Moisture Code (DMC)
    """
    The Duff Moisture Code (DMC) is another important indicator in the Canadian FWI System. 
    Unlike FFMC which targets light surface fuels, DMC focuses on the moisture content in the 
    slightly deeper organic layers just below the surface. These fuels are thicker and slower to ignite, 
    but once ignited, they can sustain combustion for longer.
    """
    dmc = np.empty_like(temps)
    dmc[0] = 0
    consecutive = 0
    for i in range(1, len(temps)):
        if temps[i] > 12 and snow24[i] < 1:
            consecutive += 1
        elif consecutive < 3:
            consecutive = 0

        if consecutive < 3:
            dmc[i] = 0
        elif consecutive == 3:
            dmc[i] = 85
            consecutive += 1
        else:
            dmc[i] = firedanger.indices.dmc(temps[i], prec24h12s[i], rhums[i], months[i], latitudes[i], dmc[i-1])

    df['dmc'] = dmc

    # Initial Spread Index (ISI)
    """
    The Initial Spread Index (ISI) predicts the potential initial spread rate of a newly ignited fire, 
    based on fine fuel moisture and wind speed.
    """
    df['isi'] = df.apply(lambda x: firedanger.indices.isi(x.wspd, x.ffmc), axis=1)

    # Buildup Index (BUI)
    """
    The Buildup Index (BUI) represents the total amount of fuel available for combustion, 
    focusing on medium to heavy fuels. It provides an estimation of how intense or long-lasting a fire could be.
    """
    df['bui'] = firedanger.indices.bui(df['dmc'], df['dc'])

    # Fire Weather Index (FWI)
    """
    The Fire Weather Index (FWI) is the main output of the Canadian FWI System. 
    It reflects the combined effects of weather conditions on fire intensity and spread potential.
    """
    df['fwi'] = firedanger.indices.fwi(df['isi'], df['bui'])

    # Daily Severity Rating (DSR)
    """
    The Daily Severity Rating (DSR) translates the Fire Weather Index (FWI) into a scale 
    reflecting how severe and difficult a potential fire might be to control on that day.
    """
    df['daily_severity_rating'] = firedanger.indices.daily_severity_rating(df['fwi'])

    # Nesterov Index
    """
    The Nesterov index is used to assess wildfire risk, particularly in grass- and shrub-dominated regions. 
    It is calculated using temperature, humidity, and precipitation, and indicates surface dryness and fuel availability.
    """
    nesterov = np.empty_like(temps)
    nesterov[0] = 0
    start = False
    for i in range(1, len(temps)):
        if prec24hs[i] > 1:
            start = True
        if start: 
            nesterov[i] = firedanger.indices.nesterov(temps15[i], rhums15[i], prec24hs[i], nesterov[i-1])
        else:
            nesterov[i] = 0
    df['nesterov'] = nesterov

    # Munger Drought Index
    """
    The Munger index is a drought indicator that accumulates over time based on low rainfall events. 
    It helps evaluate the build-up of dry conditions favorable to fire spread.
    """
    munger = np.empty_like(temps)
    munger[0] = 0
    start = False
    for i in range(1, len(temps)):
        if prec24hs[i] > 0.05:
            start = True
        if start: 
            munger[i] = firedanger.indices.munger(prec24hs[i], munger[i-1])
        else:
            munger[i] = 0
    df['munger'] = munger

    # Keetch-Byram Drought Index (KBDI)
    """
    The Keetch-Byram Drought Index (KBDI) estimates soil moisture deficit and fire potential.
    Values range from 0 (saturated soil) to 800 (extreme drought conditions).
    """
    dg = meteostat.Hourly(location, 
                        dt.datetime(date_debut.year, 1, 1), 
                        min(dt.datetime(date_debut.year+1, 1, 1), dt.datetime.now()))
    dg = dg.normalize()
    dg = dg.fetch()
    pAnnualAvg = dg['prcp'].mean()  # Average annual rainfall [mm]
    kbdi = np.empty_like(temps)
    kbdi[0] = 0
    start = False
    for i in range(1, len(temps)):
        if sum_rain_last_7_days[i] > 152:
            start = True
        if start:
            kbdi[i] = max(0, min(800, firedanger.indices.kbdi(temps24max[i], 
                                                            prec24veilles[i],
                                                            kbdi[i-1], 
                                                            sum_consecutive_rainfall[i],
                                                            sum_rain_last_7_days[i],
                                                            30,  # Rain threshold to initialize [mm]
                                                            pAnnualAvg)))
        else:
            kbdi[i] = 0
    df['kbdi'] = kbdi

    # Angstroem Index
    """
    The Angstroem Index is a quick-fire potential indicator based on relative humidity and temperature.
    It is primarily used to estimate vegetation dryness and surface fire probability.
    """
    angstroem = np.empty_like(temps)
    angstroem[0] = 0
    for i in range(1, len(temps)):
        angstroem[i] = firedanger.indices.angstroem(temps12[i], rhums12[i-1])
    df['angstroem'] = angstroem

    # Ensure non-negative and valid ranges
    df['dc'] = df['dc'].apply(lambda x : max(x, 0))
    df['ffmc'] = df['ffmc'].apply(lambda x : max(x, 0))
    df['dmc'] = df['dmc'].apply(lambda x : max(x, 0))
    df['isi'] = df['isi'].apply(lambda x : max(x, 0))
    df['bui'] = df['bui'].apply(lambda x : max(x, 0))
    df['fwi'] = df['fwi'].apply(lambda x : max(x, 0))
    df['daily_severity_rating'] = df['daily_severity_rating'].apply(lambda x : max(x, 0))
    df['nesterov'] = df['nesterov'].apply(lambda x : max(x, 0))
    df['munger'] = df['munger'].apply(lambda x : max(x, 0))
    df['kbdi'] = df['kbdi'].apply(lambda x : max(x, 0))
    df['kbdi'] = df['kbdi'].apply(lambda x : min(x, 800))
    df['angstroem'] = df['angstroem'].apply(lambda x : max(x, 0))

    return df

def get_fire_indices(point, date_debut, date_fin, departement):
    if departement not in SAISON_FEUX.keys():
        SAISON_FEUX[departement] = {}
        SAISON_FEUX[departement]['mois_debut'] = 3
        SAISON_FEUX[departement]['jour_debut'] = 1
        SAISON_FEUX[departement]['mois_fin'] = 11
        SAISON_FEUX[departement]['jour_fin'] = 1

    for annee in range(date_debut.year, date_fin.year+1):
        debut = max(date_debut, dt.datetime(annee, 1, 1))
        fin = min(date_fin, dt.datetime(annee+1, 1, 1))
        debut_saison = dt.datetime(annee, 
                                   SAISON_FEUX[departement]['mois_debut'], 
                                   SAISON_FEUX[departement]['jour_debut'])

        fin_saison = dt.datetime(annee, 
                                   SAISON_FEUX[departement]['mois_fin'], 
                                   SAISON_FEUX[departement]['jour_fin'])
        
        dg = compute_fire_indices(point, debut, debut_saison, False)
        dg2 = compute_fire_indices(point, debut_saison, fin_saison, True)
        if fin_saison < fin:
            dg3 = compute_fire_indices(point, fin_saison, fin, False)
            if 'df' not in locals():
                df = pd.concat((dg, dg2, dg3)).reset_index(drop=True)
            else:
                df = pd.concat((df, dg, dg2, dg3)).reset_index(drop=True)
        else:
            if 'df' not in locals():
                df = pd.concat((dg, dg2)).reset_index(drop=True)
            else:
                df = pd.concat((df, dg, dg2)).reset_index(drop=True)
    df = df[(df['creneau'] >= date_debut) & (df['creneau'] <= date_fin)]
    return df 

def construct_historical_meteo(start, end, region, dir_meteostat, departement):
    """
    Generate historical weather and fire risk index data across a gridded set of points covering a region.
    """
    START = dt.datetime.strptime(start, '%Y-%m-%d') #- dt.timedelta(days=10)
    END = dt.datetime.strptime(end, '%Y-%m-%d')

    END += dt.timedelta(hours=1)
    if not (dir_meteostat / 'liste_de_points.pkl').is_file():
            N = 11
            range_x = np.linspace(
                *region.iloc[0].geometry.buffer(0.15).envelope.boundary.xy[0][:2], N)
            range_y = np.linspace(
                *region.iloc[0].geometry.buffer(0.15).envelope.boundary.xy[1][1:3], N)
            points = []
            for point_y in range_y:
                for point_x in range_x:
                    if region.iloc[0].geometry.buffer(0.15).contains(Point((point_x, point_y))):
                        points.append((point_y, point_x))
            print(f"Nombre de points de surveillance pour Meteostat : {len(points)}")
            print(f"On sauvegarde ces points")
            with open(dir_meteostat / 'liste_de_points.pkl', 'wb') as f:
                pickle.dump(points, f)
    else:
        print("On relit les points de Meteostat")
        with open(dir_meteostat / 'liste_de_points.pkl', 'rb') as f:
            points = pickle.load(f)

    print("On récupère les variables du risque d'incendie par hexagone")
    data_plein, data_creux, liste = {}, {}, []
    for index, point in enumerate(sorted(points)):
        print(f"Intégration du point de coordonnées {point}")
        data_plein[point] = get_fire_indices(point, START, END, departement)
        data_plein[point]['latitude'] = point[0]
        data_plein[point]['longitude'] = point[1]
        liste.append(data_plein[point])
    
    def get_date(x):
        return x.strftime('%Y-%m-%d')

    liste = pd.concat(liste)

    liste['creneau'] = liste['creneau'].apply(get_date)
    liste.sort_values('creneau', inplace=True)
    liste.reset_index(drop=False)
    return liste

def check_and_create_path(path: Path):
    """
    Create a directory or file path if it doesn't already exist.
    """
    path_way = path.parent if path.is_file() else path

    path_way.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.touch()

def create_grid_cems(df, time, delta, variables):
    """
    Filter a dataframe for a specific date window and selected variables.
    """
    interdate = dt.datetime.strptime(time, '%Y-%m-%d') - dt.timedelta(days=delta)
    interdate = interdate.strftime('%Y-%m-%d')
    dff = df[(df["creneau"] >= interdate) & (df["creneau"] <= time)]
    dff[variables] = dff[variables].astype(float)
    if len(dff) == 0:
        return None
    return dff.reset_index(drop=True)

def interpolate_gridd(var, grid, newx, newy, met, fill_value=0):
    """
    Interpolate a variable onto a new grid using a specified method (e.g., cubic, linear).
    """
    x = grid['longitude'].values
    y = grid["latitude"].values
    points = np.zeros((y.shape[0], 2))
    points[:,0] = x
    points[:,1] = y
    return griddata(points, grid[var].values, (newx, newy), method=met, fill_value=0)

def create_dict_from_arry(array):
    """
    Convert an array of variable names into a dictionary with default zero values.
    """
    res = {}
    for var in array:
        res[var] = 0
    return res

def myRasterization(geo, tif, maskNan, sh, column):
    """
    Rasterize vector data using cluster labels over a masked image.
    """
    res = np.full(sh, np.nan, dtype=float)
    
    if maskNan is not None:
        res[maskNan[:,0], maskNan[:,1]] = np.nan

    for index, row in geo.iterrows():
        #inds = indss[row['hex_id']]
        """ mask = np.zeros((sh[0], sh[1]))
        cv2.fillConvexPoly(mask, inds, 1)
        mask = mask > 0"""
        mask = tif == row['cluster']
        res[mask] = row[column]

    return res

def rasterise_meteo_data(h3, maskh3, cems, sh, dates, dir_output):
    """
    Rasterize meteorological and fire index variables into a spatiotemporal dataset.
    """
    cems_variables = [
                    'temp', 'dwpt',
                    'rhum', 'prcp', 'wdir', 'wspd', 'snow', 'prec24h', 'snow24h',
                    'dc',
                    'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
                    'isi', 'angstroem', 'bui',
                    'fwi', 'daily_severity_rating',
                    'temp16', 'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16', 'snow24h16',
                    'days_since_rain', 'sum_consecutive_rainfall',
                    'sum_rain_last_7_days',
                    'sum_snow_last_7_days',
                    ]
    
    lenDates = len(dates)
    print(lenDates)
    for var in cems_variables:
        print(var)
        spatioTemporalRaster = np.full((sh[0], sh[1], lenDates), np.nan)

        for i, date in enumerate(dates):
            if i % 200 == 0:
                print(date)

            #ddate = dt.datetime.strptime(date, "%Y-%m-%d")
            cems_grid = create_grid_cems(cems, date, 0, var)
            if cems_grid is None:
                print('Cems is None')

            cems_grid.fillna(0, inplace=True)

            h3[var] = interpolate_gridd(var, cems_grid, h3.longitude.values, h3.latitude.values, 'cubic')
            h3[var].fillna(value=np.nanmean(h3[var]), inplace=True)
            
            h3[var] = [max(0, u) for u in  h3[var].values]

            rasterVar = myRasterization(h3, maskh3, None, maskh3.shape, var)

            if rasterVar.shape != sh:
                rasterVar = resize(rasterVar, sh[0], sh[1], 1)

            spatioTemporalRaster[:,:, i] = rasterVar
        if var == 'daily_severity_rating':
            var = 'dailySeverityRating'
        outputName = var+'raw.pkl'
        f = open(dir_output / outputName,"wb")
        pickle.dump(spatioTemporalRaster, f)

def find_dates_between(start, end):
    """
    Return a list of date strings between two dates (exclusive).
    """
    start_date = dt.datetime.strptime(start, '%Y-%m-%d').date()
    end_date = dt.datetime.strptime(end, '%Y-%m-%d').date()

    delta = dt.timedelta(days=1)
    date = start_date
    res = []
    while date < end_date:
            res.append(date.strftime("%Y-%m-%d"))
            date += delta
    return res

def get_hourly(x):
    """
    Return the hour from a datetime object.
    """
    return x.hour

def get_date(x):
    """
    Return the date portion of a datetime object as a string.
    """
    return x.date().strftime('%Y-%m-%d')


warnings.filterwarnings("ignore")


##################################################################################################
#                                       Spatial
##################################################################################################

def read_tif(name):
    """
    Read a GeoTIFF file and return its data and spatial coordinates.
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
    """
    Find the nearest pixel coordinates to a geographic point.
    """
    lonIndex = (np.abs(lons - lon)).argmin()
    latIndex = (np.abs(lats - lat)).argmin()

    lonValues = lons.reshape(-1,1)[lonIndex]
    latValues = lats.reshape(-1,1)[latIndex]
    #print(lonValues, latValues)
    return np.where((lons == lonValues) & (lats == latValues))

def resize(input_image, height, width, dim):
    """
    Resize a 3D image (dim, H, W) to a specified shape.
    """
    img = img_as_float(input_image)
    img = transform.resize(img, (dim, height, width), mode='constant', order=0,
                 preserve_range=True, anti_aliasing=True)
    return np.asarray(img)

def resize_no_dim(input_image, height, width, mode='constant', order=0,
                 preserve_range=True, anti_aliasing=True):
    """
    Resize a 2D image to a specified shape.
    """
    img = img_as_float(input_image)
    img = transform.resize(img, (height, width), mode=mode, order=order,
                 preserve_range=preserve_range, anti_aliasing=anti_aliasing)
    return np.asarray(img)

def create_geocube(df, variables, reslons, reslats):
    """
    Convert geospatial points into raster grids for specified variables.
    """
    geo_grid = make_geocube(
        vector_data=df,
        measurements=variables,
        resolution=(reslons, reslats),
        rasterize_function=rasterize_points_griddata,
        fill = 0
    )
    return geo_grid

def raster_population(tifFile, tifFile_high, dir_output, reslon, reslat, dir_data):
    """
    Rasterize population point data to match a given raster shape.
    """
    population = pd.read_csv(dir_data / 'population' / 'population.csv')
    population = gpd.GeoDataFrame(population, geometry=gpd.points_from_xy(population.longitude, population.latitude))

    population = create_geocube(population, ['population'], -reslon, reslat)
    population = population.to_array().values[0]
    population = resize_no_dim(population, tifFile.shape[0], tifFile.shape[1])

    mask = np.argwhere(np.isnan(tifFile))
    population[mask[:,0], mask[:,1]] = np.nan

    outputName = 'population.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(population,f)

def raster_elevation(tifFile, dir_output, reslon, reslat, dir_data, dept):
    """
    Rasterize elevation data and align it with satellite base image dimensions.
    """
    #elevation = gpd.read_file(dir_data / 'elevation' / 'elevation.geojson')
    elevation = pd.read_csv(dir_data / 'elevation' / 'elevation.csv')
    elevation['latitude'] = elevation['latitude'].apply(lambda x : round(x, 3))
    elevation['longitude'] = elevation['longitude'].apply(lambda x : round(x, 3))

    elevation = elevation.groupby(['longitude', 'latitude'], as_index=False)['altitude'].mean()

    try:
        elevation = gpd.GeoDataFrame(elevation, geometry=gpd.points_from_xy(elevation.longitude, elevation.latitude))
        elevation = rasterisation(elevation, reslat, reslon, 'altitude', defval=0, name=dept)
        elevation = resize_no_dim(elevation, tifFile.shape[0], tifFile.shape[1])
    except Exception as e:
        print(e)
        elevation = np.zeros(tifFile.shape)
    minusMask = np.argwhere(tifFile == -1)
    minusMask = np.argwhere(np.isnan(tifFile))
    elevation[minusMask[:,0], minusMask[:,1]] = np.nan
    outputName = 'elevation.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(elevation,f)

valeurs_foret_attribut = { # Convert forest type to int
    "Châtaignier": 1,
    "Chênes décidus": 2,
    "Chênes sempervirents": 3,
    "Conifères": 4,
    "Douglas": 5,
    "Feuillus": 6,
    "Hêtre": 7,
    "Mélèze": 8,
    "Mixte": 9,
    "NC": 10,
    "NR": 11,
    "Pin à crochets, pin cembro": 12,
    "Pin autre": 13,
    "Pin d'Alep": 14,
    "Pin laricio, pin noir": 15,
    "Pin maritime": 16,
    "Pin sylvestre": 17,
    "Pins mélangés": 18,
    "Peuplier": 19,
    "Robinier": 20,
    "Sapin, épicéa": 21
}

valeurs_cosia_couverture = { # Convert landcover to int
    'Building': 1,
    'Bare soil': 2,
    'Water surface': 3,
    'Conifer': 4,
    'Deciduous': 5,
    'Shrubland': 6,
    'Lawn': 7,
    'Crop': 8,
}

def arrondir_avec_seuil(array, seuil):
    """
    Round elements in an array based on a custom decimal threshold.
    """
    # Séparer la partie entière et la partie décimale
    partie_entière = np.floor(array)
    partie_décimale = array - partie_entière

    # Condition pour arrondir au supérieur
    arrondir_au_sup = partie_décimale >= seuil

    # Ajouter 1 à la partie entière là où on doit arrondir au supérieur
    partie_entière[arrondir_au_sup] += 1

    # Retourner la partie entière comme résultat arrondi
    return partie_entière

def raster_argile(tifFile, tifFile_high, dir_output, reslon, reslat, dir_data, dept):
    """
    Rasterize clay level data, threshold and resample it.
    """
    argile = gpd.read_file(dir_data / 'argile' / 'argile.geojson')
    #argile = create_geocube(argile, ['NIVEAU'], -reslon, reslat)
    try:
        argile = rasterisation(argile, reslat, reslon, 'NIVEAU', defval=0, name=dept)
    except:
        argile = np.zeros(tifFile.shape)
    argile = resize_no_dim(argile, tifFile.shape[0], tifFile.shape[1])
    argile = arrondir_avec_seuil(argile, 0.2)
    argile[np.isnan(tifFile)] = np.nan

    outputName = 'argile.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(argile,f)

def raster_cosia(tifFile, tifFile_high, dir_output, reslon, reslat, dir_data, region):
    """
    Process and rasterize land cover data from CoSIA into layers by type and influence.
    """
    cosia = gpd.read_file(dir_data / 'cosia' / 'cosia.geojson')
    cosia = gpd.overlay(region, cosia)
    # Modify the classes
    cosia = cosia[(cosia['numero'] != 7)]
    cosia.loc[cosia[cosia['numero'].isin([18, 2, 4])].index, 'numero'] = 1
    cosia.loc[cosia[cosia['numero'].isin([3, 5])].index, 'numero'] = 2
    cosia.loc[cosia[cosia['numero'].isin([6, 7])].index, 'numero'] = 3
    cosia.loc[cosia[cosia['numero'] == 8].index, 'numero'] = 4
    cosia.loc[cosia[cosia['numero'] == 10].index, 'numero'] = 5
    cosia.loc[cosia[cosia['numero'] == 9].index, 'numero'] = 6
    cosia.loc[cosia[cosia['numero'] == 15].index, 'numero'] = 7
    cosia.loc[cosia[cosia['numero'].isin([14, 17, 16])].index, 'numero'] = 8

    cosia = rasterisation(cosia, reslat, reslon, 'numero', defval=0, name=dept)
    cosia = resize_no_dim(cosia, tifFile_high.shape[0], tifFile_high.shape[1])
    bands = valeurs_cosia_couverture.values()
    bands = np.asarray(list(bands))
    res = np.full((np.max(bands) + 1, tifFile.shape[0], tifFile.shape[1]), fill_value=0.0)
    res2 = np.full((tifFile.shape[0], tifFile.shape[1]), fill_value=np.nan)
    res3 = np.full((np.max(bands) + 1, tifFile.shape[0], tifFile.shape[1]), fill_value=0.0)

    unodes = np.unique(tifFile)
    cosia_2 = np.empty((np.max(bands) + 1,*cosia.shape))
    
    for band in bands:
        cosia_2[band] = influence_index(cosia == band, np.isnan(cosia))

    for node in unodes:
        if node not in tifFile_high:
            continue

        mask1 = tifFile == node
        mask2 = tifFile_high == node

        for band in bands:
            res[band, mask1] = (np.argwhere(cosia[mask2] == band).shape[0] / cosia[mask2].shape[0]) * 100
            res3[band, mask1] = np.nanmean(cosia_2[band, mask2])

        if res[:, mask1].shape[1] == 1:
            res2[mask1] = np.nanargmax(res[:, mask1])
        else:
            res2[mask1] = np.nanargmax(res[:, mask1][:,0])

    res[:, np.isnan(tifFile)] = np.nan
    res2[np.isnan(tifFile)] = np.nan
    res3[:, np.isnan(tifFile)] = np.nan
    
    outputName = 'cosia.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res,f)

    outputName = 'cosia_landcover.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res2,f)

    outputName = 'cosia_influence.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res3,f)

def raster_foret(tifFile, tifFile_high, dir_output, reslon, reslat, dir_data, dept):
    """
    Rasterize forest attribute data by tree type, and compute class influence per pixel.
    """
    foret = gpd.read_file(dir_data / 'BDFORET' / 'foret.geojson')
    foret = rasterisation(foret, reslat, reslon, 'code', defval=0, name=dept)
    foret = resize_no_dim(foret, tifFile_high.shape[0], tifFile_high.shape[1])
    bands = valeurs_foret_attribut.values()
    bands = np.asarray(list(bands))
    res = np.full((np.max(bands) + 1, tifFile.shape[0], tifFile.shape[1]), fill_value=0.0)
    res2 = np.full((tifFile.shape[0], tifFile.shape[1]), fill_value=np.nan)
    res3 = np.full((np.max(bands) + 1, tifFile.shape[0], tifFile.shape[1]), fill_value=0.0)

    unodes = np.unique(tifFile)
    foret_2 = np.empty((np.max(bands) + 1,*foret.shape))

    for band in bands:
        foret_2[band] = influence_index(foret == band, np.isnan(foret))

    for node in unodes:
        if node not in tifFile_high:
            continue

        mask1 = tifFile == node
        mask2 = tifFile_high == node

        for band in bands:
            res[band, mask1] = (np.argwhere(foret[mask2] == band).shape[0] / foret[mask2].shape[0]) * 100
            res3[band, mask1] = np.nanmean(foret_2[band, mask2])

        if res[:, mask1].shape[1] == 1:
            res2[mask1] = np.nanargmax(res[:, mask1])
        else:
            res2[mask1] = np.nanargmax(res[:, mask1][:,0])

    res[:, np.isnan(tifFile)] = np.nan
    res2[np.isnan(tifFile)] = np.nan
    res3[:, np.isnan(tifFile)] = np.nan

    outputName = 'foret.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res,f)

    outputName = 'foret_landcover.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res2,f)

    outputName = 'foret_influence.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res3,f)

def raster_osmnx(tifFile, tifFile_high, dir_output, reslon, reslat, dir_data, dept):
    """
    Rasterize OSM road data and calculate influence zones for each road type.
    """
    #osmnx, _, _ = read_tif(dir_data / 'osmnx' / 'osmnx.tif')
    osmnx = gpd.read_file(dir_data / 'osmnx' / 'osmnx.geojson')
    osmnx = rasterisation(osmnx, reslat, reslon, 'label', defval=0, name=dept)

    mask = np.isnan(tifFile_high)
    osmnx = resize_no_dim(osmnx, tifFile_high.shape[0], tifFile_high.shape[1])
    osmnx[mask] = np.nan
    bands = np.asarray([0,1,2,3,4,5])
    bands = bands[~np.isnan(bands)].astype(int)
    res = np.zeros(((np.nanmax(bands) + 1), tifFile.shape[0], tifFile.shape[1]), dtype=float)
    res2 = np.full((tifFile.shape[0], tifFile.shape[1]), fill_value=np.nan)
    res3 = np.full(((np.nanmax(bands) + 1), tifFile.shape[0], tifFile.shape[1]), fill_value=0.0, dtype=float)
    res4 = np.zeros((tifFile.shape[0], tifFile.shape[1]), dtype=float)
    
    osmnx_2 = np.empty(((np.nanmax(bands) + 1), *osmnx.shape))

    for band in bands:
        osmnx_2[band] = influence_index(osmnx == band, mask)

    unodes = np.unique(tifFile)
    for node in unodes:
        mask1 = tifFile == node
        mask2 = tifFile_high == node

        if True not in np.unique(mask2):
            continue
        
        for band in bands:
            res[band, mask1] = (np.argwhere(osmnx[mask2] == band).shape[0] / osmnx[mask2].shape[0]) * 100
            res3[band, mask1] = np.nanmean(osmnx_2[band, mask2])
            if band > 0:
                res4[mask1] = res4[mask1] + res[band, mask1]

        if res[:, mask1].shape[1] == 1:
            res2[mask1] = np.nanargmax(res[:, mask1])
        else:
            res2[mask1] = np.nanargmax(res[:, mask1][:,0])

    res[:, np.isnan(tifFile)] = np.nan
    res2[np.isnan(tifFile)] = np.nan
    res3[:, np.isnan(tifFile)] = np.nan
    res4[np.isnan(tifFile)] = np.nan

    outputName = 'osmnx.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res,f)

    outputName = 'osmnx_landcover.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res2,f)

    outputName = 'osmnx_influence.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res3,f)

    outputName = 'osmnx_landcover_2.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res4,f)

def raster_sat_from_france(base, geo, dir_output, dir_france, dates):
    """
    Rasterize and align satellite imagery time series to match a base raster's shape.
    """
    size = '30m'
    res = np.full((5, base.shape[0], base.shape[1], len(dates)), np.nan)
    minusMask = np.argwhere(np.isnan(base))
    
    polygons = unary_union(geo.geometry) 

    for tifFile in dir_france.glob('*.tif'):
        tifFile = tifFile.as_posix()
        dateFile = tifFile.split('/')[-1]
        date = dateFile.split('.')[0]

        if date not in dates:
            continue

        i = dates.index(date)
        print(dateFile, i)

        with rasterio.open(tifFile) as src:
            # Masquage par polygone
            out_image, out_transform = mask(src, [polygons], crop=True, nodata=np.nan)
            out_image = out_image.astype(np.float32)
            out_image[out_image == src.nodata] = np.nan

            # Resize chaque bande à la forme de `base`
            for b in range(out_image.shape[0]):
                # Crée un tableau vide pour le résultat interpolé
                target = np.full(base.shape, np.nan, dtype=np.float32)

                #plt.imshow(out_image[b])
                #plt.show()

                """reproject(
                    source=out_image[b],
                    destination=target,
                    src_transform=out_transform,
                    src_crs=src.crs,
                    dst_transform=from_origin(0, 0, 1, 1),  # Remplacer si nécessaire
                    dst_crs=src.crs,
                    resampling=Resampling.nearest
                )"""

                target = resize_no_dim(out_image[b], base.shape[0], base.shape[1])
                #target[np.isnan(base)] = np.nan
                #plt.imshow(target)
                #plt.show()

                if i + 8 > res.shape[-1]:
                    length = res.shape[-1] - i - 1
                    target = np.repeat(target[:, :, np.newaxis], length, axis=-1)
                    res[b, :, :, i:-1] = target
                else:
                    target = np.repeat(target[:, :, np.newaxis], 8, axis=-1)
                    res[b, :, :, i:i+8] = target

    # Masque les pixels NaN d'origine
    res[:, minusMask[:, 0], minusMask[:, 1], :] = np.nan

    print(dir_output)
    outputName = 'sentinel.pkl'
    with open(dir_output / outputName, "wb") as f:
        pickle.dump(res, f)

    # Optionnel : une image moyenne ou indice NDVI par date
    return res

def rasterisation(h3, lats, longs, column='cluster', defval = 0, name='default', dir_output='/media/caron/X9 Pro/corbeille'):
    """
    Rasterize a GeoDataFrame to a raster using GDAL based on a spatial attribute.
    """
    #h3['cluster'] = h3.index

    h3.to_file(dir_output + '/' + name+'.geojson', driver='GeoJSON')

    input_geojson = dir_output + '/' + name+'.geojson'
    output_raster = dir_output + '/' + name+'.tif'

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

    # Oncrée un nouveau raster dataset et on passe de "coordonnées image" (pixels) à des coordonnées goréférencées
    driver = gdal.GetDriverByName('GTiff')
    output_ds = driver.Create(output_raster, width, height, 1, gdal.GDT_Float32)
    output_ds.GetRasterBand(1).Fill(defval)
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

    res, _, _ = read_tif(dir_output + '/' + name+'.tif')
    os.remove(dir_output + '/' + name+'.tif')
    return res[0]

def myFunctionDistanceDugrandCercle(outputShape, earth_radius=6371.0, resolution_lon=0.0002694945852352859, resolution_lat=0.0002694945852326214):
    """
    Compute a great-circle distance kernel centered on the image for influence modeling.
    """
    half_rows = outputShape[0] // 2
    half_cols = outputShape[1] // 2

    # Créer une grille de coordonnées géographiques avec les résolutions souhaitées
    latitudes = np.linspace(-half_rows * resolution_lat, half_rows * resolution_lat, outputShape[0])
    longitudes = np.linspace(-half_cols * resolution_lon, half_cols * resolution_lon, outputShape[1])
    latitudes, longitudes = np.meshgrid(latitudes, longitudes, indexing='ij')

    # Coordonnées du point central
    center_lat = latitudes[outputShape[0] // 2, outputShape[1] // 2]
    center_lon = longitudes[outputShape[0] // 2, outputShape[1] // 2]

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

def influence_index(categorical_array, mask):
    """
    Apply a convolution-based influence measure to a binary spatial class mask.
    """
    res = np.full(categorical_array.shape, np.nan)

    kernel = myFunctionDistanceDugrandCercle((30,30))
    #kernel = normalize(kernel, norm='l2')
    #res[mask[:,0], mask[:,1]] = (scipy_fft_conv(categorical_array, kernel, mode='same')[mask[:,0], mask[:,1]])

    res = convolve_fft(categorical_array, kernel, normalize_kernel=True, mask=mask)

    return res

def add_osmnx(clusterSum, dataset, dir_reg, kmeans, tifFile):
    """
    Integrate road density and proximity influence metrics into a cluster-level dataset.
    """
    print('Add osmnx')
    dir_data = dir_reg / 'osmnx'

    tif, _, _ = read_tif(dir_data / 'osmnx.tif')
    
    sentinel, lons, lats = read_tif(tifFile)
    nanmask = np.argwhere(np.isnan(sentinel[0]))
    sentinel = sentinel[0]

    coord = np.empty((sentinel.shape[0], sentinel.shape[1], 2))
    tif = resize(tif, sentinel.shape[0], sentinel.shape[1], 1).astype(int)[0]
    coord[:,:,0] = lats
    coord[:,:,1] = lons

    clustered = kmeans.predict(np.reshape(coord, (-1, coord.shape[-1])))
    clustered = clustered.reshape(sentinel.shape[0], sentinel.shape[1]).astype(float)
    clustered[nanmask[:,0], nanmask[:,1]] = np.nan

    mask = np.argwhere(~np.isnan(tif))
    density = influence_index(tif.astype(int), mask)

    dataset['highway_min'] = 0
    dataset['highway_max'] = 0
    dataset['highway_std'] = 0
    dataset['highway_mean'] = 0
    dataset['highway_sum'] = 0

    for cluster in clusterSum['cluster'].unique():
        clusterMask = clustered == cluster
        indexDataset = dataset[dataset['cluster'] == cluster].index.values

        dataset.loc[indexDataset, 'highway_sum'] = np.nansum(density[clusterMask])

        dataset.loc[indexDataset, 'highway_max'] = np.nanmax(density[clusterMask])

        dataset.loc[indexDataset, 'highway_min'] = np.nanmin(density[clusterMask])

        dataset.loc[indexDataset, 'highway_mean'] = np.nanmean(density[clusterMask])

        dataset.loc[indexDataset, 'highway_std'] =  np.nanstd(density[clusterMask])
    
    return dataset

def read_object(filename: str, path : Path):
    """
    Load a Python object from a pickle file if it exists.
    """
    if not (path / filename).is_file():
        print(f'{path / filename} not found')
        return None
    return pickle.load(open(path / filename, 'rb'))
