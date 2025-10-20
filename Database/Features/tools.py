import copy
import datetime as dt
import os
import pickle
import random
import time
import warnings
from itertools import chain
from pathlib import Path

import cv2
import firedanger
import geopandas as gpd
import meteostat
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from astropy.convolution import convolve_fft
from dico_departements import *
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata
from osgeo import gdal, ogr
from rasterio.mask import mask
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.interpolate import griddata
from shapely.geometry import Point
from shapely.ops import unary_union
from skimage import img_as_float, transform

random.seed(0)


##################################################################################################
#                                       Meteostat
##################################################################################################

def compute_fire_indices(point, date_debut, date_fin, saison_feux):
    meteostat.Point.radius = 200000
    meteostat.Point.alt_range = 1000
    meteostat.Point.max_count = 5
    location = meteostat.Point(point[0], point[1])
    # logger.info(f"Computing fire indices for the point with coordinates {point}")
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
    # Wind speed must be expressed in m/s
    df['wspd'] =  df['wspd'] * 1000 / 3600
    df.sort_values(by='creneau', inplace=True)
    # Calculate the total precipitation from the previous 24 hours
    df['prec24h'] = df['prcp'].rolling(window=24, min_periods=1).sum()
    df['snow24h'] = df['snow'].rolling(window=24, min_periods=1).sum()
    # Ensure prcp24 only contains values computed at noon (12:00)
    # by setting rows that do not correspond to noon to NaN,
    # then use ffill to propagate the last computed value
    df['hour'] = df['creneau'].dt.hour
    df['prec24h12'] = np.where(df['hour'] == 12, df['prec24h'], np.nan)
    df['snow24h12'] = np.where(df['hour'] == 12, df['snow24h'], np.nan)
    df['prec24h12'].ffill(inplace=True)
    df['snow24h12'].ffill(inplace=True)
    # 16:00 data used for Nicolas
    for col in ['temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd']:
        df[f'{col}16'] = df[col].copy()
        df[f'{col}16'] = np.where(df['hour'] == 16, df[f'{col}16'], np.nan)
        df[f'{col}16'].ffill(inplace=True)
    df['prec24h16'] = np.where(df['hour'] == 16, df['prec24h'], np.nan)
    df['snow24h16'] = np.where(df['hour'] == 16, df['snow24h'], np.nan)
    df['prec24h16'].ffill(inplace=True)
    df['snow24h16'].ffill(inplace=True)

    # 12:00 data used for Nicolas
    for col in ['temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd']:
        df[f'{col}12'] = df[col].copy()
        df[f'{col}12'] = np.where(df['hour'] == 12, df[f'{col}12'], np.nan)
        df[f'{col}12'].ffill(inplace=True)

    # 15:00 data used for Nesterov
    df['temp15h'] = df['temp'].copy()
    df['temp15h'] = np.where(df['hour'] == 15, df['temp15h'], np.nan)
    df['temp15h'].ffill(inplace=True)
    df['rhum15h'] = df['rhum'].copy()
    df['rhum15h'] = np.where(df['hour'] == 15, df['rhum15h'], np.nan)
    df['rhum15h'].ffill(inplace=True)
    # 12:00 data used for Angstroem
    df['temp12h'] = df['temp'].copy()
    df['temp12h'] = np.where(df['hour'] == 12, df['temp12h'], np.nan)
    df['temp12h'].ffill(inplace=True)
    df['rhum12h'] = df['rhum'].copy()
    df['rhum12h'] = np.where(df['hour'] == 12, df['rhum12h'], np.nan)
    df['rhum12h'].ffill(inplace=True)
    df.drop('hour', axis=1, inplace=True)

    # Maximum temperature from the previous day, for KBDI
    df.set_index(df['creneau'], inplace=True)
    df.drop('creneau', axis=1, inplace=True)
    daily_max_temp = df.resample('D').max()
    daily_max_temp['temp24max'] = daily_max_temp['temp'].shift(1)
    df = df.merge(daily_max_temp['temp24max'].asfreq('h', method='ffill'), left_index=True, right_index=True, how='left')    
    
    # Precipitation from the previous day, for KBDI
    daily_prec = df.resample('D').sum()
    daily_prec['prec24veille'] = daily_max_temp['prcp'].shift(1)
    df = df.merge(daily_prec['prec24veille'].asfreq('h', method='ffill'), left_index=True, right_index=True, how='left')    
    # Sum of the previous week's precipitation, for KBDI
    df['sum_rain_last_7_days'] = df['prcp'].rolling('7D').sum()
    df['sum_snow_last_7_days'] = df['snow'].rolling('7D').sum()
    
    df.reset_index(inplace=True)    
    # Sum of consecutive precipitation, also for KBDI
    df['no_rain'] = df['prcp'] < 1.8 # Identify days without precipitation
    df['consecutive_rain_group'] = (df['no_rain']).cumsum() # Calculate groups of consecutive days with precipitation
    df['sum_consecutive_rainfall'] = df.groupby('consecutive_rain_group')['prcp'].transform('sum') # Calculate the sum of precipitation for each group of consecutive days
    df.loc[df['no_rain'], 'sum_consecutive_rainfall'] = 0 # Reset the sum to 0 for rain-free days
    df.drop(['no_rain', 'consecutive_rain_group'], axis=1, inplace=True)
    # The indices can now be calculated
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

    # Days since rain
    treshPrec24 = 1.8
    dsr = np.empty_like(prec24h12s)
    dsr[0] = int(prec24hs[0] > treshPrec24)
    for i in range(1, len(prec24hs)):
        dsr[i] = dsr[i - 1] + 1 if prec24h12s[i] < treshPrec24 else 0
    df['days_since_rain'] = dsr

    """if not saison_feux: # -> not happy because I do not want to shrink my dataset, but we will see later
        df['dc'] = 0
        df['ffmc'] = 0
        df['dmc'] = 0
        df['isi'] = 0
        df['bui'] = 0
        df['fwi'] = 0
        df['daily_severity_rating'] = 0
        df['nesterov'] = 0
        df['munger'] = 0
        df['kbdi'] = 0
        df['angstroem'] = 0
        return df"""

    # Calculate the DC using numpy
    '''
    The Drought Code (DC) is part of the Canadian Forest Fire Weather Index (FWI) System, a comprehensive
    framework used to estimate wildfire risk. The DC specifically measures the long-term drying effects of
    sustained dry weather on deep forest fuels. In other words, it evaluates how much moisture has been removed
    from thick organic materials on the forest floor that can ignite and sustain a wildfire even without new
    moisture input for an extended period.
    '''
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
            continue

    df['dc'] = dc
    # Calculate the FFMC using numpy
    '''
    The Fine Fuel Moisture Code (FFMC) is another component of the Canadian Forest Fire Weather Index (FWI)
    System, designed to estimate the moisture content of fine, light fuels on the forest floor that can ignite
    quickly. It essentially measures how easily these fuels can ignite and sustain the initial spread of a
    wildfire. Fine fuels include items such as dead leaves, twigs, grass, and small branches with a diameter
    of less than 6 mm.
    '''
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
            continue

    df['ffmc'] = ffmc
    # Calculate the DMC
    '''
    The Duff Moisture Code (DMC) is another key indicator within the Canadian Forest Fire Weather Index (FWI)
    System. Unlike the Fine Fuel Moisture Code (FFMC), which evaluates the moisture content of fine, light
    surface fuels, the DMC focuses on the moisture content of litter layers and decomposing organic matter
    located just beneath the surface. These materials are thicker and less volatile than fine fuels, so they
    require more time to dry and ignite, but once ignited they can sustain a fire for an extended period.
    '''
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
            continue

    df['dmc'] = dmc
    # Calculate the remaining indices from the Canadian FWI
    '''
    The Initial Spread Index (ISI) is specifically designed to predict the initial rate of spread of a newly
    ignited fire based on current weather conditions.
    '''
    df['isi'] = df.apply(lambda x: firedanger.indices.isi(x.wspd, x.ffmc), axis=1)
    '''
    The Buildup Index (BUI) is designed to quantify the amount of fuel available to feed a wildfire, focusing
    primarily on medium and heavy fuels. The BUI estimates the total accumulated fuel and its ability to burn,
    providing a measure of a potential fire's heaviness and intensity.
    '''
    df['bui'] = firedanger.indices.bui(df['dmc'], df['dc'])
    '''
    The Fire Weather Index (FWI) is the main index of the Canadian Forest Fire Weather Index System. It provides
    an overall measure of wildfire danger by combining several sub-indices to estimate a fire's potential
    intensity. The FWI reflects the combined effects of current weather conditions on fire behavior, including
    spread and intensity.
    '''
    df['fwi'] = firedanger.indices.fwi(df['isi'], df['bui'])
    '''
    The Daily Severity Rating (DSR) is a component of the Canadian Forest Fire Weather Index System that provides
    a quantitative assessment of the potential intensity of a wildfire for a given day. The DSR translates the
    Fire Weather Index (FWI) into a scale that more directly reflects the potential difficulty of controlling a
    fire and the intensity and energy it might release.
    '''
    df['daily_severity_rating'] = firedanger.indices.daily_severity_rating(df['fwi'])
    # Calculate the Nesterov index
    '''
    The Nesterov index is an indicator used to assess wildfire risk. It is particularly helpful in regions where
    vegetation consists mainly of grasses and shrubs. The index is calculated from meteorological data—especially
    air temperature and precipitation amounts—and is designed to reflect surface dryness and the availability of
    fine fuels (such as dry grass) for combustion.
    '''
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

    # Calculate the Munger drought index
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

    # Calculate the KBDI
    """
    The Keetch-Byram Drought Index (KBDI) is primarily used to evaluate wildfire risk based on the amount of
    moisture present in the soil. The index typically ranges from 0 (saturated soil) to 800 (extremely dry
    conditions).
    """
    dg = meteostat.Hourly(location, 
                          dt.datetime(date_debut.year, 1, 1), 
                          min(dt.datetime(date_debut.year+1, 1, 1),dt.datetime.now()))
    dg = dg.normalize()
    dg = dg.fetch()
    pAnnualAvg = dg['prcp'].mean() # Annual rainfall average [mm].
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
                                                            30, # weekly rain threshold to initialize index [mm]
                                                            pAnnualAvg)))
        else:
            kbdi[i] = 0

    df['kbdi'] = kbdi

    # Calculate the Angstroem index
    '''
    A meteorological index primarily used to estimate the probability and intensity of wildfires. It focuses on
    two key weather variables: precipitation and relative humidity. The index provides a quick estimate of fire
    potential based on vegetation dryness and atmospheric conditions.
    '''
    angstroem = np.empty_like(temps)
    angstroem[0] = 0
    for i in range(1, len(temps)):
        angstroem[i] = firedanger.indices.angstroem(temps12[i], rhums12[i-1])
    df['angstroem'] = angstroem

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
        
        dg = compute_fire_indices(point, debut, fin, False)
        if 'df' not in locals():
                df = dg
        else:
            df = pd.concat((df, dg)).reset_index(drop=True)
            
    df = df[(df['creneau'] >= date_debut) & (df['creneau'] <= date_fin)]
    return df

def construct_historical_meteo(start, end, region, dir_meteostat, departement):
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
    Create a directory if it does not already exist.
    """
    path_way = path.parent if path.is_file() else path

    path_way.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.touch()

def create_grid_cems(df, time, delta, variables):
    interdate = dt.datetime.strptime(time, '%Y-%m-%d') - dt.timedelta(days=delta)
    interdate = interdate.strftime('%Y-%m-%d')
    dff = df[(df["creneau"] >= interdate) & (df["creneau"] <= time)]
    dff[variables] = dff[variables].astype(float)
    if len(dff) == 0:
        return None
    return dff.reset_index(drop=True)

def interpolate_gridd(var, grid, newx, newy, met, fill_value=0):
    x = grid['longitude'].values
    y = grid["latitude"].values
    points = np.zeros((y.shape[0], 2))
    points[:,0] = x
    points[:,1] = y
    return griddata(points, grid[var].values, (newx, newy), method=met, fill_value=0)

def create_dict_from_arry(array):
    res = {}
    for var in array:
        res[var] = 0
    return res

def myRasterization(geo, tif, maskNan, sh, column):

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
    return x.hour

def get_date(x):
    return x.date().strftime('%Y-%m-%d')


warnings.filterwarnings("ignore")


##################################################################################################
#                                       Spatial
##################################################################################################

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

def resize(input_image, height, width, dim):
    """
    Resize the input_image into heigh, with, dim
    """
    img = img_as_float(input_image)
    img = transform.resize(img, (dim, height, width), mode='constant', order=0,
                 preserve_range=True, anti_aliasing=True)
    return np.asarray(img)

def resize_no_dim(input_image, height, width, mode='constant', order=0,
                 preserve_range=True, anti_aliasing=True):
    """
    Resize the input_image into heigh, with, dim
    """
    img = img_as_float(input_image)
    img = transform.resize(img, (height, width), mode=mode, order=order,
                 preserve_range=preserve_range, anti_aliasing=anti_aliasing)
    return np.asarray(img)

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

def raster_population(tifFile, tifFile_high, dir_output, reslon, reslat, dir_data):
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

valeurs_foret_attribut = {
    "NoForest": 0,
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

def arrondir_avec_seuil(array, seuil):
    """
    Round the elements of a NumPy array according to a decimal threshold.
    If the decimal part is greater than or equal to the threshold, the element is rounded up.
    Otherwise, it is rounded down.

    :param array: The NumPy array to round.
    :param seuil: The decimal threshold for rounding (between 0 and 1).
    :return: The NumPy array rounded according to the threshold.
    """
    # Separate the integer and decimal parts
    partie_entière = np.floor(array)
    partie_décimale = array - partie_entière

    # Condition to round up
    arrondir_au_sup = partie_décimale >= seuil

    # Add 1 to the integer part where values must be rounded up
    partie_entière[arrondir_au_sup] += 1

    # Return the integer part as the rounded result
    return partie_entière

def raster_foret(tifFile, tifFile_high, dir_output, reslon, reslat, dir_data, dept):
    foret = gpd.read_file(dir_data / 'BDFORET' / 'foret.geojson')
    foret = rasterisation(foret, reslat, reslon, 'code', defval=0, name=dept)
    foret = resize_no_dim(foret, tifFile_high.shape[0], tifFile_high.shape[1])
    bands = valeurs_foret_attribut.values()
    bands = np.asarray(list(bands))
    res = np.full((np.max(bands) + 1, tifFile.shape[0], tifFile.shape[1]), fill_value=0.0)
    res2 = np.full((tifFile.shape[0], tifFile.shape[1]), fill_value=np.nan)

    unodes = np.unique(tifFile)

    for node in unodes:
        if node not in tifFile_high:
            continue

        mask1 = tifFile == node
        mask2 = tifFile_high == node

        for band in bands:
            res[band, mask1] = (np.argwhere(foret[mask2] == band).shape[0] / foret[mask2].shape[0]) * 100
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
    
def raster_sat_from_france(base, geo, dir_output, dir_france, dates):
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
            out_image, out_transform = mask(src, [polygons], crop=True, nodata=np.nan)
            out_image = out_image.astype(np.float32)
            out_image[out_image == src.nodata] = np.nan

            for b in range(out_image.shape[0]):
                target = np.full(base.shape, np.nan, dtype=np.float32)

                target = resize_no_dim(out_image[b], base.shape[0], base.shape[1])

                if i + 8 > res.shape[-1]:
                    length = res.shape[-1] - i - 1
                    target = np.repeat(target[:, :, np.newaxis], length, axis=-1)
                    res[b, :, :, i:-1] = target
                else:
                    target = np.repeat(target[:, :, np.newaxis], 8, axis=-1)
                    res[b, :, :, i:i+8] = target

    res[:, minusMask[:, 0], minusMask[:, 1], :] = np.nan

    print(dir_output)
    outputName = 'sentinel.pkl'
    with open(dir_output / outputName, "wb") as f:
        pickle.dump(res, f)

    return res

def rasterisation(h3, lats, longs, column='cluster', defval = 0, name='default', dir_output='/media/caron/X9 Pro1/corbeille', return_lat_lon=False):
    #h3['cluster'] = h3.index

    h3.to_file(dir_output + '/' + name+'.geojson', driver='GeoJSON')

    input_geojson = dir_output + '/' + name+'.geojson'
    output_raster = dir_output + '/' + name+'.tif'

    # If rasterizing based on a vector attribute, provide its name here
    attribute_name = column

    # Pixel size
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

    # Retrieve the raster extent
    x_min, x_max, y_min, y_max = source_layer.GetExtent()
    # Compute the number of pixels
    width = int((x_max - x_min) / pixel_size_x)
    height = int((y_max - y_min) / pixel_size_y)

    # Create a new raster dataset and convert from image coordinates (pixels) to georeferenced coordinates
    driver = gdal.GetDriverByName('GTiff')
    output_ds = driver.Create(output_raster, width, height, 1, gdal.GDT_Float32)
    output_ds.GetRasterBand(1).Fill(defval)
    output_ds.SetGeoTransform([x_min, pixel_size_x, 0, y_max, 0, -pixel_size_y])
    output_ds.SetProjection(source_layer.GetSpatialRef().ExportToWkt())

    if attribute_name != '' :
        # Rasterize according to the given attribute
        gdal.RasterizeLayer(output_ds, [1], source_layer, options=["ATTRIBUTE=" + attribute_name])
    else :
        # Rasterize so the raster takes the value 1 wherever a vector is present
        gdal.RasterizeLayer(output_ds, [1], source_layer)

    output_ds = None
    source_ds = None

    if not return_lat_lon:
        res, _, _ = read_tif(dir_output + '/' + name+'.tif')
        os.remove(dir_output + '/' + name+'.tif')
        return res[0]
    else:
        return read_tif(dir_output + '/' + name+'.tif')

def reclass_corine_by_index(array):
    """
    Reclassify a CORINE raster that uses indices (1 to 44) into 10 logical classes.
    The mapping is based on the position of the indices in the official CORINE order.

    Input:
        array : np.ndarray containing indices from 1 to 44 (e.g., [1, 2, ..., 44])

    Output:
        np.ndarray with values from 1 to 10 representing the logical classes
    """

    # Lookup table for the 10 classes (indexed from 0 to 43 → 44 values)
    # This table corresponds to the order of the indices in the raster
    mapping_10_classes = np.array([
        1, 1, 1, 1, 1, 1,        # 1–6 urban
        2, 2, 1, 1,              # 7–10 (transport, industrial areas, diffuse urban)
        3, 3, 3, 3, 3, 3, 3,     # 11–17 agriculture
        4,                      # 18 grassland
        3, 3, 3, 3,              # 19–22 agriculture
        5, 5, 5,                # 23–25 forests
        6, 6, 6, 6,             # 26–29 natural vegetation
        10, 10, 10, 10, 10,     # 30–34 rock, mineral, etc.
        7, 7,                  # 35–36 wetlands
        9, 9, 9,               # 37–39 coastline
        8, 8,                  # 40–41 water bodies
        10, 10, 10             # 42–44 sparsely vegetated areas / snow
    ])

    # Prepare a LUT array of size 256 to support any uint8 value
    lut = np.zeros(256, dtype=np.uint8)
    lut[1:45] = mapping_10_classes  # valid indices range from 1 to 44

    # Replace invalid values (-128 or 0) with 0
    array_clean = np.where((array >= 1) & (array <= 44), array, 0)
    reclassified = lut[array_clean]

    return reclassified

def load_shp_from_dir(subpath):
    gdfs = []
    # Iterate over all .shp files
    shp_files = chain(
    subpath.glob("*.shp"),
    subpath.glob("*.SHP")
    )

    for shp_file in shp_files:
        try:
            gdf = gpd.read_file(shp_file)
            gdfs.append(gdf)
        except Exception as e:
            print(f"Error while reading {shp_file.name}: {e}")

    # Final concatenation
    if gdfs:
        gdf_concat = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    else:
        print("No shapefile found or all files failed to read.")

    return gdf_concat

def raster_corine(geo, dir_output, file_subpath, tifFile, tifFile_high):
    """
    For each TIFF in ``dir_france``:
      - mask according to the ``geo`` geometry
      - reproject to the given resolution
      - convert each pixel to an RGB color (i, i, i)
      - export the result as a color GIF
    """
    dir_output = Path(dir_output)
    dir_output.mkdir(parents=True, exist_ok=True)
    
    unodes = np.unique(tifFile[~np.isnan(tifFile)])

    bands = np.arange(0, 10)

    res = np.full((np.max(bands) + 1, tifFile.shape[0], tifFile.shape[1]), fill_value=np.nan)
    res2 = np.full((tifFile.shape[0], tifFile.shape[1]), fill_value=np.nan)
    
    height, width = tifFile_high.shape
    
    with rasterio.open(file_subpath) as src:
        src_crs = src.crs
        target_crs = "EPSG:4326"  # Geographic CRS (degrees)

        # Reproject the geometry into the appropriate CRS
        if geo.crs != src_crs:
            geo_proj = geo.to_crs(src_crs)
        else:
            geo_proj = geo

        polygons = unary_union(geo_proj.geometry)
        # Spatial masking
        out_image, out_transform = mask(src, [polygons], crop=True, nodata=0)

        # Compute the target resolution
        dst_transform, width, height = calculate_default_transform(
            src.crs, target_crs,
            out_image.shape[1], out_image.shape[0],
            *polygons.bounds,
            dst_width=width, dst_height=height,
            #resolution=(reslon, reslat)  # in degrees
        )

        # Reprojected destination (expecting 1 band)
        dst_array = np.empty((1, height, width), dtype=np.float32)

        reproject(
            source=out_image,  # assume a single band representing the classes
            destination=dst_array[0],
            src_transform=out_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest
        )

        # Reclassify 1–44 → 1–10
        classified = reclass_corine_by_index(dst_array[0].astype(np.uint8))

        for node in unodes:
            if node not in tifFile_high:
                continue
            mask1 = tifFile == node
            mask2 = tifFile_high == node
            if True not in np.unique(mask2):
                continue
            for band in bands:
                res[band, mask1] = (np.argwhere(classified[mask2] == band).shape[0] / classified[mask2].shape[0]) * 100
            try:
                if res[:, mask1].shape[1] == 1:
                    res2[mask1] = np.nanargmax(res[:, mask1])
                else:
                    res2[mask1] = np.nanargmax(res[:, mask1][:,0])
            except:
                continue

    res[:, np.isnan(tifFile)] = np.nan
    res2[np.isnan(tifFile)] = np.nan
    res3[:, np.isnan(tifFile)] = np.nan
    
    outputName = 'corine.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res,f)

    outputName = 'corine_landcover.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res2,f)

def raster_route(dir_output, tifFile, tifFile_high, reslon, reslat, file_path, france):
    
    res = np.empty(tifFile.shape)
    res2 = np.copy(res)
    res3 = np.copy(res2)

    unodes = np.unique(tifFile[~np.isnan(tifFile)])

    bands = [0]

    osmnx = load_shp_from_dir(file_path)
    osmnx = osmnx[osmnx['NB_VOIES'].isin(['2 voies larges', '3 voies', '4 voies', 'Plus de 4 voies'])].copy()

    osmnx['label'] = 1  # useful only for subsequent rasterization
    print(f"Lignes routières filtrées : {osmnx.shape}")
    
    # Harmonize the CRS
    if osmnx.crs != france.crs:
        france = france.to_crs(osmnx.crs)

    # -- SPATIAL JOIN: mark polygons that intersect at least one road --
    france = france.copy()
    france['label'] = 0  # initialization

    # Use sjoin to retrieve intersections
    intersections = gpd.sjoin(france, osmnx, how='left', predicate='intersects')

    # Retrieve the polygon indices that intersect lines
    matched_idx = intersections[~intersections.index_right.isna()].index.unique()

    # Set their label to 1
    france.loc[matched_idx, 'label'] = 1
    
    france = france.to_crs("EPSG:4326")

    print(f"Polygones labellisés : {france['label'].sum()} sur {len(france)}")

    # Rasterize the polygons now labeled
    image = rasterisation(france, reslat, reslon, 'label', dir_output=dir_output, name='route')

    for node in unodes:
        if node not in tifFile_high:
            continue
        mask1 = tifFile == node
        mask2 = tifFile_high == node
        if True not in np.unique(mask2):
            continue
        for band in bands:
            res[band, mask1] = (np.argwhere(image[mask2] == band).shape[0] / image[mask2].shape[0]) * 100
        try:
            if res[:, mask1].shape[1] == 1:
                res2[mask1] = np.nanargmax(res[:, mask1])
            else:
                res2[mask1] = np.nanargmax(res[:, mask1][:,0])
        except:
            continue
        
    res[:, np.isnan(tifFile)] = np.nan
    res2[np.isnan(tifFile)] = np.nan
    res3[:, np.isnan(tifFile)] = np.nan
    
    outputName = 'route.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res,f)

    outputName = 'route_landcover.pkl'
    f = open(dir_output / outputName,"wb")
    pickle.dump(res2,f)

def read_object(filename: str, path : Path):
    if not (path / filename).is_file():
        print(f'{path / filename} not found')
        return None
    return pickle.load(open(path / filename, 'rb'))


############################### Raster loaders #################################

def _expand_static(data: np.ndarray, n_dates: int) -> np.ndarray:
    """Expand a static raster to have a date dimension."""
    if data.ndim == 2:
        return np.repeat(data[..., np.newaxis], n_dates, axis=2)
    if data.ndim == 3:
        return np.repeat(data[..., np.newaxis], n_dates, axis=3)
    return data

def load_rasterise_meteo(dir_raster: Path, dates: list, lat, lon) -> xr.Dataset:
    """Load meteo rasters produced by ``rasterise_meteo_data`` into an xarray."""
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
    data_vars = {}
    
    for var in cems_variables:
        if var == 'daily_severity_rating':
            var = 'dailySeverityRating'
        file = f'{var}raw.pkl'
        with open(dir_raster / file, "rb") as f:
            values = pickle.load(f)
        data_vars[var] = (("latitude", "longitude", "date"), values)

    coords = {"latitude": lat, "longitude": lon, "date": dates}
    return xr.Dataset(data_vars, coords=coords)

def load_raster_cosia(dir_raster: Path, dates: list, lat, lon) -> xr.Dataset:
    """Load COSIA rasters and broadcast them on the date dimension."""
    cosia = pickle.load(open(dir_raster / "cosia.pkl", "rb"))
    cosia_landcover = pickle.load(open(dir_raster / "cosia_landcover.pkl", "rb"))
    cosia_influence = pickle.load(open(dir_raster / "cosia_influence.pkl", "rb"))

    cosia = _expand_static(cosia, len(dates))
    cosia_landcover = _expand_static(cosia_landcover, len(dates))
    #cosia_influence = _expand_static(cosia_influence, len(dates))

    bands = valeurs_cosia_couverture.keys()

    data_vars = {
        "cosia_landcover": (("latitude", "longitude", "date"), cosia_landcover),
        #"cosia_influence": (("band", "latitude", "longitude", "date"), cosia_influence),
    }

    for band in bands:
        data_vars[band] = (("latitude", "longitude", "date"), cosia[valeurs_cosia_couverture[band]])

    coords = {"latitude": lat, "longitude": lon, "date": dates}
    return xr.Dataset(data_vars, coords=coords)

def load_raster_corine(dir_raster: Path, dates: list, lat, lon) -> xr.Dataset:
    """Load CORINE rasters into an xarray structure."""
    corine = pickle.load(open(dir_raster / "corine.pkl", "rb"))
    corine_land = pickle.load(open(dir_raster / "corine_landcover.pkl", "rb"))
    corine_inf = pickle.load(open(dir_raster / "corine_influence.pkl", "rb"))

    corine = _expand_static(corine, len(dates))
    corine_land = _expand_static(corine_land, len(dates))
    corine_inf = _expand_static(corine_inf, len(dates))
    
    bands = [
    'Corine_Other',
    'Corine_urban',
    'Corine_transport',
    'Corine_agricultural',
    'Corine_grass',
    'Corine_forest',
    'Corine_vegetation',
    'Corine_moisture',
    'Corine_water',
    'Corine_littoral',
    'Corine_rock'
    ]

    data_vars = {
        "corine_landcover": (("latitude", "longitude", "date"), corine_land),
        #"cosia_influence": (("band", "latitude", "longitude", "date"), cosia_influence),
    }

    for i, band in enumerate(bands):
        data_vars[band] = (("latitude", "longitude", "date"), corine[i])

    coords = { "latitude": lat, "longitude": lon, "date": dates}
    return xr.Dataset(data_vars, coords=coords)

def load_raster_elevation(dir_raster: Path, dates: list, lat, lon) -> xr.Dataset:
    """Load elevation raster and expand it across dates."""
    elevation = pickle.load(open(dir_raster / "elevation.pkl", "rb"))
    elevation = _expand_static(elevation, len(dates))

    data_vars = {
        "elevation": (("latitude", "longitude", "date"), elevation),
    }

    coords = {"latitude": lat, "longitude": lon, "date": dates}
    return xr.Dataset(data_vars, coords=coords)


def load_raster_population(dir_raster: Path, dates: list, lat, lon) -> xr.Dataset:
    """Load population raster and expand it across dates."""
    population = pickle.load(open(dir_raster / "population.pkl", "rb"))
    population = _expand_static(population, len(dates))

    data_vars = {"population": (("latitude", "longitude", "date"), population)}

    coords = {"latitude": lat, "longitude": lon, "date": dates}
    return xr.Dataset(data_vars, coords=coords)
    
def load_raster_sat(dir_raster: Path, dates: list, lat, lon) -> xr.Dataset:
    """Load Sentinel satellite rasters."""
    sentinel = pickle.load(open(dir_raster / "sentinel.pkl", "rb"))

    coords = {"latitude": lat, "longitude": lon, "date": dates}
    data_vars = {}
    for i, var in enumerate(['NDVI', 'NDMI', 'NDBI', 'NDSI', 'NDWI']):
        data_vars[var] = (("latitude", "longitude", "date"), sentinel[i]
    )
    return xr.Dataset(data_vars, coords=coords)

def load_raster_argile(dir_raster: Path, dates: list, lat, lon) -> xr.Dataset:
    """Load argile raster and expand across dates."""
    argile = pickle.load(open(dir_raster / "argile.pkl", "rb"))
    argile = _expand_static(argile, len(dates))

    data_vars = {"argile": (("latitude", "longitude", "date"), argile)}

    coords = {"latitude": lat, "longitude": lon, "date": dates}
    return xr.Dataset(data_vars, coords=coords)

def load_raster_foret(dir_raster: Path, dates: list, lat, lon) -> xr.Dataset:
    """Load forest rasters and broadcast them on the date dimension."""
    foret = pickle.load(open(dir_raster / "foret.pkl", "rb"))
    foret_land = pickle.load(open(dir_raster / "foret_landcover.pkl", "rb"))
    foret_inf = pickle.load(open(dir_raster / "foret_influence.pkl", "rb"))

    foret = _expand_static(foret, len(dates))
    foret_land = _expand_static(foret_land, len(dates))
    foret_inf = _expand_static(foret_inf, len(dates))

    bands = valeurs_foret_attribut.keys()

    data_vars = {
        "forest_landcover": (("latitude", "longitude", "date"), foret_land),
        #"cosia_influence": (("band", "latitude", "longitude", "date"), cosia_influence),
    }

    for band in bands:
        data_vars[band] = (("latitude", "longitude", "date"), foret[valeurs_foret_attribut[band]])

    coords = {"latitude": lat, "longitude": lon, "date": dates}
    return xr.Dataset(data_vars, coords=coords)

def load_raster_bdroute(dir_raster: Path, dates: list, lat, lon) -> xr.Dataset:
    """Load forest rasters and broadcast them on the date dimension."""
    route = pickle.load(open(dir_raster / "route.pkl", "rb"))
    route_land = pickle.load(open(dir_raster / "route_landcover.pkl", "rb"))
    route_inf = pickle.load(open(dir_raster / "route_influence.pkl", "rb"))

    route = _expand_static(route, len(dates))
    route_land = _expand_static(route_land, len(dates))
    route_inf = _expand_static(route_inf, len(dates))

    print(route.shape)

    bands = ['NoRoad', 'Road']

    data_vars = {
        "route_landcover": (("latitude", "longitude", "date"), route_land),
        #"cosia_influence": (("band", "latitude", "longitude", "date"), cosia_influence),
    }

    for i, band in enumerate(bands):
        data_vars[band] = (("latitude", "longitude", "date"), route[i])

    coords = {"latitude": lat, "longitude": lon, "date": dates}
    return xr.Dataset(data_vars, coords=coords)

def concat_xarrays(dir_raster: Path, dates: list) -> xr.Dataset:
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

    latitude = read_object('latitude.pkl', dir_raster)
    longitude = read_object('longitude.pkl', dir_raster)

    latitude = latitude[:, 0]
    longitude = longitude[0]

    loaders = [
        (load_rasterise_meteo),
        (load_raster_corine),
        (load_raster_elevation),
        (load_raster_population),
        (load_raster_sat),
        (load_raster_argile),
        (load_raster_foret),
        (load_raster_bdroute),
    ]

    datasets = []
    for loader in loaders:
        datasets.append(loader(dir_raster, dates, latitude, longitude))
        
    if not datasets:
        raise ValueError("No raster data found in the provided directory")

    f = open(dir_raster / f'datacube.pkl',"wb")
    pickle.dump(xr.merge(datasets),f)
