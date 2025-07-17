from ast import arg
import datetime as dt
import numpy as np
from category_encoders import CatBoostEncoder
import pickle
from pathlib import Path
import vacances_scolaires_france
import jours_feries_france
import pandas as pd
import convertdate
from skimage import img_as_float
from skimage import transform
from dico_departements import name2int

def check_and_create_path(path: Path):
    """
    Creer un dossier s'il n'existe pas
    """
    path_way = path.parent if path.is_file() else path

    path_way.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.touch()

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

def read_object(filename: str, path : Path):
    if not (path / filename).is_file():
        print(f'{path / filename} not found')
        return None
    return pickle.load(open(path / filename, 'rb'))

def save_object(obj, filename: str, path : Path):
    check_and_create_path(path)
    with open(path / filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def est_bissextile(annee):
    return annee % 4 == 0 and (annee % 100 != 0 or annee % 400 == 0)

def ajuster_jour_annee(date, dayoyyear):
    if not est_bissextile(date.year) and date > pd.Timestamp(date.year, 2, 28):
        return dayoyyear + 1
    else:
        return dayoyyear
    
def pendant_couvrefeux(date):
    # Fonction testant si une date tombe dans une période de confinement
    if ((dt.datetime(2020, 12, 15) <= date <= dt.datetime(2021, 1, 2)) 
        and (date.hour >= 20 or date.hour <= 6)):
        return 1
    elif ((dt.datetime(2021, 1, 2) <= date <= dt.datetime(2021, 3, 20))
        and (date.hour >= 18 or date.hour <= 6)):
            return 1
    elif ((dt.datetime(2021, 3, 20) <= date <= dt.datetime(2021, 5, 19))
        and (date.hour >= 19 or date.hour <= 6)):
            return 1
    elif ((dt.datetime(2021, 5, 19) <= date <= dt.datetime(2021, 6, 9))
        and (date.hour >= 21 or date.hour <= 6)):
            return 1
    elif ((dt.datetime(2021, 6, 9) <= date <= dt.datetime(2021, 6, 30))
        and (date.hour >= 23 or date.hour <= 6)):
            return 1
    return 0

def resize_no_dim(input_image, height, width):
    """
    Resize the input_image into heigh, with, dim
    """
    img = img_as_float(input_image)
    img = transform.resize(img, (height, width), mode='constant', order=0,
                 preserve_range=True, anti_aliasing=True)
    return np.asarray(img)

def get_academic_zone(name, date):
    dict_zones = {
        'Aix-Marseille': ('B', 'B'),
        'Amiens': ('B', 'B'),
        'Besançon': ('B', 'A'),
        'Bordeaux': ('C', 'A'),
        'Caen': ('A', 'B'),
        'Clermont-Ferrand': ('A', 'A'),
        'Créteil': ('C', 'C'),
        'Dijon': ('B', 'A'),
        'Grenoble': ('A', 'A'),
        'Lille': ('B', 'B'),
        'Limoges': ('B', 'A'),
        'Lyon': ('A', 'A'),
        'Montpellier': ('A', 'C'),
        'Nancy-Metz': ('A', 'B'),
        'Nantes': ('A', 'B'),
        'Nice': ('B', 'B'),
        'Orléans-Tours': ('B', 'B'),
        'Paris': ('C', 'C'),
        'Poitiers': ('B', 'A'),
        'Reims': ('B', 'B'),
        'Rennes': ('A', 'B'),
        'Rouen': ('B', 'B'),
        'Strasbourg': ('B', 'B'),
        'Toulouse': ('A', 'C'),
        'Versailles': ('C', 'C'),
        'Guadeloupe': ('C', 'C'),
        'Martinique': ('C', 'C'),
        'Guyane': ('C', 'C'),
        'La Réunion': ('C', 'C'),
        'Mayotte': ('C', 'C'),
        'Normandie': ('A', 'B'),  # Choix arbitraire de zone pour l'académie Normandie après 2020
    }

    if name == 'Normandie':
        if date < dt.datetime(2020, 1, 1):
            if date < dt.datetime(2016, 1, 1):
                # Avant 2016, on prend en compte l'ancienne académie de Caen ou Rouen
                return 'A'  # Zone de Caen
            return 'B'  # Zone de Rouen après 2016
        else:
            return dict_zones[name][1]  # Zone après la fusion en 2020
        
    # Cas général pour les autres académies
    if date < dt.datetime(2016, 1, 1):
        return dict_zones[name][0]
    return dict_zones[name][1]

def encode(path_to_target, trainDates, expe, train_departements, dir_output):

    jours_feries = sum([list(jours_feries_france.JoursFeries.for_year(k).values()) for k in range(2017,2023)],[]) # French Jours fériés, used in features_*.py 
    veille_jours_feries = sum([[l-dt.timedelta(days=1) for l \
                in jours_feries_france.JoursFeries.for_year(k).values()] for k in range(2017,2023)],[]) # French Veille Jours fériés, used in features_*.py 
    vacances_scolaire = vacances_scolaires_france.SchoolHolidayDates() # French Holidays used in features_*.py

    ACADEMIES = {
    '1': 'Lyon',
    '2': 'Amiens',
    '3': 'Clermont-Ferrand',
    '4': 'Aix-Marseille',
    '5': 'Aix-Marseille',
    '6': 'Nice',
    '7': 'Grenoble',
    '8': 'Reims',
    '9': 'Toulouse',
    '10': 'Reims',
    '11': 'Montpellier',
    '12': 'Toulouse',
    '13': 'Aix-Marseille',
    '14': 'Caen',
    '15': 'Clermont-Ferrand',
    '16': 'Poitiers',
    '17': 'Poitiers',
    '18': 'Orléans-Tours',
    '19': 'Limoges',
    '21': 'Dijon',
    '22': 'Rennes',
    '23': 'Limoges',
    '24': 'Bordeaux',
    '25': 'Besançon',
    '26': 'Grenoble',
    '27': 'Normandie',
    '28': 'Orléans-Tours',
    '29': 'Rennes',
    '30': 'Montpellier',
    '31': 'Toulouse',
    '32': 'Toulouse',
    '33': 'Bordeaux',
    '34': 'Montpellier',
    '35': 'Rennes',
    '36': 'Orléans-Tours',
    '37': 'Orléans-Tours',
    '38': 'Grenoble',
    '39': 'Besançon',
    '40': 'Bordeaux',
    '41': 'Orléans-Tours',
    '42': 'Lyon',
    '43': 'Clermont-Ferrand',
    '44': 'Nantes',
    '45': 'Orléans-Tours',
    '46': 'Toulouse',
    '47': 'Bordeaux',
    '48': 'Montpellier',
    '49': 'Nantes',
    '50': 'Normandie',
    '51': 'Reims',
    '52': 'Reims',
    '53': 'Nantes',
    '54': 'Nancy-Metz',
    '55': 'Nancy-Metz',
    '56': 'Rennes',
    '57': 'Nancy-Metz',
    '58': 'Dijon',
    '59': 'Lille',
    '60': 'Amiens',
    '61': 'Normandie',
    '62': 'Lille',
    '63': 'Clermont-Ferrand',
    '64': 'Bordeaux',
    '65': 'Toulouse',
    '66': 'Montpellier',
    '67': 'Strasbourg',
    '68': 'Strasbourg',
    '69': 'Lyon',
    '70': 'Besançon',
    '71': 'Dijon',
    '72': 'Nantes',
    '73': 'Grenoble',
    '74': 'Grenoble',
    '75': 'Paris',
    '76': 'Normandie',
    '77': 'Créteil',
    '78': 'Versailles',
    '79': 'Poitiers',
    '80': 'Amiens',
    '81': 'Toulouse',
    '82': 'Toulouse',
    '83': 'Nice',
    '84': 'Aix-Marseille',
    '85': 'Nantes',
    '86': 'Poitiers',
    '87': 'Limoges',
    '88': 'Nancy-Metz',
    '89': 'Dijon',
    '90': 'Besançon',
    '91': 'Versailles',
    '92': 'Versailles',
    '93': 'Créteil',
    '94': 'Créteil',
    '95': 'Versailles',
    '971': 'Guadeloupe',
    '972': 'Martinique',
    '973': 'Guyane',
    '974': 'La Réunion',
    '976': 'Mayotte',
    }

    allDates = find_dates_between('2017-06-12', '2023-12-31')
    print(f'Create encoder for categorical features using {train_departements}, at expe {expe}')
    stop_calendar = 11
    
    trainDate = np.asarray([allDates.index(date) for date in trainDates])
    foret = []
    cosia = []
    corine = []
    route = []
    gt = []
    landcover = []
    argile_value = []
    temporalValues = []
    spatialValues = []
    ids_value = []
    cluster_value = []
    osmnx = []
    calendar_array = [[] for j in range(stop_calendar)]
    geo_array = []

    for dep in train_departements:

        dir_data = 'path_to_raster' / dep

        tar = read_object(dep+'binScale0.pkl', path_to_target)
        if tar is None:
            continue
        tar = tar[:,:,trainDate]
        gt += list(tar[~np.isnan(tar)])

        temporalValues.append(np.nansum(tar.reshape(-1, tar.shape[2]), axis=0))
        spatialValues += list(np.nansum(tar, axis=2)[~np.isnan(tar[:,:,0])].reshape(-1))

        ##################### Forest #################
        fore = read_object('foret_landcover.pkl', dir_data)
        if fore is not None:
            fore = resize_no_dim(fore, tar.shape[0], tar.shape[1])
            foret += list(fore[~np.isnan(tar[:,:,0])])
        
        ##################### Road #################
        os = read_object('osmnx_landcover.pkl', dir_data)
        if os is not None:
            os = resize_no_dim(os, tar.shape[0], tar.shape[1])
            osmnx += list(os[~np.isnan(tar[:,:,0])])

        ##################### Clay Soil #################
        argile = read_object('argile.pkl', dir_data)
        if argile is not None:
            argile = resize_no_dim(argile, tar.shape[0], tar.shape[1])
            argile_value += list(argile[~np.isnan(tar[:, :, 0])])

        ##################### Clustering #################
        dir_clustering = Path(dir_output / '..' /  'time_series_clustering')
        cluster_mask = read_object(f'time_series_clustering.pkl', dir_clustering)
        if cluster_mask is not None:
            cluster_value += list(cluster_mask[~np.isnan(tar[:, :, 0])])

        ##################### Landcover #################
        cosia_image = read_object('cosia_landcover.pkl', dir_data)
        if cosia_image is not None:
            cosia_image = resize_no_dim(cosia_image, tar.shape[0], tar.shape[1])
            cosia += list(cosia_image[~np.isnan(tar[:, :, 0])])
        
        calendar = np.empty((tar.shape[2], stop_calendar))

        ######################################## Calendar variables ##################################

        for i, date in enumerate(allDates):
            
            if i not in trainDate:
                break
            
            ddate = dt.datetime.strptime(date, '%Y-%m-%d')
            calendar[i, 0] = int(date.split('-')[1]) # month
            calendar[i, 1] = ajuster_jour_annee(ddate, ddate.timetuple().tm_yday) # dayofyear
            calendar[i, 2] = ddate.weekday() # dayofweek
            calendar[i, 3] = ddate.weekday() >= 5 # isweekend
            calendar[i, 4] = pendant_couvrefeux(ddate) # couvrefeux
            
            calendar[i, 5] = 1 if (
                dt.datetime(2020, 3, 17, 12) <= ddate <= dt.datetime(2020, 5, 11)
                or dt.datetime(2020, 10, 30) <= ddate <= dt.datetime(2020, 12, 15)
            ) else 0
            
            calendar[i, 6] = 1 if convertdate.islamic.from_gregorian(ddate.year, ddate.month, ddate.day)[1] == 9 else 0 # ramadan
            calendar[i, 7] = 1 if ddate in jours_feries else 0 # bankHolidays
            calendar[i, 8] = 1 if ddate in veille_jours_feries else 0 # bankHolidaysEve
            calendar[i, 9] = 1 if vacances_scolaire.is_holiday_for_zone(ddate.date(), get_academic_zone(ACADEMIES[str(name2int[dep])], ddate)) else 0 # holidays
            calendar[i, 10] = (1 if vacances_scolaire.is_holiday_for_zone(ddate.date() + dt.timedelta(days=1), get_academic_zone(ACADEMIES[str(name2int[dep])], ddate)) else 0 ) \
                or (1 if vacances_scolaire.is_holiday_for_zone(ddate.date() - dt.timedelta(days=1), get_academic_zone(ACADEMIES[str(name2int[dep])], ddate)) else 0) # holidaysBorder
                
        for j in range(stop_calendar):
            calendar_array[j] += list(calendar[:, j])

        geo = np.empty((tar.shape[2], 1))
        geo[:, :] = name2int[dep]
        geo_array += list(geo)

    gt = np.asarray(gt)
    temporalValues = np.asarray(temporalValues)
    spatialValues = np.asarray(spatialValues)
    foret = np.asarray(foret)
    osmnx = np.asarray(osmnx)
    ids_value = np.asarray(ids_value)
    cluster_value = np.asarray(cluster_value)
    landcover = np.asarray(landcover)
    argile_value = np.asarray(argile_value)
    cosia = np.asarray(cosia)
    corine = np.asarray(corine)
    route = np.asarray(route)
    geo_array = np.asarray(geo_array)
    calendar_array = np.asarray(calendar_array)

    spatialValues = spatialValues.reshape(-1,1)
    foret = foret.reshape(-1,1)
    landcover = landcover.reshape(-1,1)
    calendar_array = np.moveaxis(calendar_array, 0, 1)
    osmnx = osmnx.reshape(-1,1)
    argile_value = argile_value.reshape(-1,1)
    geo_array = geo_array.reshape(-1,1)
    ids_value = ids_value.reshape(-1,1)
    cluster_value = cluster_value.reshape(-1,1)
    temporalValues = temporalValues.reshape(-1,1)
    cosia = cosia.reshape(-1,1)
    corine = corine.reshape(-1,1)
    route = route.reshape(-1,1)

    # Calendar
    encoder = CatBoostEncoder(cols=np.arange(0, stop_calendar))
    encoder.fit(calendar_array, temporalValues)
    save_object(encoder, f'encoder_calendar_{expe}.pkl', dir_output)

    if landcover.shape == spatialValues.shape:
        # Landcover
        encoder = CatBoostEncoder(cols=np.arange(0, 1))
        encoder.fit(landcover, spatialValues)
        save_object(encoder, f'encoder_landcover_{expe}.pkl', dir_output)

    if foret.shape == spatialValues.shape:
        # Foret
        encoder = CatBoostEncoder(cols=np.arange(0, 1))
        encoder.fit(foret, spatialValues)
        save_object(encoder, f'encoder_foret_{expe}.pkl', dir_output)

    if osmnx.shape == spatialValues.shape:
        # OSMNX
        encoder = CatBoostEncoder(cols=np.arange(0, 1))
        encoder.fit(osmnx, spatialValues)
        save_object(encoder, f'encoder_osmnx_{expe}.pkl', dir_output)

    if argile_value.shape == spatialValues.shape:
        # OSMNX
        encoder = CatBoostEncoder(cols=np.arange(0, 1))
        encoder.fit(argile_value, spatialValues)
        save_object(encoder, f'encoder_argile_{expe}.pkl', dir_output)

    if cluster_value.shape == spatialValues.shape:
        encoder = CatBoostEncoder(cols=np.arange(0, 1))
        encoder.fit(cluster_value, spatialValues)
        save_object(encoder, f'encoder_cluster.pkl', dir_output)

    if cosia.shape == spatialValues.shape:
        encoder = CatBoostEncoder(cols=np.arange(0, 1))
        encoder.fit(cosia, spatialValues)
        save_object(encoder, f'encoder_cosia_{expe}.pkl', dir_output)
    
    # Geo
    encoder = CatBoostEncoder(cols=np.arange(0, 1))
    encoder.fit(geo_array, temporalValues)
    save_object(encoder, f'encoder_geo_{expe}.pkl', dir_output)
