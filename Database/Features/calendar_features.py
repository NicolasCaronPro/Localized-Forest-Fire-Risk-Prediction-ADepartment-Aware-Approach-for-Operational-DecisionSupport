"""
This code defines a set of utilities and a function to generate calendar-based features for a machine learning model using French context data. Here's a high-level explanation:

    Holidays and lockdowns: It loads French public holidays, holiday eves, and school vacations (2017–2023). It also defines specific curfew periods and COVID lockdown periods in France.

    Date adjustments: The ajuster_jour_annee function accounts for leap years when computing the day of the year.

    Academic zones: The get_academic_zone function maps each French education region to its vacation calendar zone (A, B, or C), considering historical changes like the Normandie merger.

    Main function get_calendar_features(...):

        For each date and subnode (geographical unit), it computes:

            Month, day of year, weekday, weekend indicator

            Whether it's during curfew, lockdown, or Ramadan

            Whether it's a public holiday, holiday eve, or school holiday

        It encodes these calendar variables and computes aggregated features: mean, max, min, and sum over the calendar variables.

Essentially, the function builds a feature matrix X with rich time-related data to help the model understand seasonality, public behavior, and government policies.
"""



import vacances_scolaires_france
import jours_feries_france
import convertdate
from .dico_departements.py import *

jours_feries = sum([list(jours_feries_france.JoursFeries.for_year(k).values()) for k in range(2017,2023)],[]) # French Jours fériés, used in features_*.py 
veille_jours_feries = sum([[l-dt.timedelta(days=1) for l \
            in jours_feries_france.JoursFeries.for_year(k).values()] for k in range(2017,2023)],[]) # French Veille Jours fériés, used in features_*.py 
vacances_scolaire = vacances_scolaires_france.SchoolHolidayDates() # French Holidays used in features_*.py

def ajuster_jour_annee(date, dayoyyear):
    if not est_bissextile(date.year) and date > pd.Timestamp(date.year, 2, 28):
        # Ajuster le jour de l'année pour les années non bissextiles après le 28 février
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

def get_calendar_features(subnode, allDates):
  """
  Create a numpy array with calendar features (day of year, day of week, holiday...). 
  -subnode is a numpy array of shape (X, 6) : ['graph_id', 'id', 'longitude', 'latitude', 'departement', 'date']. Graph_id and id are node id (=department in the article), longitude and latitude define location,
  department is the code department (1 to 95), date is the date's index in allDates (0 = 2017-06-12)
  -allDates is the list of all dates used in the database (2017-06-12 to 2023-12-31) in the article.
  """
  ids_columns =  ['graph_id', 'id', 'longitude', 'latitude', 'departement', 'date']
  calendar_variables = ['month', 'dayofyear', 'dayofweek', 'isweekend', 'couvrefeux', 'confinemenent',
                        'ramadan', 'bankHolidays', 'bankHolidaysEve', 'holidays', 'holidaysBorder',
                        'calendar_mean', 'calendar_min', 'calendar_max', 'calendar_sum']
  band = calendar_variables[0]
  features_name = calendar_variables
  size_calendar = len(calendar_variables)
  X = np.zeros((subnode.shape[0], len(calendar_variables)))
  for unDate, date in enumerate(allDates):
    ddate = dt.datetime.strptime(date, '%Y-%m-%d')
    index = np.argwhere((subnode[:, ids_columns.index('date')] == unDate))
    X[index, calendar_variables.index(band)] = int(date.split('-')[1]) # month
    X[index, calendar_variables.index(band) + 1] = ajuster_jour_annee(ddate, ddate.timetuple().tm_yday) # dayofyear
    X[index, calendar_variables.index(band) + 2] = ddate.weekday() # dayofweek
    X[index, calendar_variables.index(band) + 3] = ddate.weekday() >= 5 # isweekend
    X[index, calendar_variables.index(band) + 4] = pendant_couvrefeux(ddate) # couvrefeux
      
    X[index, calendar_variables.index(band) + 5] = if (
                dt.datetime(2020, 3, 17, 12) <= ddate <= dt.datetime(2020, 5, 11)
                or dt.datetime(2020, 10, 30) <= ddate <= dt.datetime(2020, 12, 15)
            ) else 0
    
    X[index, calendar_variables.index(band) + 6] = 1 if convertdate.islamic.from_gregorian(ddate.year, ddate.month, ddate.day)[1] == 9 else 0 # ramadan
    X[index, calendar_variables.index(band) + 7] = 1 if ddate in jours_feries else 0 # bankHolidays
    X[index, calendar_variables.index(band) + 8] = 1 if ddate in veille_jours_feries else 0 # bankHolidaysEve
    X[index, calendar_variables.index(band) + 9] = 1 if vacances_scolaire.is_holiday_for_zone(ddate.date(), get_academic_zone(ACADEMIES[str(name2int[departement])], ddate)) else 0 # holidays
    X[index, calendar_variables.index(band) + 10] = (1 if vacances_scolaire.is_holiday_for_zone(ddate.date() + dt.timedelta(days=1), get_academic_zone(ACADEMIES[str(name2int[departement])], ddate)) else 0 ) \
        or (1 if vacances_scolaire.is_holiday_for_zone(ddate.date() - dt.timedelta(days=1), get_academic_zone(ACADEMIES[str(name2int[departement])], ddate)) else 0) # holidaysBorder
    
    stop_calendar = 11
    
    X[index, features_name.index(band) : features_name.index(band) + stop_calendar] = \
            np.round(encoder_calendar.transform(np.moveaxis(X[index, features_name.index(band) : features_name.index(band) + stop_calendar], 1, 2).reshape(-1, stop_calendar)).values.reshape(-1, 1, stop_calendar), 3)
    
    for ir in range(stop_calendar, size_calendar):
        var_ir = calendar_variables[ir]
        if var_ir == 'calendar_mean':
            X[index, features_name.index(band) + ir] = round(np.mean(X[index, features_name.index(band) : features_name.index(band) + stop_calendar]), 3)
        elif var_ir == 'calendar_max':
            X[index, features_name.index(band) + ir] = round(np.max(X[index, features_name.index(band) : features_name.index(band) + stop_calendar]), 3)
        elif var_ir == 'calendar_min':
            X[index, features_name.index(band) + ir] = round(np.min(X[index, features_name.index(band) : features_name.index(band) + stop_calendar]), 3)
        elif var_ir == 'calendar_sum':
            X[index, features_name.index(band) + ir] = round(np.sum(X[index, features_name.index(band) : features_name.index(band) + stop_calendar]), 3)
        else:
            logger.info(f'Unknow operation {var_ir}')
            exit(1)
  return X
