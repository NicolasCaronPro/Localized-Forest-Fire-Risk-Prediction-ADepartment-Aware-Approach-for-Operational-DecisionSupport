cems_variables = ['temp',
                  'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'prec24h',
                'dc', 'ffmc', 'dmc', 'nesterov', 'munger', 'kbdi',
                'isi', 'angstroem', 'bui', 'fwi', 'dailySeverityRating',
                'temp16',
                'dwpt16', 'rhum16', 'prcp16', 'wdir16', 'wspd16', 'prec24h16',
                'days_since_rain', 'sum_consecutive_rainfall', 'sum_rain_last_7_days',
                'sum_snow_last_7_days', 'snow24h', 'snow24h16',
                'precipitationIndexN3', 'precipitationIndexN5', 'precipitationIndexN7'
                ]

air_variables = ['O3', 'NO2', 'PM10', 'PM25']

# Encoder
sentinel_variables = ['NDVI', 'NDMI', 'NDBI', 'NDSI', 'NDWI']
landcover_variables = [
                      'foret_encoder',
                      'argile_encoder',
                      'cosia_encoder',
                        ]

cluster_encoder = ['cluster_encoder']

calendar_variables = ['month', 'dayofyear', 'dayofweek', 'isweekend', 'couvrefeux', 'confinemenent',
                    'ramadan', 'bankHolidays', 'bankHolidaysEve', 'holidays', 'holidaysBorder',
                    'calendar_mean', 'calendar_min', 'calendar_max', 'calendar_sum']

geo_variables = ['departement_encoder']
region_variables = ['region_class']

foret_variables = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']

cosia_variables = [
    'Other',
    'Building',
    'Bare soil',
    'Water surface',
    'Conifer',
    'Deciduous',
    'Shrubland',
    'Lawn',
    'Crop'
]

osmnx_variables = ['0', '1', '2', '3', '4', '5']

# Other
elevation_variables = ['elevation']
population_variabes = ['population']

foretint2str = {
    '0': 'PasDeforet',
    '1': 'Châtaignier',
    '2': 'Chênes décidus',
    '3': 'Chênes sempervirents',
    '4': 'Conifères',
    '5': 'Douglas',
    '6': 'Feuillus',
    '7': 'Hêtre',
    '8': 'Mélèze',
    '9': 'Mixtes',
    '10': 'NC',
    '11': 'NR',
    '12': 'Pin à crochets, pin cembro',
    '13': 'Pin autre',
    '14': 'Pin d\'Alep',
    '15': 'Pin laricio, pin noir',
    '16': 'Pin maritime',
    '17': 'Pin sylvestre',
    '18': 'Pins mélangés',
    '19': 'Peuplier',
    '20': 'Robinier',
    '21': 'Sapin, épicéa'
}

osmnxint2str = {
'0' : 'PasDeRoute',
'1':'motorway',
 '2': 'primary',
 '3': 'secondary',
 '4': 'tertiary', 
 '5': 'path'}