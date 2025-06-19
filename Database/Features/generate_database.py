"""Entry point to build the full feature database for a department."""

from download import *
import logging
import argparse

############################### Logger #####################################

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")

# Handler pour afficher les logs dans le terminal
if (logger.hasHandlers()):
    logger.handlers.clear()
streamHandler = logging.StreamHandler(stream=sys.stdout)
streamHandler.setFormatter(logFormatter)
logger.addHandler(streamHandler)
################################ Database ####################################

class GenerateDatabase():
    def __init__(self, departement,
                 compute_meteostat_features, meteostatParams,
                 compute_temporal_features, outputParams,
                 compute_spatial_features, spatialParams,
                 compute_air_features, airParams,
                 compute_trafic_features, bouchonParams,
                 compute_vigicrues_features, vigicrueParams,
                 compute_nappes_features, nappesParams,
                 region, h3,
                 dir_raster
                 ):
        
        self.departement = departement

        self.compute_meteostat_features = compute_meteostat_features
        self.meteostatParams = meteostatParams

        self.compute_temporal_features = compute_temporal_features
        self.outputParams = outputParams

        self.compute_spatial_features = compute_spatial_features
        self.spatialParams = spatialParams

        self.compute_air_features = compute_air_features
        self.airParams = airParams

        self.compute_trafic_features = compute_trafic_features
        self.bouchonParams = bouchonParams

        self.compute_vigicrues_features = compute_vigicrues_features
        self.vigicrueParams = vigicrueParams

        self.compute_nappes_features = compute_nappes_features
        self.nappesParams = nappesParams

        self.dir_raster = dir_raster

        self.region = region

        self.h3 = h3

        #self.elevation, self.lons, self.lats = read_tif(self.spatialParams['dir'] / 'elevation' / self.spatialParams['elevation_file'])

    def compute_meteo_stat(self):
        """
        Download and build historical weather data using Meteostat.
        
        - Creates 'meteostat.csv' in the specified directory.
        - Adds a 'year' column extracted from the 'creneau' field.
        """
        logger.info('Compute compute_meteo_stat')
        check_and_create_path(self.meteostatParams['dir'])
        self.meteostat = construct_historical_meteo(self.meteostatParams['start'], self.meteostatParams['end'],
                                                    self.region, self.meteostatParams['dir'], self.departement)
        
        self.meteostat['year'] = self.meteostat['creneau'].apply(lambda x : x.split('-')[0])
        self.meteostat.to_csv(self.meteostatParams['dir'] / 'meteostat.csv', index=False)

    def compute_temporal(self):
        """
        Rasterize the temporal weather data onto the hexagonal H3 grid.

        Uses the low-resolution H3 cluster grid and Meteostat data to create
        time-series raster layers and saves them to the output directory.
        """
        logger.info('Compute compute_temporal')
        rasterise_meteo_data(self.clusterSum, self.h3tif, self.meteostat, self.h3tif.shape, self.dates, self.dir_raster)

    def add_spatial(self):
        """
        Download and rasterize spatial features for the specified department:
        
        - Satellite indices (NDVI, NDMI, etc.)
        - Land cover classification
        - Population distribution
        - OpenStreetMap network data (roads, buildings, etc.)
        - Elevation data
        - Forest areas (from BDFORET)

        Checks if data exists before downloading to avoid redundancy.
        """
        logger.info('Add spatial')
        code_dept = name2int[self.departement]
        if code_dept < 10:
            code_dept = f'0{code_dept}'
        else:
            code_dept = f'{code_dept}'

        raster_sat_from_france(self.h3tif, self.h3, self.dir_raster, Path('path_to_database') / 'csv' / 'france' / 'data' / 'GEE' / resolution, self.dates) # To download the original sat images, you will need a Google Earth Engine account

        if not (self.spatialParams['dir'] / 'cosia' / 'cosia.geojson').is_file():
            download_cosia(code_dept, self.region, self.spatialParams['dir'] / 'cosia')
        raster_cosia(self.h3tif, self.h3tif_high, self.dir_raster, self.resLon_high, self.resLat_high, self.spatialParams['dir'], self.region)

        if not (self.spatialParams['dir'] / 'population' / 'population.csv').is_file():
            download_population(Path('/path_to_database/csv/france/data/population'), self.region, self.spatialParams['dir'] / 'population')
        
        raster_population(self.h3tif, self.h3tif_high, self.dir_raster, self.resLon, self.resLat, self.spatialParams['dir'])

        if not (self.spatialParams['dir'] / 'osmnx' / 'osmnx.geojson').is_file():
            download_osnmx(self.region, self.spatialParams['dir'] / 'osmnx')
        raster_osmnx(self.h3tif, self.h3tif_high, self.dir_raster, self.resLon, self.resLat, self.spatialParams['dir'], self.departement)

        if not (self.spatialParams['dir'] / 'elevation' / 'elevation.csv').is_file():
            download_elevation(code_dept, self.region, self.spatialParams['dir'] / 'elevation')
        raster_elevation(self.h3tif, self.dir_raster, self.resLon, self.resLat, self.spatialParams['dir'], self.departement)
        
        if not (self.spatialParams['dir'] / 'BDFORET' / 'foret.geojson').is_file():
            download_foret(code_dept, self.departement, self.spatialParams['dir'])
        raster_foret(self.h3tif, self.h3tif_high, self.dir_raster, self.resLon_high, self.resLat_high, self.spatialParams['dir'], self.departement)

    def process(self, start, stop, resolution):
        """
        Main pipeline to generate the dataset for a department.
        
        Steps:
        1. Load or download Meteostat weather data (if enabled).
        2. Compute low and high resolution raster H3 grids.
        3. Generate temporal features (if enabled).
        4. Generate spatial features (if enabled).

        Parameters:
        - start (str): Start date in 'YYYY-MM-DD' format.
        - stop (str): End date in 'YYYY-MM-DD' format.
        - resolution (str): Raster resolution, e.g., '2x2', '1x1', etc.
        """
        logger.info(self.departement)
        
        if self.compute_meteostat_features:
            self.compute_meteo_stat()
        else:
            if (self.meteostatParams['dir'] / 'meteostat.csv').is_file():
                self.meteostat = pd.read_csv(self.meteostatParams['dir'] / 'meteostat.csv')

        self.h3['latitude'] = self.h3['geometry'].apply(lambda x : float(x.centroid.y))
        self.h3['longitude'] = self.h3['geometry'].apply(lambda x : float(x.centroid.x))
        self.h3['altitude'] = 0

        self.clusterSum = self.h3.copy(deep=True)
        self.clusterSum['cluster'] = self.clusterSum.index.values.astype(int)
        self.cluster = None
        sdate = start
        edate = stop
        self.dates = find_dates_between(sdate, edate)

        resolutions = {'2x2' : {'x' : 0.02875215641173088,'y' :  0.020721094073767096},
                '1x1' : {'x' : 0.01437607820586544,'y' : 0.010360547036883548},
                '0.5x0.5' : {'x' : 0.00718803910293272,'y' : 0.005180273518441774},
                '0.03x0.03' : {'x' : 0.0002694945852326214,'y' :  0.0002694945852352859}}
        
        #n_pixel_x = 0.016133099692723363
        #n_pixel_y = 0.016133099692723363

        n_pixel_x = resolutions[resolution]['x']
        n_pixel_y = resolutions[resolution]['y']
        
        self.resLon = n_pixel_x
        self.resLat = n_pixel_y
        self.h3tif = rasterisation(self.clusterSum, n_pixel_y, n_pixel_x, column='cluster', defval=np.nan, name=self.departement+'_low')
        logger.info(f'Low scale {self.h3tif.shape}')

        n_pixel_x = resolutions['0.03x0.03']['x']
        n_pixel_y = resolutions['0.03x0.03']['y']

        self.resLon_high = n_pixel_x
        self.resLat_high = n_pixel_y
        self.h3tif_high = rasterisation(self.clusterSum, n_pixel_y, n_pixel_x, column='cluster', defval=np.nan, name=self.departement+'_high')
        logger.info(f'High scale {self.h3tif_high.shape}')

        if self.compute_temporal_features:
            self.compute_temporal()

        if self.compute_spatial_features:
            self.add_spatial()
            
def launch(departement, resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, 
           compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop):
    """
    Launch the dataset generation process for a specific department.
    
    Initializes all feature parameters and runs the full processing pipeline.

    Parameters:
    - departement (str): Department identifier (e.g., 'departement-01-ain')
    - resolution (str): Spatial resolution for raster grids
    - compute_* (bool): Flags to control which feature sets to generate
    - start (str): Start date (format 'YYYY-MM-DD')
    - stop (str): End date (format 'YYYY-MM-DD')
    """
    dir_data_disk = Path('path_to_database') / 'csv' / departement / 'data' # Path to original data
    dir_data = dir_data_disk
    dir_raster =  Path('path_to_database') / 'csv' / departement / 'raster' / resolution # Path to ouput raster
    
    dir_meteostat = dir_data / 'meteostat'
    check_and_create_path(dir_raster)
    
    meteostatParams = {'start' : '2016-01-01',
                    'end' : stop,
                    'dir' : dir_meteostat}

    outputParams  = {'start' : start,
                    'end' : stop}
    
    spatialParams = {'dir_sat':  dir_data_disk,
                    'dir' : dir_data,
                    'sentBand': ['NDVI', 'NDMI', 'NDBI', 'NDSI', 'NDWI'],
                    'addsent': True,
                    'addland': True}
    
    airParams = {'dir' : dir_data,
                 'start' : start}

    bouchonParams = {'dir' : dir_data}

    vigicrueParams = {'dir' : dir_data,
                      'start' : start,
                        'end' : stop}
    
    nappesParams = {'dir' : dir_data,
                      'start' : start,
                        'end' : stop}
    
    if not (dir_data / 'geo/geo.geojson').is_file():
        download_region(departement, dir_data / 'geo')

    region_path = dir_data / 'geo/geo.geojson'
    region = gpd.read_file(region_path)
    
    if not (dir_data / 'spatial/hexagones.geojson').is_file():
        download_hexagones(Path('path_to_database/csv/france/data/geo'), region, dir_data / 'spatial', departement)

    h3 = gpd.read_file(dir_data / 'spatial/hexagones.geojson')

    database = GenerateDatabase(departement,
                    compute_meteostat_features, meteostatParams,
                    compute_temporal_features, outputParams,
                    compute_spatial_features, spatialParams,
                    compute_air_features, airParams,
                    compute_trafic_features, bouchonParams,
                    compute_vigicrues_features, vigicrueParams,
                    compute_nappes_features, nappesParams,
                    region, h3,
                    dir_raster)

    database.process(start, stop, resolution)

if __name__ == '__main__':
    RASTER = True
    parser = argparse.ArgumentParser(
        prog='Train',
        description='Create graph and database according to config.py and tained model',
    )
    parser.add_argument('-m', '--meteostat', type=str, help='Compute meteostat')
    parser.add_argument('-t', '--temporal', type=str, help='Temporal')
    parser.add_argument('-s', '--spatial', type=str, help='Spatial')
    parser.add_argument('-a', '--air', type=str, help='Air')
    parser.add_argument('-v', '--vigicrues', type=str, help='Vigicrues')
    parser.add_argument('-n', '--nappes', type=str, help='Nappes')
    parser.add_argument('-r', '--resolution', type=str, help='Resolution')
    
    args = parser.parse_args()

    # Input config
    compute_meteostat_features = args.meteostat == "True"
    compute_temporal_features = args.temporal == "True"
    compute_spatial_features = args.spatial == "True"
    compute_air_features = args.air == 'True'
    compute_trafic_features = False
    compute_nappes_features = args.nappes == 'True'
    compute_vigicrues_features = args.vigicrues == "True"
    resolution = args.resolution

    start = '2017-06-12'
    stop = '2024-06-29'
    
    ################## Ain ######################
    launch('departement-01-ain', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Aisne ######################
    launch('departement-02-aisne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Allier ######################
    launch('departement-03-allier', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Alpes-de-Haute-Provence ######################
    launch('departement-04-alpes-de-haute-provence', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Hautes-Alpes ######################
    launch('departement-05-hautes-alpes', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Alpes-Maritimes ######################
    launch('departement-06-alpes-maritimes', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Ardeche ######################
    launch('departement-07-ardeche', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Ardennes ######################
    launch('departement-08-ardennes', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Ariege ######################
    launch('departement-09-ariege', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Aube ######################
    launch('departement-10-aube', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Aude ######################
    launch('departement-11-aude', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Aveyron ######################
    launch('departement-12-aveyron', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Bouches-du-Rhone ######################
    launch('departement-13-bouches-du-rhone', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start,   stop)
    
    ################## Calvados ######################
    launch('departement-14-calvados', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Cantal ######################
    launch('departement-15-cantal', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Charente ######################
    launch('departement-16-charente', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Charente-Maritime ######################
    launch('departement-17-charente-maritime', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Cher ######################
    launch('departement-18-cher', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Correze ######################
    launch('departement-19-correze', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Cote-d-Or ######################
    launch('departement-21-cote-d-or', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Cotes-d-Armor ######################
    launch('departement-22-cotes-d-armor', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Creuse ######################
    launch('departement-23-creuse', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Dordogne ######################
    launch('departement-24-dordogne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Doubs ######################
    launch('departement-25-doubs', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Drome ######################
    launch('departement-26-drome', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Eure ######################
    launch('departement-27-eure', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Eure-et-Loir ######################
    launch('departement-28-eure-et-loir', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Finistere ######################
    launch('departement-29-finistere', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Corse-du-Sud ######################
    #launch('departement-2A-corse-du-sud', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Haute-Corse ######################
    #launch('departement-2B-haute-corse', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Gard ######################
    launch('departement-30-gard', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Haute-Garonne ######################
    launch('departement-31-haute-garonne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Gers ######################
    launch('departement-32-gers', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Gironde ######################
    launch('departement-33-gironde', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Herault ######################
    launch('departement-34-herault', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Ille-et-Vilaine ######################
    launch('departement-35-ille-et-vilaine', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Indre ######################
    launch('departement-36-indre', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Indre-et-Loire ######################
    launch('departement-37-indre-et-loire', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Isere ######################
    launch('departement-38-isere', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Jura ######################
    launch('departement-39-jura', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Landes ######################
    launch('departement-40-landes', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Loir-et-Cher ######################
    launch('departement-41-loir-et-cher', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Loire ######################
    launch('departement-42-loire', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Haute-Loire ######################
    launch('departement-43-haute-loire', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Loire-Atlantique ######################
    launch('departement-44-loire-atlantique', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Loiret ######################
    launch('departement-45-loiret', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Lot ######################
    launch('departement-46-lot', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Lot-et-Garonne ######################
    launch('departement-47-lot-et-garonne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Lozere ######################
    launch('departement-48-lozere', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Maine-et-Loire ######################
    launch('departement-49-maine-et-loire', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Manche ######################
    launch('departement-50-manche', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Marne ######################
    launch('departement-51-marne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Haute-Marne ######################
    launch('departement-52-haute-marne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Mayenne ######################
    launch('departement-53-mayenne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Meurthe-et-Moselle ######################
    launch('departement-54-meurthe-et-moselle', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Meuse ######################
    launch('departement-55-meuse', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Morbihan ######################
    launch('departement-56-morbihan', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Moselle ######################
    launch('departement-57-moselle', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Nievre ######################
    launch('departement-58-nievre', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Nord ######################
    launch('departement-59-nord', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Oise ######################
    launch('departement-60-oise', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Orne ######################
    launch('departement-61-orne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Pas-de-Calais ######################
    launch('departement-62-pas-de-calais', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Puy-de-Dome ######################
    launch('departement-63-puy-de-dome', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Pyrenees-Atlantiques ######################
    launch('departement-64-pyrenees-atlantiques', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Hautes-Pyrenees ######################
    launch('departement-65-hautes-pyrenees', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Pyrenees-Orientales ######################
    launch('departement-66-pyrenees-orientales', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Bas-Rhin ######################
    launch('departement-67-bas-rhin', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Haut-Rhin ######################
    launch('departement-68-haut-rhin', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Rhone ######################
    launch('departement-69-rhone', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Haute-Saone ######################
    launch('departement-70-haute-saone', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Saone-et-Loire ######################
    launch('departement-71-saone-et-loire', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Sarthe ######################
    launch('departement-72-sarthe', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Savoie ######################
    launch('departement-73-savoie', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Haute-Savoie ######################
    launch('departement-74-haute-savoie', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Paris ######################
    launch('departement-75-paris', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Seine-Maritime ######################
    launch('departement-76-seine-maritime', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Seine-et-Marne ######################
    launch('departement-77-seine-et-marne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Yvelines ######################
    launch('departement-78-yvelines', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Deux-Sevres ######################
    launch('departement-79-deux-sevres', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Somme ######################
    launch('departement-80-somme', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Tarn ######################
    launch('departement-81-tarn', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Tarn-et-Garonne ######################
    launch('departement-82-tarn-et-garonne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Var ######################
    launch('departement-83-var', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Vaucluse ######################
    launch('departement-84-vaucluse', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Vendee ######################
    launch('departement-85-vendee', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Vienne ######################
    launch('departement-86-vienne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Haute-Vienne ######################
    launch('departement-87-haute-vienne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Vosges ######################
    launch('departement-88-vosges', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Yonne ######################
    launch('departement-89-yonne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Territoire-de-Belfort ######################
    launch('departement-90-territoire-de-belfort', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Essonne ######################
    launch('departement-91-essonne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
    ################## Hauts-de-Seine ######################
    launch('departement-92-hauts-de-seine', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Seine-Saint-Denis ######################
    launch('departement-93-seine-saint-denis', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Val-de-Marne ######################
    launch('departement-94-val-de-marne', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)

    ################## Val-d-Oise ######################
    launch('departement-95-val-d-oise', resolution, compute_meteostat_features, compute_temporal_features, compute_spatial_features, compute_air_features, compute_trafic_features, compute_vigicrues_features, compute_nappes_features, start, stop)
    
