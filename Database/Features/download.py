"""Utilities to download and preprocess various geographic datasets."""

from shapely import unary_union, set_precision  # geometry utilities for OSMnx data
import pandas as pd  # dataframe manipulations
import osmnx as ox  # openstreetmap toolkit
import subprocess  # run command line processes
from itertools import chain  # flatten nested iterables
import re  # regex parsing
import geopandas as gpd  # geospatial dataframes
import os  # filesystem helpers
import py7zr  # handle 7z archives
import glob  # find files by pattern
import zipfile  # handle zip archives
from tools import *  # shared utilities across the project


def myround(x):
    """Round coordinates to 3 decimal places."""
    # x is a tuple of floats (lon, lat)
    return (round(x[0], 3), round(x[1], 3))  # return rounded tuple

def get_latitude(x):
    """Return the latitude (second element) from a (lon, lat) tuple."""
    # pick latitude at index 1
    return x[1]

def get_longitude(x):
    """Return the longitude (first element) from a (lon, lat) tuple."""
    # pick longitude at index 0
    return x[0]

def unzip_7z(file_path, destination_folder) -> None:
    """Extracts a .7z archive into the specified destination folder."""
    # ensure destination exists
    os.makedirs(destination_folder, exist_ok=True)
    # open the archive
    with py7zr.SevenZipFile(file_path, mode='r') as archive:
        # extract all contents
        archive.extractall(path=destination_folder)

def unzip_file(file_path, destination_folder) -> None:
    """Extracts a .zip archive into the specified destination folder."""
    # ensure destination exists
    os.makedirs(destination_folder, exist_ok=True)
    # open the archive
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        # extract all files
        zip_ref.extractall(destination_folder)

def download_population(path, geo, dir_output) -> None:
    """Filters and saves population data within a specified geographic boundary. https://www.kontur.io/datasets/population-dataset/"""
    check_and_create_path(dir_output)  # ensure output directory
    print('DONWLOADING POPULATION')  # user feedback
    if not (path / 'kontur_population_FR_20231101.gpkg').is_file():
        print('ERROR : Add population download')  # file missing notification
        exit(1)  # abort if dataset not present
    file = gpd.read_file(path / 'kontur_population_FR_20231101.gpkg')  # load data
    file.to_crs('EPSG:4326', inplace=True)  # unify CRS
    file['isdep'] = file['geometry'].apply(lambda x : geo.contains(x))  # filter on region
    geo_pop = file[file['isdep']]  # keep relevant rows
    geo_pop['latitude'] = geo_pop['geometry'].apply(lambda x : float(x.centroid.y))  # centroid lat
    geo_pop['longitude'] = geo_pop['geometry'].apply(lambda x : float(x.centroid.x))  # centroid lon
    geo_pop.to_file(dir_output / 'population.geojson')  # save GeoJSON
    geo_pop.to_csv(dir_output / 'population.csv', index=False)  # save CSV

def create_elevation_geojson(path, bounds, dir_output):
    """Processes elevation contour data and outputs a GeoJSON and CSV of average point altitudes."""
    path = path.as_posix()  # convert Path to string
    files = glob.glob(path+'/COURBE/**/*.shp', recursive=True)  # list all shapefiles
    gdf = []  # accumulate per-file data
    for f in files:  # iterate over shapefiles
        try:
            gdft = gpd.read_file(f).to_crs(crs=4326)  # load and project
        except:
            continue  # skip unreadable files
        if "ALTITUDE" not in gdft.columns:
            continue  # ignore files without altitude
        gdft['points'] = gdft.apply(lambda x: [y for y in x['geometry'].coords], axis=1)  # expand coords
        gdft['altitude'] = gdft.apply(lambda x: [x['ALTITUDE'] for y in x['geometry'].coords], axis=1)  # duplicate altitude
        points = pd.Series(list(chain.from_iterable(gdft['points'])))  # flatten points
        altitude = pd.Series(list(chain.from_iterable(gdft['altitude'])))  # flatten altitude
        gdft = pd.DataFrame({'altitude': altitude, 'geometry': points})  # new dataframe
        gdft['points'] = gdft['geometry'].apply(myround)  # round coordinates
        gb = gdft.groupby('points')  # group by unique point
        mean = gb['altitude'].mean().reset_index()  # mean altitude per point
        mean['latitude'] = mean['points'].apply(get_latitude)  # extract lat
        mean['longitude'] = mean['points'].apply(get_longitude)  # extract lon
        mean = gpd.GeoDataFrame(mean, geometry=gpd.points_from_xy(mean.longitude, mean.latitude))  # to GeoDataFrame
        mean = mean[(mean['latitude'] >= bounds['miny'].values[0]) & (mean['latitude'] <= bounds['maxy'].values[0]) & \
                    (mean['longitude'] >= bounds['minx'].values[0]) & (mean['longitude'] <= bounds['maxx'].values[0])]  # clip to region
        if mean.empty:
            continue  # ignore empty results
        gdf.append(mean)  # accumulate

    gdf = pd.concat(gdf)  # merge all
    gdf.drop('points', axis=1, inplace=True)  # remove helper column
    gdf.to_csv(dir_output / 'elevation.csv', index=False)  # write CSV
    gpf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(gdf.longitude, gdf.latitude))  # rebuild geometry
    gpf.to_file(dir_output / 'elevation.geojson')  # write GeoJSON

def create_cosia_geojson(path, bounds, dir_output):
    """Processes and merges CoSIA land cover datasets into a deduplicated GeoJSON."""
    path = path.as_posix()  # convert to string
    files = glob.glob(path+'/*.gpkg', recursive=True)  # list all packages
    gdf = []  # store processed pieces
    leni = len(files)  # number of files
    for i, f in enumerate(files):  # iterate with index
        try:
            gfile = gpd.read_file(f).to_crs(crs=4326)  # load
        except Exception as e:
            print(e)  # show error
            continue  # skip broken file
        print(f'{i}/{leni}')  # progress
        gfile['geometry'] = set_precision(gfile.geometry, grid_size=0.001, mode='pointwise')  # precision reduction
        gfile.drop_duplicates(subset=['geometry'], inplace=True, keep='first')  # remove duplicates
        gfile = gfile.copy(deep=True)  # ensure independent copy
        gfile = gfile[~gfile.geometry.is_empty]  # drop empty geometries
        gdf.append(gfile)  # collect

    gdf = pd.concat(gdf)  # merge parts
    leni = len(gdf)  # count features
    gdf = gdf[gdf['geometry'] != None]  # remove missing geometry
    print(f'{leni} -> {len(gdf)}')  # display reduction
    gdf.to_file(dir_output / 'cosia.geojson', driver='GeoJSON')  # export

def download_elevation(code_dept: int, geo, dir_output: str) -> None:
    """Downloads and processes elevation data for a given department. https://geoservices.ign.fr/courbes-de-niveau"""
    # TO DO : The exact URL may be different for each department -> Automatisation
    check_and_create_path(dir_output)  # prepare directory
    print('DONWLOADING ELEVATION')
    if not (dir_output / f'{dir_output}/COURBE_1-0__SHP_LAMB93_D0{code_dept}_2021-01-01.7z').is_file():
        url = f'https://data.geopf.fr/telechargement/download/COURBES/COURBE_1-0__SHP_LAMB93_D0{code_dept}_2021-01-01/COURBE_1-0__SHP_LAMB93_D0{code_dept}_2021-01-01.7z'
        subprocess.run(['wget', url, '-O', f'{dir_output}/COURBE_1-0__SHP_LAMB93_D0{code_dept}_2021-04-01.7z'], check=True)
        unzip_7z(dir_output / f'COURBE_1-0__SHP_LAMB93_D0{code_dept}_2021-04-01.7z', dir_output)
    create_elevation_geojson(dir_output / f'COURBE_1-0__SHP_LAMB93_D0{code_dept}_2021-01-01', geo.bounds, dir_output)

def download_cosia(code_dept: int, geo, dir_output: str) -> None:
    """Downloads and processes COSIA land cover data for a department if available between 2019-2023. https://cosia.ign.fr/"""
    check_and_create_path(dir_output)  # ensure directory
    for y in np.arange(2019, 2024):  # iterate over years
        if (dir_output / f'{dir_output}/CoSIA_D0{code_dept}_{y}.zip').is_file():
            unzip_file(dir_output / f'CoSIA_D0{code_dept}_{y}.zip', dir_output)  # extract archive
        if (dir_output / f'{dir_output}/CoSIA_D0{code_dept}_{y}').is_dir():
            create_cosia_geojson(dir_output / f'CoSIA_D0{code_dept}_{y}', geo.bounds, dir_output)  # convert dataset
            break  # stop after first available year
    try:
        os.remove(dir_output / f'CoSIA_D0{code_dept}_{y}.zip')  # cleanup
    except:
        pass  # ignore missing file

def download_foret(code_dept: int, dept, dir_output) -> None:
    """Downloads and processes forest type land cover data from BDFORET. https://geoservices.ign.fr/bdforet#telechargementv2"""

    check_and_create_path(dir_output)  # base folder
    check_and_create_path(dir_output / 'BDFORET')  # subfolder
    print('DONWLOADING FOREST LANDCOVER')
    url = dico_foret_url[dept]  # dataset url
    date_pattern = r'\d{4}-\d{2}-\d{2}'  # search pattern
    match = re.search(date_pattern, url)  # regex match
    if match:
        date = match.group()  # keep date
        print("Date found:", date)
    else:
        print("No date found in the URL")
    path = (dir_output / f'BDFORET_2-0__SHP_LAMB93_D0{code_dept}_{date}' / 'BDFORET').as_posix()  # folder path
    files = glob.glob(path+'/**/*.shp', recursive=True)  # list shapefiles
    print(files, path)  # debug
    gdf = gpd.read_file(files[0]).to_crs(crs=4326)  # open
    gdf['code'] = gdf['ESSENCE'].apply(lambda x : valeurs_foret_attribut[x])  # map codes
    gdf.to_file(dir_output / 'BDFORET' / 'foret.geojson')  # export

def graph2geo(graph, nodes, edges, name):
    """Converts a NetworkX graph from OSMnx to GeoDataFrames and appends them to provided lists."""
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(graph)  # convert to GeoDataFrames
    gdf_nodes['name'] = name  # annotate nodes
    gdf_edges['name'] = name  # annotate edges
    nodes.append(gdf_nodes)  # store nodes
    edges.append(gdf_edges)  # store edges

def download_osnmx(geo, dir_output) -> None:
    """Downloads and processes road network data using OSMnx and exports it to GeoJSON."""
    check_and_create_path(dir_output)  # ensure folder
    print('DONWLOADING OSMNX')  # info
    graph = ox.graph_from_polygon(unary_union(geo['geometry'].values), network_type='all')  # fetch graph
    nodesArray = []  # to collect nodes
    edgesArray = []  # to collect edges
    graph2geo(graph, nodesArray, edgesArray, '')  # convert graph
    edges = pd.concat(edgesArray)  # merge edges
    subedges = edges[edges['highway'].isin(['motorway', 'primary', 'secondary', 'tertiary', 'path'])]  # filter
    subedges['coef'] = 1  # constant coefficient
    subedges['label'] = 0  # init label
    for i, hw in enumerate(['motorway', 'primary', 'secondary', 'tertiary', 'path']):
        subedges.loc[subedges['highway'] == hw, 'label'] = i + 1  # set label
    subedges[['geometry', 'label']].to_file(dir_output / 'osmnx.geojson', driver='GeoJSON')  # export

def download_region(departement, dir_output):
    """Downloads the administrative boundary (GeoJSON) of a French department.https://france-geojson.gregoiredavid.fr/"""
    print('DONWLING REGION')  # progress
    check_and_create_path(dir_output)  # ensure folder
    url = f'https://france-geojson.gregoiredavid.fr/repo/departements/{name2intstr[departement]}-{name2strlow[departement]}/{departement}.geojson'
    subprocess.run(['wget', url, '-O', f'{dir_output}/geo.json'], check=True)  # download file
    region = json.load(open(f'{dir_output}/geo.json'))  # load json
    polys = [region["geometry"]]  # polygon list
    geom = [shape(i) for i in polys]  # shapely geometry
    region = gpd.GeoDataFrame({'geometry':geom})  # to GeoDataFrame
    region.to_file(dir_output / 'geo.geojson')  # save result

def download_hexagones(path, geo, dir_output, departement):
    """Filters a hexagonal grid file to match the specified region and exports it to GeoJSON. See Kontur"""
    print('CREATE HEXAGONES')  # progress
    check_and_create_path(dir_output)  # ensure folder
    if 'corse' in departement:
        hexa_france = gpd.read_file(path / 'h3_corse_7.gpkg')  # load Corsica grid
    else:
        hexa_france = gpd.read_file(path / 'hexagones_france.gpkg')  # load mainland grid
    hexa_france['isdep'] = hexa_france['geometry'].apply(lambda x : geo.contains(x))  # filter by geometry
    hexa = hexa_france[hexa_france['isdep']]  # keep matching cells
    hexa.to_file(dir_output / 'hexagones.geojson', driver='GeoJSON')  # export

def haversine(p1, p2, unit = 'kilometer'):
    """Computes the Haversine distance between two geographic points in kilometers or meters."""
    import math  # import inside to avoid global dependency
    lon1, lat1 = p1  # first point
    lon2, lat2 = p2  # second point
    R = 6371000  # Earth radius in meters
    phi_1 = math.radians(lat1)  # convert to radians
    phi_2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2  # formula
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))  # central angle
    meters = R * c  # distance in meters
    km = meters / 1000.0  # convert to km
    if unit == 'kilometer':
        return round(km, 3)  # return in km
    elif unit == 'meters':
        return round(meters)  # return in meters
    else:
        return math.inf  # invalid unit
