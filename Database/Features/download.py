import requests
from shapely import unary_union, set_precision
import pandas as pd
import osmnx as ox
from sympy import subsets
import wget
import subprocess
from itertools import chain
import re
import geopandas as gpd
import os
import py7zr
import glob
import zipfile
import geojson
from tools import *

def myround(x):
    """Round coordinates to 3 decimal places."""
    return (round(x[0], 3), round(x[1], 3))

def get_latitude(x):
    """Return the latitude (second element) from a (lon, lat) tuple."""
    return x[1]

def get_longitude(x):
    """Return the longitude (first element) from a (lon, lat) tuple."""
    return x[0]

def unzip_7z(file_path, destination_folder) -> None:
    """Extracts a .7z archive into the specified destination folder."""
    os.makedirs(destination_folder, exist_ok=True)
    with py7zr.SevenZipFile(file_path, mode='r') as archive:
        archive.extractall(path=destination_folder)

def unzip_file(file_path, destination_folder) -> None:
    """Extracts a .zip archive into the specified destination folder."""
    os.makedirs(destination_folder, exist_ok=True)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)

def download_population(path, geo, dir_output) -> None:
    """Filters and saves population data within a specified geographic boundary. https://www.kontur.io/datasets/population-dataset/"""
    check_and_create_path(dir_output)
    print('DONWLOADING POPULATION')
    if not (path / 'kontur_population_FR_20231101.gpkg').is_file():
        print('ERROR : Add population download')
        exit(1)
    file = gpd.read_file(path / 'kontur_population_FR_20231101.gpkg')
    file.to_crs('EPSG:4326', inplace=True)
    file['isdep'] = file['geometry'].apply(lambda x : geo.contains(x))
    geo_pop = file[file['isdep']]
    geo_pop['latitude'] = geo_pop['geometry'].apply(lambda x : float(x.centroid.y))
    geo_pop['longitude'] = geo_pop['geometry'].apply(lambda x : float(x.centroid.x))
    geo_pop.to_file(dir_output / 'population.geojson')
    geo_pop.to_csv(dir_output / 'population.csv', index=False)

def create_elevation_geojson(path, bounds, dir_output):
    """Processes elevation contour data and outputs a GeoJSON and CSV of average point altitudes."""
    path = path.as_posix()
    files = glob.glob(path+'/COURBE/**/*.shp', recursive=True)
    gdf = []
    for f in files:
        try:
            gdft = gpd.read_file(f).to_crs(crs=4326)
        except:
            continue
        if "ALTITUDE" not in gdft.columns:
            continue
        gdft['points'] = gdft.apply(lambda x: [y for y in x['geometry'].coords], axis=1)
        gdft['altitude'] = gdft.apply(lambda x: [x['ALTITUDE'] for y in x['geometry'].coords], axis=1)
        points = pd.Series(list(chain.from_iterable(gdft['points'])))
        altitude = pd.Series(list(chain.from_iterable(gdft['altitude'])))
        gdft = pd.DataFrame({'altitude': altitude, 'geometry': points})
        gdft['points'] = gdft['geometry'].apply(myround)
        gb = gdft.groupby('points')
        mean = gb['altitude'].mean().reset_index()
        mean['latitude'] = mean['points'].apply(get_latitude)
        mean['longitude'] = mean['points'].apply(get_longitude)
        mean = gpd.GeoDataFrame(mean, geometry=gpd.points_from_xy(mean.longitude, mean.latitude))
        mean = mean[(mean['latitude'] >= bounds['miny'].values[0]) & (mean['latitude'] <= bounds['maxy'].values[0]) & \
                    (mean['longitude'] >= bounds['minx'].values[0]) & (mean['longitude'] <= bounds['maxx'].values[0])]
        if mean.empty:
            continue
        gdf.append(mean)

    gdf = pd.concat(gdf)
    gdf.drop('points', axis=1, inplace=True)
    gdf.to_csv(dir_output / 'elevation.csv', index=False)
    gpf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(gdf.longitude, gdf.latitude))
    gpf.to_file(dir_output / 'elevation.geojson')

def create_cosia_geojson(path, bounds, dir_output):
    """Processes and merges CoSIA land cover datasets into a deduplicated GeoJSON."""
    path = path.as_posix()
    files = glob.glob(path+'/*.gpkg', recursive=True)
    gdf = []
    leni = len(files)
    for i, f in enumerate(files):
        try:
            gfile = gpd.read_file(f).to_crs(crs=4326)
        except Exception as e:
            print(e)
            continue
        print(f'{i}/{leni}')
        gfile['geometry'] = set_precision(gfile.geometry, grid_size=0.001, mode='pointwise')
        gfile.drop_duplicates(subset=['geometry'], inplace=True, keep='first')
        gfile = gfile.copy(deep=True)
        gfile = gfile[~gfile.geometry.is_empty]
        gdf.append(gfile)

    gdf = pd.concat(gdf)
    leni = len(gdf)
    gdf = gdf[gdf['geometry'] != None]
    print(f'{leni} -> {len(gdf)}')
    gdf.to_file(dir_output / 'cosia.geojson', driver='GeoJSON')

def download_elevation(code_dept: int, geo, dir_output: str) -> None:
    """Downloads and processes elevation data for a given department. https://geoservices.ign.fr/courbes-de-niveau"""
    check_and_create_path(dir_output)
    print('DONWLOADING ELEVATION')
    if not (dir_output / f'{dir_output}/COURBE_1-0__SHP_LAMB93_D0{code_dept}_2021-01-01.7z').is_file():
        url = f'https://data.geopf.fr/telechargement/download/COURBES/COURBE_1-0__SHP_LAMB93_D0{code_dept}_2021-01-01/COURBE_1-0__SHP_LAMB93_D0{code_dept}_2021-01-01.7z'
        subprocess.run(['wget', url, '-O', f'{dir_output}/COURBE_1-0__SHP_LAMB93_D0{code_dept}_2021-04-01.7z'], check=True)
        unzip_7z(dir_output / f'COURBE_1-0__SHP_LAMB93_D0{code_dept}_2021-04-01.7z', dir_output)
    create_elevation_geojson(dir_output / f'COURBE_1-0__SHP_LAMB93_D0{code_dept}_2021-01-01', geo.bounds, dir_output)

def download_cosia(code_dept: int, geo, dir_output: str) -> None:
    """Downloads and processes COSIA land cover data for a department if available between 2019-2023. https://cosia.ign.fr/"""
    check_and_create_path(dir_output)
    for y in np.arange(2019, 2024):
        if (dir_output / f'{dir_output}/CoSIA_D0{code_dept}_{y}.zip').is_file():
            unzip_file(dir_output / f'CoSIA_D0{code_dept}_{y}.zip', dir_output)
        if (dir_output / f'{dir_output}/CoSIA_D0{code_dept}_{y}').is_dir():
            create_cosia_geojson(dir_output / f'CoSIA_D0{code_dept}_{y}', geo.bounds, dir_output)
            break
    try:
        os.remove(dir_output / f'CoSIA_D0{code_dept}_{y}.zip')
    except:
        pass

def download_foret(code_dept: int, dept, dir_output) -> None:
    """Downloads and processes forest type land cover data from BDFORET. https://geoservices.ign.fr/bdforet#telechargementv2"""
    
    check_and_create_path(dir_output)
    check_and_create_path(dir_output / 'BDFORET')
    print('DONWLOADING FOREST LANDCOVER')
    url = dico_foret_url[dept]
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    match = re.search(date_pattern, url)
    if match:
        date = match.group()
        print("Date found:", date)
    else:
        print("No date found in the URL")
    path = (dir_output / f'BDFORET_2-0__SHP_LAMB93_D0{code_dept}_{date}' / 'BDFORET').as_posix()
    files = glob.glob(path+'/**/*.shp', recursive=True)
    print(files, path)
    gdf = gpd.read_file(files[0]).to_crs(crs=4326)
    gdf['code'] = gdf['ESSENCE'].apply(lambda x : valeurs_foret_attribut[x])
    gdf.to_file(dir_output / 'BDFORET' / 'foret.geojson')

def graph2geo(graph, nodes, edges, name):
    """Converts a NetworkX graph from OSMnx to GeoDataFrames and appends them to provided lists."""
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(graph)
    gdf_nodes['name'] = name
    gdf_edges['name'] = name
    nodes.append(gdf_nodes)
    edges.append(gdf_edges)

def download_osnmx(geo, dir_output) -> None:
    """Downloads and processes road network data using OSMnx and exports it to GeoJSON."""
    check_and_create_path(dir_output)
    print('DONWLOADING OSMNX')
    graph = ox.graph_from_polygon(unary_union(geo['geometry'].values), network_type='all')
    nodesArray = []
    edgesArray = []
    graph2geo(graph, nodesArray, edgesArray, '')
    edges = pd.concat(edgesArray)
    subedges = edges[edges['highway'].isin(['motorway', 'primary', 'secondary', 'tertiary', 'path'])]
    subedges['coef'] = 1
    subedges['label'] = 0
    for i, hw in enumerate(['motorway', 'primary', 'secondary', 'tertiary', 'path']):
        subedges.loc[subedges['highway'] == hw, 'label'] = i + 1
    subedges[['geometry', 'label']].to_file(dir_output / 'osmnx.geojson', driver='GeoJSON')

def download_region(departement, dir_output):
    """Downloads the administrative boundary (GeoJSON) of a French department.https://france-geojson.gregoiredavid.fr/"""
    print('DONWLING REGION')
    check_and_create_path(dir_output)
    url = f'https://france-geojson.gregoiredavid.fr/repo/departements/{name2intstr[departement]}-{name2strlow[departement]}/{departement}.geojson'
    subprocess.run(['wget', url, '-O', f'{dir_output}/geo.json'], check=True)
    region = json.load(open(f'{dir_output}/geo.json'))
    polys = [region["geometry"]]
    geom = [shape(i) for i in polys]
    region = gpd.GeoDataFrame({'geometry':geom})
    region.to_file(dir_output / 'geo.geojson')

def download_hexagones(path, geo, dir_output, departement):
    """Filters a hexagonal grid file to match the specified region and exports it to GeoJSON. Use them from kontur"""
    print('CREATE HEXAGONES')
    check_and_create_path(dir_output)
    if 'corse' in departement:
        hexa_france = gpd.read_file(path / 'h3_corse_7.gpkg')
    else:
        hexa_france = gpd.read_file(path / 'hexagones_france.gpkg')
    hexa_france['isdep'] = hexa_france['geometry'].apply(lambda x : geo.contains(x))
    hexa = hexa_france[hexa_france['isdep']]
    hexa.to_file(dir_output / 'hexagones.geojson', driver='GeoJSON')

def haversine(p1, p2, unit = 'kilometer'):
    """Computes the Haversine distance between two geographic points in kilometers or meters."""
    import math
    lon1, lat1 = p1
    lon2, lat2 = p2
    R = 6371000
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    meters = R * c
    km = meters / 1000.0
    if unit == 'kilometer':
        return round(km, 3)
    elif unit == 'meters':
        return round(meters)
    else:
        return math.inf