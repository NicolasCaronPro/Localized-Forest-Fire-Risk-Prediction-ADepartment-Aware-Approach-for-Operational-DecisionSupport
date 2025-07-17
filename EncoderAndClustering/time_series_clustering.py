from tslearn.clustering import TimeSeriesKMeans
import numpy as np
from pathlib import Path
import pickle

def read_object(filename: str, path : Path):
    if not (path / filename).is_file():
        print(f'{path / filename} not found')
        return None
    return pickle.load(open(path / filename, 'rb'))

def cluster_time_series_from_targets(
    departments,
    target,
    train_dates,
    all_dates,
    drop_departments,
    root_target,
    resolution,
    raster_dir,
    n_clusters=4,
    distance_metric="dtw",
    max_iter=10,
    random_state=42,
):
    """
    Loads node-level time series targets and clusters them using TimeSeriesKMeans.

    Args:
        departments (list): List of department IDs.
        target (str): Target type, e.g., 'risk' or 'nbsinister'.
        train_date (str): Cut-off date for training (format: 'YYYY-MM-DD').
        all_dates (list): List of all available dates (strings).
        drop_departments (list): Departments to exclude from processing.
        root_target (Path): Root directory containing target data.
        resolution (str): Data resolution folder name.
        raster_dir (Path): Directory where raster files are stored.
        n_clusters (int): Number of clusters.
        distance_metric (str): Clustering distance metric (e.g., "dtw").
        max_iter (int): Max iterations for clustering.
        random_state (int): Random seed for reproducibility.
        read_object (callable): Function used to load .pkl files.

    Returns:
        cluster_model: Trained TimeSeriesKMeans model.
        node_cluster (dict): Mapping from node ID to assigned cluster.
        all_clusters (np.ndarray): Cluster label for each node.
    """
    assert read_object is not None, "A read_object must be provided to load data."

    dir_target_bin = Path('path_to_target')
    target_per_node = {}

    train_dates_id = np.asarray([all_dates.index(date) for date in train_dates])

    for dept in departments:
        if dept in drop_departments:
            continue

        target_value = read_object(f'{dept}binScale0.pkl', dir_target_bin)

        assert target_value is not None, f"Missing target for department {dept}"

        # Time slice
        target_value = target_value[:, :, train_dates_id]

        target_vector = np.nansum(target_value, axis=0)
        target_per_dept[dept] = target_vector

    # Convert to matrix
    time_series_matrix = np.asarray(list(target_per_dept.values()))

    # Clustering
    cluster_model = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric=distance_metric,
        max_iter=max_iter,
        random_state=random_state
    )
    cluster_model.fit(time_series_matrix)
    all_clusters = cluster_model.predict(time_series_matrix)

    node_cluster = {
        node: cluster for node, cluster in zip(target_per_dept.keys(), all_clusters)
    }

    return cluster_model, node_cluster, all_clusters
