from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from astropy.convolution import convolve_fft
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy.ndimage import generic_filter
from copy import deepcopy

class MeanSequence:
    def __init__(self):
        pass

    def compute_sequence_month(self, dataframe, database_name):
        # Define month groups representing different seasons
        group_month = [
            [2, 3, 4, 5],    # Medium season
            [6, 7, 8, 9],    # High season
            [10, 11, 12, 1]  # Low season
        ]

        # Column name containing the number of incidents
        nbsinister_col = 'nbsinister'

        # Log the total number of incidents
        logger.info(f'{nbsinister_col} -> {dataframe[nbsinister_col].sum()}')

        # Names of the three seasonal groups
        name = ['medium', 'high', 'low']

        # Extract the month from the date using the allDates mapping
        dataframe['month_non_encoder'] = dataframe['date'].apply(
            lambda x: int(allDates[int(x)].split('-')[1])
        )

        # Sort the data by graph point and date.
        # In our case graph_id is equivalent at department
        dataframe = dataframe.sort_values(by=['graph_id', 'date'])

        # Extract the unique graph points (nodes)
        points = dataframe['graph_id'].unique()

        # Initialize dictionary to store sequences by season and graph point
        self.sequences_month = {}

        # Time delta threshold in days for continuing a sequence
        td = 3

        # Keep only rows where the number of incidents is greater than 0
        fire_dataframe = dataframe[dataframe[nbsinister_col] > 0].reset_index()

        # Loop over each season group
        for i, months in enumerate(group_month):

            # Filter the dataframe to keep only months from the current season
            df = fire_dataframe[fire_dataframe['month_non_encoder'].isin(months)]

            # Initialize dictionary for the current season
            self.sequences_month[name[i]] = {}

            # Loop over each unique graph point
            for point in points:
                # Initialize storage for this graph point in the current season
                self.sequences_month[name[i]][point] = {}
                self.sequences_month[name[i]][point]['dates'] = []
                self.sequences_month[name[i]][point]['mean_size'] = 1

                # Filter data for this specific graph point
                df_point = df[df['graph_id'] == point]

                # If no data exists for this point, skip to the next
                if len(df_point) == 0:
                    continue

                # Get unique IDs from the point data
                uids = df_point['id'].unique()

                # Only consider the first unique ID for further processing
                df_point = df_point[df_point['id'] == uids[0]]

                # Group the data by date and iterate
                for date, group in df_point.groupby('date'):

                    # If no sequence exists yet, start a new one
                    if not self.sequences_month[name[i]][point]['dates']:
                        self.sequences_month[name[i]][point]['dates'].append([date])
                        continue

                    find_seq = False

                    # Check if the current date continues any existing sequence
                    for sei, seq in enumerate(self.sequences_month[name[i]][point]['dates']):
                        # If the current date is within the threshold of the last date in the sequence
                        if (date - seq[-1]) <= td:
                            # Avoid adding the same date twice
                            if date in seq:
                                continue
                            # Add the date to the sequence
                            seq.append(date)
                            find_seq = True

                            # Update the sequence
                            self.sequences_month[name[i]][point]['dates'][sei] = seq
                            break

                    # If no existing sequence can be continued, start a new one
                    if not find_seq:
                        self.sequences_month[name[i]][point]['dates'].append([date])

                # Compute the average size of all sequences for this point
                total_size = sum(len(seq_dates) for seq_dates in self.sequences_month[name[i]][point]['dates'])
                num_sequences = len(self.sequences_month[name[i]][point]['dates'])
                mean = round(total_size / num_sequences)

                # Compute mean_size with logic to increase it if the mean is larger than 1
                if mean > 1:
                    self.sequences_month[name[i]][point]['mean_size'] = 2 * mean + 1
                else:
                    self.sequences_month[name[i]][point]['mean_size'] = 1

class KMeansRisk:
    """
    Class that uses KMeans to identify risk categories.
    """

    def __init__(self, n_clusters):
        """
        :param n_clusters: Number of clusters for KMeans.
        """
        self.n_clusters = n_clusters  # Set the number of clusters
        self.name = f'KMeansRisk_{n_clusters}'  # A name identifier for the instance
        self.label_map = None  # Will store a mapping from raw KMeans labels to ordered labels

    def fit(self, X, y):
        """
        Fit the KMeans model to the data.

        :param X: Data to fit the model on.
        :param y: (Unused) Labels, can be ignored or removed.
        """
        # If the number of unique values is less than the number of clusters, don't cluster
        if np.unique(X).shape[0] < self.n_clusters:
            self.zeros = True  # Flag indicating clustering was skipped
            return np.zeros(X.shape)  # Return zeros as default cluster

        self.zeros = False  # Clustering will be performed

        # Initialize and fit the KMeans model on unique values to avoid bias
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.model.fit(np.unique(X).reshape(-1, 1))  # Reshape required for scikit-learn

        centroids = self.model.cluster_centers_  # Extract cluster centers

        # Determine sorting order of clusters based on distance from origin
        if centroids.shape[1] > 1:
            magnitudes = np.linalg.norm(centroids, axis=1)  # Euclidean norm for multidimensional data
            sorted_indices = np.argsort(magnitudes)  # Sort by magnitude
        else:
            centroids = centroids.flatten()
            sorted_indices = np.argsort(centroids)  # Sort 1D centroids directly

        # Create a label map that reorders cluster labels from lowest to highest
        self.label_map = {label: i for i, label in enumerate(sorted_indices)}

    def predict(self, X):
        """
        Predict cluster classes using the fitted KMeans model.

        :param X: Data to classify.
        :return: Predicted cluster classes.
        """
        if self.zeros:
            return np.zeros_like(X)  # Return all zeros if clustering was skipped

        kmeans_labels = self.model.predict(X)  # Get KMeans cluster labels
        return np.vectorize(self.label_map.get)(kmeans_labels)  # Map to ordered risk levels
    
class KMeansRiskZerosHandle:
    """
    Class that uses KMeans to identify risk categories,
    with special handling for zero values in the input.
    """

    def __init__(self, n_clusters):
        """
        :param n_clusters: Number of desired total clusters, including a special one for zeros.
        """
        self.n_clusters = n_clusters - 1  # Reserve one cluster index for zero values
        self.name = f'KMeansRisk_{n_clusters}'  # Model identifier
        self.label_map = None  # To map raw cluster labels to ordered risk levels

    def fit(self, X, y):
        """
        Fit the KMeans model to non-zero data values.

        :param X: Input data to cluster.
        :param y: (Unused) Target labels, not used in this method.
        """
        # Extract only positive (non-zero) values
        X_val = X[X > 0].reshape(-1, 1)

        # If there are no non-zero values, skip fitting
        if np.unique(X_val).shape[0] == 0:
            self.model = None  # No model will be trained
            return

        # Train KMeans on the non-zero values with up to (n_clusters - 1) clusters
        self.model = KMeans(
            n_clusters=min(self.n_clusters, np.unique(X_val).shape[0]),
            random_state=42,
            n_init=10
        )
        self.model.fit(X_val)

        centroids = self.model.cluster_centers_  # Get cluster centers

        # Sort clusters by magnitude or scalar value
        if centroids.shape[1] > 1:
            magnitudes = np.linalg.norm(centroids, axis=1)
            sorted_indices = np.argsort(magnitudes)
        else:
            centroids = centroids.flatten()
            sorted_indices = np.argsort(centroids)

        # Map KMeans labels to ordered cluster indices starting from 1
        # (0 will be reserved for original zero values in data)
        self.label_map = {label: i + 1 for i, label in enumerate(sorted_indices)}

    def predict(self, X):
        """
        Predict risk classes for the input data using the trained KMeans model.

        :param X: Input data for classification.
        :return: Array of predicted cluster classes.
        """
        res = np.zeros(X.shape)  # Default all predictions to 0 (for zeros in input)

        # If model was never trained (e.g., all inputs were zero)
        if self.model is None:
            return res.reshape(-1, 1)  # Return zero cluster for all

        # Extract non-zero values for prediction
        X_val = X[X > 0].reshape(-1, 1)

        # If there are no non-zero values, return all-zero result
        if X_val.shape[0] == 0:
            return res.reshape(-1)

        # Predict clusters for non-zero values
        kmeans_labels = self.model.predict(X_val)

        # Map predicted labels to ordered risk levels
        res[X > 0] = np.vectorize(self.label_map.get)(kmeans_labels)

        return res.reshape(-1)

class PreprocessorConv:
    def __init__(self, graph, kernel='Specialized', conv_type='laplace+mean', id_col=None, persistence=False,):
        """
        Initialize the PreprocessorConv.

        :param graph: An object containing `sequences_month` data structure.
        :param conv_type: Type of convolution to apply ('laplace', 'mean', 'laplace+mean', 'sum', or 'max').
        :param id_col: List of column names or indices representing IDs. Defaults to None.
        """
        assert conv_type in ['laplace', 'laplace+median', 'mean', 'laplace+mean', 'sum', 'max', 'median', 'gradient', 'gaussian', 'cubic', 'quartic', 'circular'], \
            "conv_type must be 'laplace', 'mean', 'laplace+mean', 'sum', 'gradient' or 'max'."
        self.graph = graph
        self.conv_type = conv_type
        self.id_col = id_col if id_col is not None else ['id']
        self.kernel = kernel
        self.persistence = persistence

    def _get_season_name(self, month):
        """
        Determine the season name based on the month.

        :param month: Integer representing the month (1 to 12).
        :return: Season name as a string ('medium', 'high', or 'low').
        """
        group_month = [
            [2, 3, 4, 5],    # Medium season
            [6, 7, 8, 9],    # High season
            [10, 11, 12, 1]  # Low season
        ]
        names = ['medium', 'high', 'low']
        for i, group in enumerate(group_month):
            if month in group:
                return names[i]
        raise ValueError(f"Month {month} does not belong to any season group.")

    def _laplace_convolution(self, X, kernel_size):
        """
        Apply Laplace convolution using a custom kernel from Astropy.

        :param X: Input array for convolution.
        :param kernel_size: Size of the convolution kernel.
        :return: Convoluted array.
        """
        # Create Laplace kernel
        kernel_daily = np.abs(np.arange(-(kernel_size // 2), kernel_size // 2 + 1))
        kernel_daily += 1
        kernel_daily = 1 / kernel_daily
        kernel_daily = kernel_daily.reshape(-1,1)

        if self.persistence:
            if kernel_size > 1:
                kernel_daily[:kernel_size // 2] = 0
            else:
                return np.zeros(X.shape).reshape(-1)

        if np.unique(kernel_daily)[0] == 0 and np.unique(kernel_daily).shape[0] == 1:
            kernel_daily += 1
            
        res = convolve_fft(X.reshape(-1), kernel_daily.reshape(-1), normalize_kernel=False, fill_value=0.)

        return res

    def _gaussian_convolution(self, X, kernel_size):
        if kernel_size <= 1:
            return np.zeros_like(X).reshape(-1)
        
        if kernel_size % 2 == 0:
            kernel_size += 1

        sigma = (kernel_size - 1) / 6
        x = np.linspace(-kernel_size // 2, kernel_size // 2 + 1, kernel_size)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        if self.persistence:
            kernel[:kernel_size // 2] = 0

        return convolve_fft(X.reshape(-1), kernel.reshape(-1), fill_value=0.0).reshape(-1)
    
    def _cubic_convolution(self, X, kernel_size):
        if kernel_size <= 1:
            return np.zeros_like(X).reshape(-1)
        
        if kernel_size % 2 == 0:
            kernel_size += 1

        x = np.linspace(-1, 1, kernel_size)
        kernel = (1 - np.abs(x))**3
        kernel = np.clip(kernel, 0, None)

        if self.persistence:
            kernel[:kernel_size // 2] = 0

        return convolve_fft(X.reshape(-1), kernel.reshape(-1), fill_value=0.0).reshape(-1)
    
    def _quartic_convolution(self, X, kernel_size):
        if kernel_size <= 1:
            return np.zeros_like(X).reshape(-1)
        
        if kernel_size % 2 == 0:
            kernel_size += 1

        x = np.linspace(-1, 1, kernel_size)
        kernel = (1 - x**2)**2
        kernel = np.clip(kernel, 0, None)
        if self.persistence:
            kernel[:kernel_size // 2] = 0
        return convolve_fft(X.reshape(-1), kernel.reshape(-1), fill_value=0.0).reshape(-1)
    
    def _circular_convolution(self, X, kernel_size):
        if kernel_size <= 1:
            return np.zeros_like(X).reshape(-1)
        
        if kernel_size % 2 == 0:
            kernel_size += 1 

        x = np.linspace(-1, 1, kernel_size)
        kernel = np.sqrt(1 - x**2)
        kernel = np.clip(kernel, 0, None)
        if self.persistence:
            kernel[:kernel_size // 2] = 0
        return convolve_fft(X.reshape(-1), kernel.reshape(-1), fill_value=0.0).reshape(-1)
    
    def _mean_convolution(self, X, kernel_size):
        """
        :param X: Input array for convolution.
        :param kernel_size: Size
        Apply mean convolution using a custom kernel from Astropy.of the convolution kernel.
        :return: Convoluted array.
        """
        # Create mean kernel
        if kernel_size == 1:
            return np.zeros_like(X).reshape(-1)
        
        kernel_season = np.ones(kernel_size, dtype=float)
        kernel_season /= kernel_size  # Normalize the kernel

        if self.persistence:
            kernel_season[:kernel_size // 2] = 0
 
        #kernel_season[kernel_size // 2] = 0
        res = convolve_fft(X.reshape(-1), kernel_season.reshape(-1), normalize_kernel=False, fill_value=0.).reshape(-1)

        return res
    
    def median_convolution(self, X, kernel_size):
        """
        Apply median convolution using a custom kernel, excluding the center pixel.

        Parameters:
            X (array-like): Input array for convolution.
            kernel_size (int): Size of the convolution kernel (must be odd).

        Returns:
            numpy.ndarray: Convoluted array with median filtering applied.
        """
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer.")
        
        if kernel_size == 1:
            return np.zeros(X.shape).reshape(-1)

        def custom_median_filter(window):
            center = len(window) // 2
            if self.persistence:
                window[center:] = np.nan
            window = np.delete(window, center)
            return np.nanmedian(window)
        
        # Apply the custom median filter
        return generic_filter(X, custom_median_filter, size=kernel_size).reshape(-1)

    def _gradient_convolution(self, X, kernel_size):
        """
        Applique une convolution de gradient sur l'entrée avec une fenêtre de taille `kernel_size`.

        :param X: Tableau d'entrée (2D: samples x features).
        :param kernel_size: Taille de la fenêtre de convolution.
        :return: Tableau du gradient calculé sur la fenêtre glissante.
        """
        from scipy.ndimage import convolve1d

        # Création d'un noyau de gradient centré
        kernel = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
        kernel = kernel / np.sum(np.abs(kernel))  # Normalisation

        return convolve1d(X, kernel, mode='nearest', axis=0).reshape(-1)

    def _sum_convolution(self, X, kernel_size):
        """
        Apply sum operation on the input with a sliding window of size `kernel_size`.

        :param X: Input array for summation (2D: samples x features).
        :param kernel_size: Size of the sliding window.
        :return: Array where each element is the sum of elements in the sliding window.
        """
        from scipy.ndimage import uniform_filter1d

        if self.persistence:
            mask = np.zeros_like(X, dtype=float)
            mask[kernel_size // 2:] = 0
            mask[:kernel_size // 2] = 1  # Persistence applies only to later values
            X = X * mask

        return uniform_filter1d(X, size=kernel_size, mode='constant', origin=0, axis=0) * kernel_size

    def _max_convolution(self, X, kernel_size):
        """
        Apply max operation on the input with a sliding window of size `kernel_size`.

        :param X: Input array for max operation (2D: samples x features).
        :param kernel_size: Size of the sliding window.
        :return: Array where each element is the max of elements in the sliding window.
        """
        from scipy.ndimage import maximum_filter1d

        if self.persistence:
            mask = np.zeros_like(X, dtype=float)
            mask[kernel_size // 2:] = 0
            mask[:kernel_size // 2] = 1
            X = X * mask

        res = maximum_filter1d(X, size=kernel_size, mode='constant', origin=0, axis=0)
        return res

    def apply(self, X, ids):
        """
        Apply the convolution based on the given type to the input data.

        :param X: Input array of shape (n_samples, n_features).
        :param ids: Array of shape (n_samples, len(id_col)) if `id_col` is a list, otherwise (n_samples, 1).
        :return: Processed array with convolutions applied.
        """
        X_processed = np.zeros_like(X, dtype=np.float32).reshape(-1)
        unique_ids = np.unique(ids, axis=0)

        for unique_id in unique_ids:

            if len(self.id_col) > 1 and not isinstance(self.id_col, str):
                mask = (ids[:, 0] == unique_id[0]) & (ids[:, 1] == unique_id[1])
            else:
                mask = ids[:, 0] == unique_id[0]

            if not np.any(mask):
                continue

            if self.kernel == 'Specialized':
                # Handle month_non_encoder
                if 'month_non_encoder' in self.id_col:
                    month_idx = self.id_col.index('month_non_encoder')
                    month = unique_id[month_idx]
                    season_name = self._get_season_name(month)
                    kernel_size = int(self.graph.sequences_month[season_name][unique_id[1]]['mean_size'])
                else:
                    raise ValueError(
                        "Error: 'month_non_encoder' is not specified in id_col. Please include it in id_col to proceed."
                    )
            else:
                kernel_size = (int(self.kernel) * 2 + 1) + 2 # Add 2 to make 0 bound

            if kernel_size == 1:
                X_processed[mask] = 0.0

            # Apply Laplace convolution if specified
            elif self.conv_type == 'laplace':
                laplace_result = self._laplace_convolution(X[mask], kernel_size)
                X_processed[mask] = laplace_result.reshape(-1)

            # Apply mean convolution if specified
            elif self.conv_type == 'mean':
                mean_result = self._mean_convolution(X[mask], kernel_size)
                X_processed[mask] = mean_result.reshape(-1)

            # Apply laplace+mean convolutions if specified
            elif self.conv_type == 'laplace+mean':
                laplace_result = self._laplace_convolution(X[mask], kernel_size)
                mean_result = self._mean_convolution(X[mask], kernel_size)
                X_processed[mask] = (laplace_result + mean_result).reshape(-1)

            # Apply sum operation if specified
            elif self.conv_type == 'sum':
                sum_result = self._sum_convolution(X[mask], kernel_size)
                X_processed[mask] = sum_result.reshape(-1)

            # Apply max operation if specified
            elif self.conv_type == 'max':
                max_result = self._max_convolution(X[mask], kernel_size)
                X_processed[mask] = max_result.reshape(-1)

            elif self.conv_type == 'median':
                med_result = self.median_convolution(X[mask], kernel_size)
                #X_processed[mask] = (X[mask].reshape(-1) + med_result).reshape(-1)
                X_processed[mask] = med_result.reshape(-1)

            elif self.conv_type == 'laplace+median':
                laplace_result = self._laplace_convolution(X[mask], kernel_size)
                med_result = self.median_convolution(X[mask], kernel_size)
                X_processed[mask] = (laplace_result + med_result).reshape(-1)
            
            elif self.conv_type == 'gradient':
                gradient_result = self._gradient_convolution(X[mask], kernel_size)
                X_processed[mask] = gradient_result

            elif self.conv_type == 'gaussian':
                result = self._gaussian_convolution(X[mask], kernel_size)
                X_processed[mask] = result.reshape(-1)

            elif self.conv_type == 'cubic':
                result = self._cubic_convolution(X[mask], kernel_size)
                X_processed[mask] = result.reshape(-1)
            
            elif self.conv_type == 'quartic':
                result = self._quartic_convolution(X[mask], kernel_size)
                X_processed[mask] = result.reshape(-1)
            
            elif self.conv_type == 'circular':
                result = self._circular_convolution(X[mask], kernel_size)
                X_processed[mask] = result.reshape(-1)

            else:
                raise ValueError(
                    f"Error: conv_type must be in  ['laplace', 'mean', 'laplace+mean', 'sum', 'max'], got {self.conv_type}"
                )
            
        return X_processed
    
class ScalerClassRisk:
    def __init__(self, col_id, dir_output, target, scaler=None, class_risk=None, preprocessor=None):
        """
        Initialize the ScalerClassRisk.

        :param col_id: Column ID used to group data.
        :param dir_output: Directory to save output files (e.g., histograms).
        :param target: Target name used for labeling outputs.
        :param scaler: A scaler object (e.g., StandardScaler) for normalization. If None, no scaling is applied.
        :param class_risk: An object with `fit` and `predict` methods for risk classification.
        :param preprocessor: An object with an `apply` method to preprocess X before fitting.
        """
        self.n_clusters = 5
        self.col_id = col_id

        if scaler is None:
            scaler_name = None
        elif isinstance(scaler, MinMaxScaler):
            scaler_name = 'MinMax'
        elif isinstance(scaler, StandardScaler):
            scaler_name = 'Standard'
        elif isinstance(scaler, RobustScaler):
            scaler_name = 'Robust'
        else:
            raise ValueError(f'{scaler} wrong value')
        
        class_risk_name = class_risk.name if class_risk is not None else ''
        preprocessor_name = f'{preprocessor.conv_type}_{preprocessor.kernel}' if preprocessor is not None else None
        self.name = f'ScalerClassRisk_{preprocessor_name}_{scaler_name}_{class_risk_name}_{target}'
        self.dir_output = dir_output / self.name
        self.target = target
        self.scaler = scaler
        self.class_risk = class_risk
        self.preprocessor = preprocessor
        self.preprocessor_id_col = preprocessor.id_col if preprocessor is not None else None
        #check_and_create_path(self.dir_output)
        self.models_by_id = {}
        self.is_fit = False

    def fit(self, X, sinisters, ids, ids_preprocessor=None):
        """
        Fit models for each unique ID and calculate statistics.

        :param X: Array of values to scale and classify.
        :param sinisters: Array of sinisters values corresponding to each value in X.
        :param ids: Array of IDs corresponding to each value in X.
        :param ids_preprocessor: IDs to use for preprocessing (if preprocessor is not None).
        """
        self.is_fit = True
        print(f'########################################## {self.name} ##########################################')

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        if sinisters is not None and len(sinisters.shape) == 1:
            sinisters = sinisters.reshape(-1)

        X_ = np.copy(X)

        # Apply preprocessor if provided
        if self.preprocessor is not None:
            if ids_preprocessor is None:
                raise ValueError("ids_preprocessor must be provided when preprocessor is not None.")
            X = self.preprocessor.apply(X, ids_preprocessor)

        lambda_function = lambda x: max(x, 1)
        self.models_by_id = {}

        for unique_id in np.unique(ids):
            mask = ids == unique_id
            X_id = X[mask]

            if sinisters is not None:
                sinisters_id = sinisters[mask]
            else:
                sinisters_id = None

            # Scale data if scaler is provided
            if self.scaler is not None:
                scaler = deepcopy(self.scaler)
                scaler.fit(X_id.reshape(-1, 1))
                X_scaled = scaler.transform(X_id.reshape(-1, 1))
            else:
                scaler = None
                X_scaled = X_id

            X_scaled = X_scaled.astype(np.float32)

            # Fit the class_risk model
            class_risk = deepcopy(self.class_risk)

            if len(X_scaled.shape) == 1:
                X_scaled = np.reshape(X_scaled, (-1,1))

            sinisters_mean = None
            sinisters_min = None
            sinisters_max = None
            if class_risk is not None:
                class_risk.fit(X_scaled, sinisters_id)

                # Predict classes for current ID
                classes = class_risk.predict(X_scaled)

                if sinisters is not None:
                
                    # Apply the lambda function using np.vectorize for the condition `sinisters > 0`
                    if True in np.unique(sinisters_id > 0):
                        classes[sinisters_id > 0] = np.vectorize(lambda_function)(classes[sinisters_id > 0])

                    # Calculate statistics for each class
                    sinisters_mean = []
                    sinisters_min = []
                    sinisters_max = []

                    for cls in np.unique(classes):
                        class_mask = (classes == cls).reshape(-1)
                        sinisters_class = sinisters_id[class_mask]
                        sinisters_mean.append(np.mean(sinisters_class))
                        sinisters_min.append(np.min(sinisters_class))
                        sinisters_max.append(np.max(sinisters_class))

                    sinisters_mean = np.array(sinisters_mean)
                    sinisters_min = np.array(sinisters_min)
                    sinisters_max = np.array(sinisters_max)

            self.models_by_id[unique_id] = {
                'scaler': scaler,
                'class_risk': class_risk,
                'sinistres_mean': sinisters_mean,
                'sinistres_min': sinisters_min,
                'sinistres_max': sinisters_max
            }

        if sinisters is None:
            return

        pred = self.predict(X_, sinisters, ids, ids_preprocessor)

        if self.class_risk is not None:
            for cl in np.unique(pred):
                print(f'{cl} -> {pred[pred == cl].shape[0]}')

    def predict(self, X, sinisters, ids, ids_preprocessor=None):
        """
        Predict class labels using the appropriate model for each ID.

        :param X: Array of values to predict.
        :param ids: Array of IDs corresponding to each value in X.
        :param ids_preprocessor: IDs to use for preprocessing (if preprocessor is not None).
        :return: Array of predicted class labels.
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # Apply preprocessor if provided
        if self.preprocessor is not None:
            if ids_preprocessor is None:
                raise ValueError("ids_preprocessor must be provided when preprocessor is not None.")
            X = self.preprocessor.apply(X, ids_preprocessor)

        if self.class_risk is not None:
            predictions = np.zeros_like(ids, dtype=int)
        else:    
            predictions = np.zeros_like(ids, dtype=np.float32)

        for unique_id, model in self.models_by_id.items():
            if unique_id not in np.unique(ids):
                continue

            mask = ids == unique_id
            scaler = model['scaler']
            class_risk = model['class_risk']

            # Scale data if scaler exists
            if scaler is not None:
                X_scaled = scaler.transform(X[mask].reshape(-1,1))
            else:
                X_scaled = X[mask]

            if len(X_scaled.shape) == 1:
                X_scaled = np.reshape(X_scaled, (-1,1))

            if class_risk is not None:
                predictions[mask] = class_risk.predict(X_scaled.astype(np.float32)).reshape(-1)
            else:
                predictions[mask] = X_scaled.astype(np.float32).reshape(-1)
        
        if class_risk is not None:
            predictions[predictions >= self.n_clusters] = self.n_clusters - 1

        # Define the lambda function
        lambda_function = lambda x: max(x, 1)

        if sinisters is not None and class_risk is not None:
            sinisters = sinisters.reshape(-1)
            if np.any(sinisters > 0):
                # Apply the lambda function using np.vectorize for the condition `sinisters > 0`
                predictions[sinisters > 0] = np.vectorize(lambda_function)(predictions[sinisters > 0])

        return predictions

    def fit_predict(self, X, ids, sinisters, ids_preprocessor=None):
        self.fit(X, ids, sinisters, ids_preprocessor)
        return self.predict(X, ids, ids_preprocessor)

    def predict_stat(self, X, ids, stat_key):
        """
        Generic method to predict statistics (mean, min, max) for sinistres based on the class of each sample.

        :param X: Array of input values.
        :param ids: Array of IDs corresponding to each value in X.
        :param stat_key: Key to fetch the required statistic ('sinistres_mean', 'sinistres_min', 'sinistres_max').
        :return: Array of predicted statistics for each sample based on its class.
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        predictions = np.zeros(X.shape[0])

        for unique_id, model in self.models_by_id.items():
            if unique_id not in np.unique(ids):
                continue

            mask = ids == unique_id
            
            stats = model[stat_key]
            predictions[mask] = np.array([stats[int(cls)] for cls in X[mask]])

        return predictions

    def predict_nbsinister(self, X, ids):
        """
        Predict the mean sinistres for the class of each instance in X.

        :param X: Array of input values.
        :param ids: Array of IDs corresponding to each value in X.
        :return: Array of predicted mean sinistres for each class.
        """
        return self.predict_stat(X, ids, stat_key='sinistres_mean')

    def predict_nbsinister_min(self, X, ids):
        """
        Predict the minimum sinistres for the class of each instance in X.

        :param X: Array of input values.
        :param ids: Array of IDs corresponding to each value in X.
        :return: Array of predicted minimum sinistres for each class.
        """
        return self.predict_stat(X, ids, stat_key='sinistres_min')

    def predict_nbsinister_max(self, X, ids):
        """
        Predict the maximum sinistres for the class of each instance in X.

        :param X: Array of input values.
        :param ids: Array of IDs corresponding to each value in X.
        :return: Array of predicted maximum sinistres for each class.
        """
        return self.predict_stat(X, ids, stat_key='sinistres_max')
    
    def predict_risk(self, X, sinisters, ids, ids_preprocessor=None):
        if self.is_fit:
            return self.predict(X, sinisters, ids, ids_preprocessor)
        else:
            return self.fit_predict(X, sinisters, ids, ids_preprocessor)

def process_target(train_dataset, val_dataset, test_dataset, dir_post_process, meanSequence):

    graph_method = graph.graph_method

    new_cols = []

    train_dataset_ = train_dataset.copy(deep=True)
    val_dataset_ = val_dataset.copy(deep=True)
    test_dataset_ = test_dataset.copy(deep=True)
    
    res = {}

    ####################################################################################
    
    obj2 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target='nbsinister', scaler=None, class_risk=KMeansRiskZerosHandle(5))

    obj2.fit(train_dataset_['nbsinister'].values, train_dataset_['nbsinister'].values, train_dataset_['departement'].values)

    train_dataset_['nbsinister-kmeans-5-Class-Dept'] = obj2.predict(train_dataset_['nbsinister'].values,  train_dataset_['nbsinister'].values, train_dataset_['departement'].values)
    val_dataset_['nbsinister-kmeans-5-Class-Dept'] = obj2.predict(val_dataset_['nbsinister'].values,  val_dataset_['nbsinister'].values, val_dataset_['departement'].values)
    test_dataset_['nbsinister-kmeans-5-Class-Dept'] = obj2.predict(test_dataset_['nbsinister'].values,  test_dataset_['nbsinister'].values, test_dataset_['departement'].values)
    
    res[obj2.name] = obj2

    new_cols.append('nbsinister-kmeans-5-Class-Dept')

    ######################################################################################

    obj2 = ScalerClassRisk(col_id='departement', dir_output = dir_post_process, target='burned_area', scaler=None, class_risk=KMeansRiskZerosHandle(5))

    obj2.fit(train_dataset_['burned_area'].values, train_dataset_['nbsinister'].values, train_dataset_['departement'].values)

    train_dataset_['burnedarea-kmeans-5-Class-Dept'] = obj2.predict(train_dataset_['burned_area'].values,  train_dataset_['burned_area'].values, train_dataset_['departement'].values)
    val_dataset_['burnedarea-kmeans-5-Class-Dept'] = obj2.predict(val_dataset_['burned_area'].values,  val_dataset_['burned_area'].values, val_dataset_['departement'].values)
    test_dataset_['burnedarea-kmeans-5-Class-Dept'] = obj2.predict(test_dataset_['burned_area'].values,  test_dataset_['burned_area'].values, test_dataset_['departement'].values)
    
    res[obj2.name] = obj2

    new_cols.append('burnedarea-kmeans-5-Class-Dept')

    ################################## Pst Risk and Past burned area feature ########################################

    if meanSequence.sequences_month is None:
        meanSequence.compute_sequence_month(pd.concat([train_dataset, test_dataset]), 'bdiff')
    
    conv_types = ['cubic']

    kernels = ['Specialized']

    ###############################################################################

    n_clusters = 5

    for conv_type in conv_types:
        for kernel in kernels:
            print(f"Testing with convolution type: {conv_type}")
        
            train_col = f"nbsinisterDaily-kmeans-{n_clusters}-Class-Dept-{conv_type}-{kernel}-Past"
    
            # Sélection du préprocesseur
            preprocessor = PreprocessorConv(graph=graph, conv_type=conv_type, kernel=kernel, id_col=['month_non_encoder', 'graph_id'], persistence=True)

            # Définition de l'objet ScalerClassRisk
            class_risk = KMeansRisk(n_clusters=n_clusters)
            obj = ScalerClassRisk(
                col_id='departement',
                dir_output=dir_post_process,
                target='nbsinisterDaily',
                scaler=None,
                class_risk=class_risk,
                preprocessor=preprocessor
            )

            # Application du fit et prédictions
            obj.fit(
                train_dataset_['nbsinisterDaily'].values,
                train_dataset_['nbsinisterDaily'].values,
                train_dataset_['departement'].values,
                train_dataset_[['month_non_encoder', 'graph_id']].values
            )

            train_dataset_[train_col] = obj.predict(
                train_dataset_['nbsinisterDaily'].values,
                train_dataset_['nbsinisterDaily'].values,
                train_dataset_['departement'].values,
                train_dataset_[['month_non_encoder', 'graph_id']].values
            )

            val_dataset_[train_col] = obj.predict(
                val_dataset_['nbsinisterDaily'].values,
                val_dataset_['nbsinisterDaily'].values,
                val_dataset_['departement'].values,
                val_dataset_[['month_non_encoder', 'graph_id']].values
            )
            test_dataset_[train_col] = obj.predict(
                test_dataset_['nbsinisterDaily'].values,
                test_dataset_['nbsinisterDaily'].values,
                test_dataset_['departement'].values,
                test_dataset_[['month_non_encoder', 'graph_id']].values
            )

            train_col = f"burnedareaDaily-kmeans-{n_clusters}-Class-Dept-{conv_type}-{kernel}-Past"

            # Sélection du préprocesseur
            preprocessor = PreprocessorConv(graph=graph, conv_type=conv_type, kernel=kernel, id_col=['month_non_encoder', 'graph_id'], persistence=True)

            # Définition de l'objet ScalerClassRisk
            class_risk = KMeansRisk(n_clusters=n_clusters)
            obj = ScalerClassRisk(
                col_id='departement',
                dir_output=dir_post_process,
                target='burnedareaDaily',
                scaler=None,
                class_risk=class_risk,
                preprocessor=preprocessor
            )

            # Application du fit et prédictions
            obj.fit(
                train_dataset_['burnedareaDaily'].values,
                train_dataset_['burnedareaDaily'].values,
                train_dataset_['departement'].values,
                train_dataset_[['month_non_encoder', 'graph_id']].values
            )

            train_dataset_[train_col] = obj.predict(
                train_dataset_['burnedareaDaily'].values,
                train_dataset_['burnedareaDaily'].values,
                train_dataset_['departement'].values,
                train_dataset_[['month_non_encoder', 'graph_id']].values
            )

            val_dataset_[train_col] = obj.predict(
                val_dataset_['burnedareaDaily'].values,
                val_dataset_['burnedareaDaily'].values,
                val_dataset_['departement'].values,
                val_dataset_[['month_non_encoder', 'graph_id']].values
            )
            test_dataset_[train_col] = obj.predict(
                test_dataset_['burnedareaDaily'].values,
                test_dataset_['burnedareaDaily'].values,
                test_dataset_['departement'].values,
                test_dataset_[['month_non_encoder', 'graph_id']].values
            )

            res[obj.name] = deepcopy(obj)
            new_cols.append(train_col)

    print(f"Completed processing for convolution type: {conv_type} with kernel {kernel}")
    
    ################################################

    print(f'Post process Model -> {res}')

    train_dataset = train_dataset_
    val_dataset = val_dataset_
    test_dataset = test_dataset_

    return res, train_dataset, val_dataset, test_dataset, new_cols
