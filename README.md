### Abstract

Forest fire prediction involves estimating the likelihood of fire ignition or related risk levels in a specific area over a defined time period. With climate change intensifying fire behavior and frequency, accurate prediction has become one of the most pressing challenges in Artificial Intelligence (AI). Traditionally, fire ignition is approached as a binary classification task in the literature. However, this formulation oversimplifies the problem, especially from the perspective of end-users such as firefighters. In general, as is the case in France, firefighting units are organized by department, each with its terrain, climate conditions, and historical experience with fire events. Consequently, fire risk should be modeled in a way that is sensitive to local conditions and does not assume uniform risk across all regions. This paper proposes a new approach that tailors fire risk assessment to departmental contexts, offering more actionable and region-specific predictions for operational use. With this, we present the first national-scale AI benchmark for metropolitan France using state-of-the-art AI models on a relatively unexplored dataset. Finally, we offer a summary of important future works that should be taken into account.
Read the original paper on https://arxiv.org/abs/2506.04254


### Description

#### Database
This folder contains the scripts used to create the input features and targets. Both are, firstly, converted to pickle file by departments to limit the memory usage in the process. To explore all features, you mainly need around 250 gigabytes of storage. Additionally, the data can be converted into a datacube (xarray) structured by latitude, longitude, and date.
To compute the features
```bash
python3.9 generate_database.py -m True -t True -s True -r 2x2
```

To compute the target for fire occurrence
```bash
python3.9 high_scale_database.py -r False -s firepoint -re 2x2 -d bdiff -se occurrence -od bdiff
```

**Contents**

* **Features**
  * `generate_database.py` – main driver that downloads and rasterises the data for each department.
  * `calendar_features.py` – generates calendar features such as public holidays and lockdown periods.
  * `download.py` – helpers to retrieve and convert geographic datasets (population, roads, elevation…).
  * `tools.py` – geospatial utilities and fire index calculations.
  * `dico_departements.py` – dictionaries mapping department codes to names and storing specific parameters.
  * `GEEExportData.ipynb` – notebook demonstrating how to export Landsat from Google Earth Engine, including cloud masking and index computation steps.
  * The code will generate a pickle file for each feature; files will be located in the `dir_raster` directory.
  * Features have been aggregated by departments (average, maximum, and minimum) (code in progress)
 
* **Automation**

| Feature        | Automation Description                                                                 |
|----------------|-----------------------------------------------------------------------------------------|
| Meteorological | OK   |
| Landsat       | Need GEE account |
| Landcover      | Need original sources                    |
| Elevation      | To Do    |
| Population     | OK            |
| Forest cover   | OK |
| Clay soil      |To do              |
| Calendar      | OK              |


#### EncoderAndClustering
This folder gathers utilities to encode categorical features and cluster time series.
These scripts rely on the pickle files generated during the dataset creation phase.

* `encoding.py` – prepares and encodes the variables and saves the results as pickle files.
* `time_series_clustering.py` – groups time series using the encoded data.

* **Targets**
  * Download the fire file on the BDIFF website : https://bdiff.agriculture.gouv.fr/. Select "Diffuser" then on "Ajouter un critère" select "Localisation" (Departement). We used data between 2017-06-12 and 2023-12-31
  * `high_scale_database.py` – script for generating the high-resolution probabilistic database.
  * `tools_functions.py` – utility functions used in the probabilistic pipeline.
  * `dico_departements.py` – departments lookup for target generation.
  * `check.ipynb` – notebook validating the target creation process.
  * `bdiff_plot.png` – example difference plot produced during validation.
  * `discretization.py` – discretisation and aggregation methods to build the supervised targets.
  * Targets have been summed by departments, taking the total number of fires (or burned area) in a day (code in progress)

#### Models
This folder implements the machine learning models and evaluation metrics used in the article.

* `dp_models.py` – PyTorch neural networks (GRU, LSTM and spatio‑temporal variants).
* `skl_models.py` – scikit-learn/XGBoost/LightGBM implementations.
* `score.py` – functions to compute metrics such as IoU and F1.
* `ModelArchitecture.drawio.png` – high-resolution diagram of the architecture.

### supplementary_materials.pdf
Contains the figures, tables and the full list of variables used to train the models.

### Root files

* `requirements.txt` – Python dependencies required to run the scripts.

Although this GitHub repository uses the generated data to study wildfire risk prediction, the variables can also be used in other areas of spatial analysis or risk management. It is possible to easily select specific departments for study in order to reduce processing time.

> **If you use our code, please cite:**
> 
> Caron, N., Guyeux, C., Noura, H., & Aynes, B. (2025). *Localized Forest Fire Risk Prediction: A Department-Aware Approach for Operational Decision Support*. arXiv:2506.04254. [https://arxiv.org/abs/2506.04254](https://arxiv.org/abs/2506.04254)

