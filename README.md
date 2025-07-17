### Abstract

Forest fire prediction involves estimating the likelihood of fire ignition or related risk levels in a specific area over a defined time period. With climate change intensifying fire behavior and frequency, accurate prediction has become one of the most pressing challenges in Artificial Intelligence (AI). Traditionally, fire ignition is approached as a binary classification task in the literature. However, this formulation oversimplifies the problem, especially from the perspective of end-users such as firefighters. In general, as is the case in France, firefighting units are organized by department, each with its terrain, climate conditions, and historical experience with fire events. Consequently, fire risk should be modeled in a way that is sensitive to local conditions and does not assume uniform risk across all regions. This paper proposes a new approach that tailors fire risk assessment to departmental contexts, offering more actionable and region-specific predictions for operational use. With this, we present the first national-scale AI benchmark for metropolitan France using state-of-the-art AI models on a relatively unexplored dataset. Finally, we offer a summary of important future works that should be taken into account.
Read the original paper on https://arxiv.org/abs/2506.04254


### Description

#### Database
This folder contains the scripts used to create the input features and targets.
To build the full dataset you can run for instance:

```bash
python generate_database.py -m True -t True -s True -r 2x2
```

**Contents**

* **Features**
  * `generate_database.py` – main driver that downloads and rasterises the data for each department.
  * `calendar_features.py` – generates calendar features such as public holidays and lockdown periods.
  * `download.py` – helpers to retrieve and convert geographic datasets (population, roads, elevation…).
  * `tools.py` – geospatial utilities and fire index calculations.
  * `dico_departements.py` – dictionaries mapping department codes to names and storing specific parameters.
  * The code will generate a pickle file for each feature, files will be located in the `dir_raster` directory.

* **Targets**
  * Download the fire file on the BDIFF website : https://bdiff.agriculture.gouv.fr/. Select "Diffuser" then on "Ajouter un critère" select "Localisation" (Departement). We use data between 2017-06-12 and 2023-12-31
  * `discretization.py` – discretisation and aggregation methods to build the supervised targets.
  * `high_scale_database.py` – script for generating the high-resolution probabilistic database.
  * `tools_functions.py` – utility functions used in the probabilistic pipeline.
  * `dico_departements.py` – departments lookup for target generation.
  * `check.ipynb` – notebook validating the target creation process.
  * `bdiff_plot.png` – example difference plot produced during validation.

#### models
This folder implements the machine learning models and evaluation metrics used in the article.

* `dp_models.py` – PyTorch neural networks (GRU, LSTM and spatio‑temporal variants).
* `skl_models.py` – scikit-learn/XGBoost/LightGBM implementations.
* `score.py` – functions to compute metrics such as IoU and F1.
* `ModelArchitecture.drawio.png` – high resolution diagram of the architecture.

### supplementary_materials.pdf
Contains the figures, tables and the full list of variables used to train the models.

### Root files

* `requirements.txt` – Python dependencies required to run the scripts.
