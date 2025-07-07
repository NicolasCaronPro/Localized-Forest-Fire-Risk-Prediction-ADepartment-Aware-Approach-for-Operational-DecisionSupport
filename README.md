### Abstract

Forest fire prediction involves estimating the likelihood of fire ignition or related risk levels in a specific area over a defined time period. With climate change intensifying fire behavior and frequency, accurate prediction has become one of the most pressing challenges in Artificial Intelligence (AI). Traditionally, fire ignition is approached as a binary classification task in the literature. However, this formulation oversimplifies the problem, especially from the perspective of end-users such as firefighters. In general, as is the case in France, firefighting units are organized by department, each with its terrain, climate conditions, and historical experience with fire events. Consequently, fire risk should be modeled in a way that is sensitive to local conditions and does not assume uniform risk across all regions. This paper proposes a new approach that tailors fire risk assessment to departmental contexts, offering more actionable and region-specific predictions for operational use. With this, we present the first national-scale AI benchmark for metropolitan France using state-of-the-art AI models on a relatively unexplored dataset. Finally, we offer a summary of important future works that should be taken into account.
Read the original paper on https://arxiv.org/abs/2506.04254


### Description

#### Database
The folder contains code for constructing the target and creating the 3D raster features.

How to use :
python generate_database.py -m True -t True -s True -r 2x2

#### models
The folder contains the model's code used in the article and the high-resolution image of the model's architecture.

### supplementary_materials.pdf
The folder contains additional figures, tables, and the full list of features used to train the models.

Python3.9 version
