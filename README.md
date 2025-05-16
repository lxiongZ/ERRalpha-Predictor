# ERRalpha-Predictor
A Framework of Ensemble Models for Predicting ERRα Binders, Antagonists, and Agonists Using Artificial Intelligence

![](https://github.com/lxiongZ/ERRalpha-Predictor/blob/main/workflow.png)

## Overview:

- `datasets/` contains the three dataset files ("xxx_train" used for 8:1:1 data splitting, "xxx_external" used for external validation);
- `ensemble_models/` take the external validation set of three datasets as examples to use ERRα-Predictor;
- `graph_models/` contains the final selected GNN models for the three datasets;
- `ml_final_models/` contains the final selected ML models for three datasets;
- `mmpa_and_representative_substructure/` contains reference repository code
- `streamlit/` contains local web deployment of ERRα-Predictor;
- `supplementary)information/` contains supplementary materials;
- `ml_best_hyperparameters.pkl` optimal hyperparameters for ML models;
- `best_hyperparameters.json` optimal hyperparameters for GNN models;
- `GNNExplainer_viz.ipynb` visualization of GNN models

## Conda environments:

python = 3.7  
deepchem = 2.7.1  
imbalanced-learn = 0.7.0  
lightgbm = 4.5.0  
numpy = 1.21.6  
optuna = 4.0.0  
pytorch = 1.13.1  
pytorch-cuda = 11.7  
rdkit = 2023.3.2  
scikit-learn = 1.0.2  
torch-geometric = 2.3.1  
xgboost = 1.6.2  
mol2vec = 0.2.2  
openpyxl = 3.1.3  
streamlit = 1.23.1  
streamlit-ketcher = 0.0.1  

## Experiments:

### ML models

You need to download the mol2vec pkl file from [here](https://github.com/samoturk/mol2vec/blob/master/examples/models/model_300dim.pkl) first,
then for the three datasets, run the machine learning codes `molecular_representation.py` and `model_ml_construct.py`, get representation and model training results respectively.

### GNN models

Given the json file of the best hyperparameters, you only need to run `model_graph_merge_hyper.py`;
if you want to try this hyperparameter exploration process, run `graph_hyper_tune.py`.

### Ensemble models
You can look at the examples in `ensemble_models`：for binders, antagonists, and agonists, ERRα-Predictor was used to predict them respectively.

## Usage:
If you want to use ERRα-Predictor, after configuring the required Python environment and downloading the [mol2vec](https://github.com/samoturk/mol2vec/blob/master/examples/models/model_300dim.pkl) pkl file, it is recommended that you enter the `streamlit` for local web deployment.

Use the command:

```
cd streamlit
streamlit run app.py
```

The web page incorporates functionalities for data retrieval and prediction. The former involves querying the collected dataset of ERRα ligands, while the latter utilizes ERRα-Predictor models to perform predictions.  
Users can choose to submit either a single SMILES string or multiple SMILES strings for querying or prediction, and the results can be downloaded.

![](https://github.com/lxiongZ/ERRalpha-Predictor/blob/main/streamlit/Schematic%20diagram.png)

