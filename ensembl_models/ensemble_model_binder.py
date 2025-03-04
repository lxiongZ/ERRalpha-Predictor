import json
import os
import torch
import numpy as np
import pandas as pd
from joblib import load
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, matthews_corrcoef, \
                            roc_auc_score, precision_recall_curve, auc
from rdkit import Chem
from rdkit.Chem import Descriptors
import deepchem as dc

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import LoadDataset
from model import GraphTransformerModel, GINModel

from torch_geometric.nn.models import AttentiveFP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def split_data(df):
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    return X, y

def load_ml_model(model_path):
    return load(model_path)

def get_ml_predictions(model, X):
    return model.predict_proba(X)[:, 1]

def initialize_model(model_class, params):
    if model_class == GraphTransformerModel:
        model = model_class(in_channels=32, hidden_channels=params['hidden_channels'], out_channels=1,
                            edge_dim=11, num_layers=params['num_layers'], dropout=params['dropout'],
                            n_heads=params['n_heads'])
    elif model_class == AttentiveFP:
        model = model_class(in_channels=32, hidden_channels=params['hidden_channels'], out_channels=1,
                            edge_dim=11, num_layers=params['num_layers'], dropout=params['dropout'],
                            num_timesteps=params['num_timesteps'])
    else:
        model = model_class(in_channels=32, hidden_channels=params['hidden_channels'], out_channels=1,
                            edge_dim=11, num_layers=params['num_layers'], dropout=params['dropout'])
    return model

def load_model_weights(model, model_name):
    model_path = f'../graph_models/binder/ros/{model_name}.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate_model(model, test_loader):
    model.to(device)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x.float(), data.edge_index.long(), data.edge_attr.float(), data.batch.long())
            prob = torch.sigmoid(out).squeeze().cpu().numpy()
            all_probs.append(prob)
    return np.concatenate(all_probs)

def calculate_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    SE = TP / (TP + FN) if (TP + FN) != 0 else 0
    SP = TN / (TN + FP) if (TN + FP) != 0 else 0
    ACC = accuracy_score(y_true, y_pred)
    BA = balanced_accuracy_score(y_true, y_pred)
    MCC = matthews_corrcoef(y_true, y_pred)
    AUC = roc_auc_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    PR_AUC = auc(recall, precision)
    return SE, SP, ACC, BA, MCC, AUC, PR_AUC

with open('../best_hyperparameters.json', 'r') as f:
    best_params = json.load(f)

def calculate_descriptors(smiles_list):
    descriptors = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            desc_values = [desc(mol) for name, desc in Descriptors.descList]
            descriptors.append(desc_values)
    return descriptors

def calculate_mol2vec(smiles_list, pretrain_model_path):
    featurizer = dc.feat.Mol2VecFingerprint(pretrain_model_path=pretrain_model_path)
    mol2vec_fps = []
    for smiles in smiles_list:
        mol2vec_fp = featurizer.featurize(smiles)
        mol2vec_fp_list = mol2vec_fp[0].tolist()
        mol2vec_fps.append(mol2vec_fp_list)
    return mol2vec_fps

def add_smiles_labels(features, smiles, labels, feature_names):
    df_features = pd.DataFrame(features, columns=feature_names)
    df_features.insert(0, 'SMILES', smiles)
    df_features['Label'] = labels
    return df_features

def save_data(df, pretrain_model_path):
    smiles_list = df['standardized_smiles'].tolist()
    labels = df['Label'].tolist()

    descriptors = calculate_descriptors(smiles_list)
    mol2vec_fps = calculate_mol2vec(smiles_list, pretrain_model_path)

    desc_df = add_smiles_labels(descriptors, smiles_list, labels, [name for name, _ in Descriptors.descList])
    mol2vec_df = add_smiles_labels(mol2vec_fps, smiles_list, labels, [f'Mol2Vec_{i}' for i in range(len(mol2vec_fps[0]))])

    # Take the external validation set data as an example
    desc_folder_path = f'../datasets/binder_external/descriptors'
    mol2vec_folder_path = f'../datasets/binder_external/mol2vec'
    
    os.makedirs(desc_folder_path, exist_ok=True)
    os.makedirs(mol2vec_folder_path, exist_ok=True)

    desc_df.to_csv(os.path.join(desc_folder_path, 'descriptors.csv'), index=False)
    mol2vec_df.to_csv(os.path.join(mol2vec_folder_path, 'mol2vec.csv'), index=False)

pretrain_model_path = '../model_300dim.pkl'
df = pd.read_excel('../datasets/binder_external/raw/binder_external.xlsx')
save_data(df, pretrain_model_path)


df_mol2vec = pd.read_csv("../datasets/binder_external/mol2vec/mol2vec.csv")
X_mol2vec, y_mol2vec = split_data(df_mol2vec)

df_desc = pd.read_csv("../datasets/binder_external/descriptors/descriptors.csv")
X_desc, y_desc = split_data(df_desc)

selected_dl_models = [
    (GINModel, "GIN"),
    (GraphTransformerModel, "GT"),
    (AttentiveFP, "AFP")
]

dl_models = []
batch_size = None
for model_class, model_name in selected_dl_models:
    study_name = f"binder_ros_{model_name}"
    params = best_params[study_name]
    if batch_size is None:
        batch_size = params['batch_size']
    model = initialize_model(model_class, params)
    model = load_model_weights(model, model_name)
    model.to(device)
    dl_models.append(model)

external_graphs = LoadDataset(root='../datasets/binder_external', raw_filename='binder_external.xlsx')
external_loader = DataLoader(external_graphs, batch_size=batch_size, shuffle=False)

ml_models = [
    ('../ml_final_models/binder/rus/lgb+mol2vec.joblib', X_mol2vec),
    ('../ml_final_models/binder/rus/RF+mol2vec.joblib', X_mol2vec),
    ('../ml_final_models/binder/rus/lgb+descriptors.joblib', X_desc),
    ('../ml_final_models/binder/rus/RF+descriptors.joblib', X_desc),

    ('../ml_final_models/binder/ros/RF+mol2vec.joblib', X_mol2vec),
    ('../ml_final_models/binder/ros/RF+descriptors.joblib', X_desc)
]

ml_external_probs = [get_ml_predictions(load_ml_model(path), X) for path, X in ml_models]

dl_external_probs = [evaluate_model(model, external_loader) for model in dl_models]

all_probs = np.array(ml_external_probs + dl_external_probs).T
consensus_all_probs = np.mean(all_probs, axis=1)

threshold = 0.5
predicted = (consensus_all_probs >= threshold).astype(int)
SE, SP, ACC, BA, MCC, AUC, PR_AUC = calculate_metrics(y_mol2vec, predicted, consensus_all_probs) 

def save_results_to_excel(results, output_path):
    df_results = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_results.to_excel(output_path, index=False)

results = []
results.append({
    "model": "ERR_alpha_stack",
    "SE" : round(SE, 3),
    "SP" : round(SP, 3),
    "ACC" : round(ACC, 3),
    "BA" : round(BA, 3),
    "AUC": round(AUC, 3),
    "PR_AUC": round(PR_AUC, 3),
    "MCC": round(MCC, 3)
})

save_results_to_excel(results, "ERR_alpha_stack/results_binder.xlsx")
