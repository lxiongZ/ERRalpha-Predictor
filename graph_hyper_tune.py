import optuna
from optuna.samplers import TPESampler

import json
from optuna.trial import Trial
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import torch
from torch_geometric.loader import DataLoader
import numpy as np
import random
import os

from utils import LoadDataset
from torch_geometric.nn.models import AttentiveFP
from model import GraphTransformerModel, GINModel, GCNModel, GATModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True) 
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def split_rus_graph(dataset, test_size=0.1, val_size=0.1, random_state=42):
    train_g, temp_g = train_test_split(
        dataset, test_size=(test_size + val_size), random_state=6
    )
    val_g, test_g = train_test_split(
        temp_g, test_size=(test_size / (test_size + val_size)), random_state=6
    )
    X_train = np.arange(len(train_g)).reshape(-1, 1)
    y_train = np.array([data.y.item() for data in train_g])
    rus = RandomUnderSampler(random_state=random_state)
    X_samp, _ = rus.fit_resample(X_train, y_train)
    sampled_train_g = [train_g[i[0]] for i in X_samp]
    return sampled_train_g, val_g, test_g

def split_ros_graph(dataset, test_size=0.1, val_size=0.1, random_state=42):
    train_g, temp_g = train_test_split(
        dataset, test_size=(test_size + val_size), random_state=6
    )
    val_g, test_g = train_test_split(
        temp_g, test_size=(test_size / (test_size + val_size)), random_state=6
    )
    X_train = np.arange(len(train_g)).reshape(-1, 1)
    y_train = np.array([data.y.item() for data in train_g])
    ros = RandomOverSampler(random_state=random_state)
    X_samp, _ = ros.fit_resample(X_train, y_train)
    sampled_train_g = [train_g[i[0]] for i in X_samp]
    return sampled_train_g, val_g, test_g

def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x.float(), data.edge_index.long(), data.edge_attr.float(), data.batch.long())
        labels = data.y.float().to(device)
        loss = loss_fn(out.squeeze(), labels)  
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)

def validate_model(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data.x.float(), data.edge_index.long(), data.edge_attr.float(), data.batch.long())
            val_loss += loss_fn(out.squeeze(), data.y.float().to(device)).item()
    return val_loss / len(val_loader) 

def objective(trial: Trial, dataset, model_class, sampling_method):
    hidden_channels = trial.suggest_categorical('hidden_channels', [16, 32, 64, 128])
    num_layers = trial.suggest_int('num_layers', 1, 5) 
    dropout = trial.suggest_categorical('dropout', [0.2, 0.5])
    lr = trial.suggest_categorical('lr', [0.1, 0.01, 0.001, 0.0001])
    weight_decay = trial.suggest_categorical('weight_decay', [0.0001, 0.00001, 0.001])
    gamma = trial.suggest_categorical('gamma', [0.995, 0.9, 0.8, 0.5, 1])
    pos_weight = trial.suggest_float('pos_weight', 1.0, 2.0) 
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 100, 128]) 

    if model_class == GraphTransformerModel:
        n_heads = trial.suggest_int('n_heads', 1, 5)
        model = model_class(in_channels=32, hidden_channels=hidden_channels, out_channels=1,
                            edge_dim=11, num_layers=num_layers, dropout=dropout, n_heads=n_heads)
    elif model_class == AttentiveFP:
        num_timesteps = trial.suggest_int('num_timesteps', 1, 5)
        model = model_class(in_channels=32, hidden_channels=hidden_channels, out_channels=1,
                            edge_dim=11, num_layers=num_layers, dropout=dropout, num_timesteps=num_timesteps)
    else:
        model = model_class(in_channels=32, hidden_channels=hidden_channels, out_channels=1,
                            edge_dim=11, num_layers=num_layers, dropout=dropout)

    if sampling_method == 'rus':
        train_graphs, val_graphs, _ = split_rus_graph(dataset)
    else:
        train_graphs, val_graphs, _ = split_ros_graph(dataset)

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True) 
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device))

    best_loss = float('inf')
    early_stopping_counter = 0
    patience = 5

    for epoch in range(200):  
        if early_stopping_counter <= patience:

            _ = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            
            if (epoch + 1) % 5 == 0:
                val_loss = validate_model(model, val_loader, loss_fn, device)
                print(f"Epoch [{epoch + 1}/200] | Validation Loss {val_loss:.4f}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

            scheduler.step()
        else:
            print("Early stopping due to no improvement.")
            break

    return best_loss  

model_name_map = {
    'GraphTransformerModel': "GT",
    'GINModel': "GIN",
    'GCNModel': "GCN",
    'GATModel': "GAT",
    'AttentiveFP': "AFP"
}

def hyperparameter_search(datasets):
    best_params = {}
    for dataset_name, dataset in datasets.items():
        for sampling_method in ['rus', 'ros']:
            for model_class in [GraphTransformerModel, GINModel, GCNModel, GATModel, AttentiveFP]:
                sampler = TPESampler(seed=42)
                model_name = model_name_map[model_class.__name__]
                study_name = f"{dataset_name}_{sampling_method}_{model_name}"
                study = optuna.create_study(direction='minimize', study_name=study_name, sampler=sampler)
                study.optimize(lambda trial: objective(trial, dataset, model_class, sampling_method), n_trials=30) 
                best_params[study_name] = study.best_params

    with open('best_hyperparameters.json', 'w') as f:
        json.dump(best_params, f, indent=4)

set_seed(0)
datasets = { 
    'binder': LoadDataset(root='./datasets/binder_train', raw_filename='binder_train.xlsx'), 
    'agonist': LoadDataset(root='./datasets/agonist_train', raw_filename='agonist_train.xlsx'),
    'antagonist': LoadDataset(root='./datasets/antagonist_train', raw_filename='antagonist_train.xlsx')
}
hyperparameter_search(datasets)
