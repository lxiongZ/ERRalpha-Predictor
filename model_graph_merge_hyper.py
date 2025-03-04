import os
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix

import torch
from torch_geometric.loader import DataLoader

import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from utils import LoadDataset
from torch_geometric.nn.models import AttentiveFP
from model import GraphTransformerModel, GINModel, GCNModel, GATModel

import random
import json  

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


def split_external_graph(dataset):
    graphs = [data for data in dataset]
    return graphs


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

def train_model(model, train_loader, val_loader, optimizer, scheduler, loss_fn, device, patience=5):
    best_loss = float('inf')
    early_stopping_counter = 0
    best_model_state_dict = None
    best_metrics = None

    for epoch in range(200):  
        if early_stopping_counter <= patience:
            avg_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch + 1}/200], Loss: {avg_loss:.4f}")
            if (epoch + 1) % 5 == 0:
                val_loss, metrics = evaluate_model(model, val_loader, loss_fn, device)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_state_dict = model.state_dict()
                    best_metrics = metrics
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
            scheduler.step()
        else:
            print("Early stopping due to no improvement.")
            break
        
    model.load_state_dict(best_model_state_dict)
    return model, best_metrics  


def evaluate_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    y_pred = []
    y_true = []
    y_prob = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out = model(data.x.float(), data.edge_index.long(), data.edge_attr.float(), data.batch.long())
            prob = torch.sigmoid(out).squeeze().cpu().numpy()
            predicted = np.rint(prob).astype(int)
            y_pred.extend(predicted)
            y_true.extend(data.y.cpu().numpy())
            y_prob.extend(prob)
            total_loss += loss_fn(out.squeeze(), data.y.float().to(device)).item()
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    return total_loss / len(data_loader), metrics


def save_model_and_results(model, results, dataset_name, sampling_method, model_name, all_results):
    
    model_dir = f"graph_models/{dataset_name}/{sampling_method}/"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.pth")  
    torch.save(model.state_dict(), model_path)

    results_dir = f"graph_results/{dataset_name}/{sampling_method}/"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "results.xlsx")

    metrics_names = ['SE', 'SP', 'ACC', 'BA', 'MCC', 'AUC', 'PR_AUC']
    formatted_results = {
        'Model': model_name,
        **{name: f"{value:.3f}" for name, value in zip(metrics_names, results['best_metrics'])}
    }

    all_results['training'].append(formatted_results)
    all_results['test'].append({
        'Model': model_name,
        **{name: f"{value:.3f}" for name, value in zip(metrics_names, results['test_metrics'])}
    })
    all_results['external'].append({
        'Model': model_name,
        **{name: f"{value:.3f}" for name, value in zip(metrics_names, results['external_metrics'])}
    })

    with pd.ExcelWriter(results_path) as writer:
        pd.DataFrame(all_results['training']).to_excel(writer, sheet_name='Training', index=False)
        pd.DataFrame(all_results['test']).to_excel(writer, sheet_name='Test', index=False)
        pd.DataFrame(all_results['external']).to_excel(writer, sheet_name='External', index=False)

def train_with_best_params(datasets, external_datasets, best_params_file='best_hyperparameters.json'):
    with open(best_params_file, 'r') as f:
        best_params = json.load(f)

    model_name_map = {
        GraphTransformerModel: "GT",
        GINModel: "GIN",
        GCNModel: "GCN",
        GATModel: "GAT",
        AttentiveFP: "AFP"
    }

    for dataset_name, dataset in datasets.items():
        for sampling_method in ['rus', 'ros']:
            print("*"*100)
            print(f"Processing dataset: {dataset_name} with {sampling_method}")

            all_results = {
                'training': [],
                'test': [],
                'external': []
            }
            for model_class, model_name in model_name_map.items():  
                print(f"Training and evaluating {model_name} on {dataset_name} with {sampling_method}...")
                
                study_name = f"{dataset_name}_{sampling_method}_{model_name}"
                params = best_params[study_name]  
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

                if sampling_method == 'rus':
                    train_graphs, val_graphs, test_graphs = split_rus_graph(dataset)
                else:
                    train_graphs, val_graphs, test_graphs = split_ros_graph(dataset)

                train_loader = DataLoader(train_graphs, batch_size=params['batch_size'], shuffle=True)
                val_loader = DataLoader(val_graphs, batch_size=params['batch_size'], shuffle=False)
                test_loader = DataLoader(test_graphs, batch_size=params['batch_size'], shuffle=False)

                model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params['gamma'])
                loss_fn = torch.nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([params['pos_weight']], dtype=torch.float32).to(device))

                best_model, best_metrics = train_model(model, train_loader, val_loader, optimizer, scheduler, loss_fn,
                                                       device)

                _, test_metrics = evaluate_model(best_model, test_loader, loss_fn, device)

                external_graphs = split_external_graph(external_datasets[dataset_name])
                external_loader = DataLoader(external_graphs, batch_size=params['batch_size'], shuffle=False)
                _, external_metrics = evaluate_model(best_model, external_loader, loss_fn, device)

                model_results = {
                    'best_metrics': best_metrics,
                    'test_metrics': test_metrics,
                    'external_metrics': external_metrics
                }

                save_model_and_results(best_model, model_results, dataset_name, sampling_method, model_name,
                                       all_results)

set_seed(0)
datasets = {
    'binder': LoadDataset(root='./datasets/binder_train', raw_filename='binder_train.xlsx'),
    'agonist': LoadDataset(root='./datasets/agonist_train', raw_filename='agonist_train.xlsx'),
    'antagonist': LoadDataset(root='./datasets/antagonist_train', raw_filename='antagonist_train.xlsx')
}
external_datasets = {
    'binder': LoadDataset(root='./datasets/binder_external', raw_filename='binder_external.xlsx'),
    'agonist': LoadDataset(root='./datasets/agonist_external', raw_filename='agonist_external.xlsx'),
    'antagonist': LoadDataset(root='./datasets/antagonist_external', raw_filename='antagonist_external.xlsx')
}
train_with_best_params(datasets, external_datasets)

