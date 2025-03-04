import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data 
import os
from tqdm import tqdm
import deepchem as dc
import rdkit
from rdkit import Chem


class LoadDataset(Dataset):
    def __init__(self, root, raw_filename, transform=None, pre_transform=None):
        self.raw_filename = raw_filename
        super(LoadDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.raw_filename

    @property
    def processed_file_names(self):
        # self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        self.data = pd.read_excel(self.raw_paths[0]).reset_index()
        return [f"molecule_{i}.pt" for i in list(self.data.index)]

    def download(self):
        pass
    
    def process(self):  
        self.data = pd.read_excel(self.raw_paths[0]).reset_index()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True)
        for idx, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):

            mol = Chem.MolFromSmiles(row["standardized_smiles"])
            f = featurizer._featurize(mol)
            node_features = torch.tensor(f.node_features)
            edge_index = torch.tensor(f.edge_index)
            edge_attr = torch.tensor(f.edge_features)
 
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
            data.y = self._get_label(row["Label"])  
            data.smiles = row["standardized_smiles"]
            torch.save(data, os.path.join(self.processed_dir, f"molecule_{idx}.pt"))

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.float32)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f"molecule_{idx}.pt"))

# 预测时候的代码
def mol_to_graph_data_obj_simple(mol):
    """
    将RDKit分子对象转换为PyTorch Geometric的Data对象
    
    参数:
    mol (rdkit.Chem.rdchem.Mol): RDKit分子对象
    
    返回:
    torch_geometric.data.Data: 包含分子图结构的Data对象
    """
    import deepchem as dc
    import torch
    from torch_geometric.data import Data
    
    # 使用DeepChem的MolGraphConvFeaturizer进行特征化
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True)
    
    # 特征化分子
    f = featurizer._featurize(mol)
    
    # 提取节点特征、边索引和边特征
    node_features = torch.tensor(f.node_features)
    edge_index = torch.tensor(f.edge_index)
    edge_attr = torch.tensor(f.edge_features)
    
    # 创建Data对象
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    
    # 添加SMILES作为属性
    data.smiles = Chem.MolToSmiles(mol)
    
    return data

