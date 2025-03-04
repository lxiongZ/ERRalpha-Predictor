from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, AllChem
import pandas as pd
import os
import deepchem as dc


def calculate_descriptors(smiles_list):
    descriptors = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            desc_values = [desc(mol) for name, desc in Descriptors.descList]
            descriptors.append(desc_values)
    return descriptors

def calculate_maccs(smiles_list):
    maccs_fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            maccs_fps.append(list(maccs_fp))
    return maccs_fps

def calculate_rdk(smiles_list):
    rdk_fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            rdk_fp = Chem.RDKFingerprint(mol)
            rdk_fps.append(list(rdk_fp))
    return rdk_fps

def calculate_morgan(smiles_list, radius=2, nBits=2048):
    morgan_fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
            morgan_fps.append(list(morgan_fp))
    return morgan_fps

def calculate_mol2vec(smiles_list, pretrain_model_path):
    featurizer = dc.feat.Mol2VecFingerprint(pretrain_model_path=pretrain_model_path)
    mol2vec_fps = []
    for smiles in smiles_list:
        mol2vec_fp = featurizer.featurize(smiles)
        mol2vec_fp_list = mol2vec_fp[0].tolist()
        mol2vec_fps.append(mol2vec_fp_list)
    return mol2vec_fps

def save_to_csv(df, filename):
    df.to_csv(filename, index=False)

def process_dataset(base_dir, dataset_name, pretrain_model_path):
    train_file = f"{base_dir}/{dataset_name}_train/raw/{dataset_name}_train.xlsx"
    external_file = f"{base_dir}/{dataset_name}_external/raw/{dataset_name}_external.xlsx"
    
    for file_type, file_path in [("train", train_file), ("external", external_file)]:
        df = pd.read_excel(file_path)
        smiles_list = df['standardized_smiles'].tolist()
        labels = df['Label'].tolist()  
        
        descriptors = calculate_descriptors(smiles_list)
        maccs_fps = calculate_maccs(smiles_list)
        rdk_fps = calculate_rdk(smiles_list)
        morgan_fps = calculate_morgan(smiles_list)
        mol2vec_fps = calculate_mol2vec(smiles_list, pretrain_model_path)

        def add_smiles_and_labels(features, smiles, labels, feature_names): 
            df_features = pd.DataFrame(features, columns=feature_names)
            df_features.insert(0, 'SMILES', smiles)  
            df_features['Label'] = labels  
            return df_features
        
        os.makedirs(f"{base_dir}/{dataset_name}_{file_type}/descriptors", exist_ok=True)
        os.makedirs(f"{base_dir}/{dataset_name}_{file_type}/maccs", exist_ok=True)
        os.makedirs(f"{base_dir}/{dataset_name}_{file_type}/rdk", exist_ok=True)
        os.makedirs(f"{base_dir}/{dataset_name}_{file_type}/morgan", exist_ok=True)
        os.makedirs(f"{base_dir}/{dataset_name}_{file_type}/mol2vec", exist_ok=True)

        descriptor_names = [name for name, desc in Descriptors.descList]
        maccs_names = [f"MACCS_{i}" for i in range(len(maccs_fps[0]))]
        rdk_names = [f"RDK_{i}" for i in range(len(rdk_fps[0]))]
        morgan_names = [f"Morgan_{i}" for i in range(len(morgan_fps[0]))]
        mol2vec_names = [f"Mol2Vec_{i}" for i in range(len(mol2vec_fps[0]))]

        save_to_csv(add_smiles_and_labels(descriptors, smiles_list, labels, descriptor_names), 
                    f"{base_dir}/{dataset_name}_{file_type}/descriptors/descriptors.csv")
        
        save_to_csv(add_smiles_and_labels(maccs_fps, smiles_list, labels, maccs_names), 
                    f"{base_dir}/{dataset_name}_{file_type}/maccs/maccs.csv")
        
        save_to_csv(add_smiles_and_labels(rdk_fps, smiles_list, labels, rdk_names), 
                    f"{base_dir}/{dataset_name}_{file_type}/rdk/rdk.csv")
        
        save_to_csv(add_smiles_and_labels(morgan_fps, smiles_list, labels, morgan_names), 
                    f"{base_dir}/{dataset_name}_{file_type}/morgan/morgan.csv")
        
        save_to_csv(add_smiles_and_labels(mol2vec_fps, smiles_list, labels, mol2vec_names), 
                    f"{base_dir}/{dataset_name}_{file_type}/mol2vec/mol2vec.csv")

def main():
    base_dir = "datasets"
    datasets = ["binder", "antagonist", "agonist"]
    pretrain_model_path = 'model_300dim.pkl'
    for dataset_name in datasets:
        process_dataset(base_dir, dataset_name, pretrain_model_path)

if __name__ == "__main__":
    main()
    print("success")