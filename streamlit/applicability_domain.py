import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity

# Applicability domain parameters for each model
AD_PARAMS = {
    'Binder': {
        'Z': 2.8,
        'k': 2,
        'fingerprint_type': 'RDKit',
        'mean_similarity': 0.10197498266147081,
        'std_similarity': 0.08904234223682518
    },
    'Antagonist': {
        'Z': 3.0,
        'k': 4,
        'fingerprint_type': 'RDKit',
        'mean_similarity': 0.10079366503284654,
        'std_similarity': 0.08908573556246693
    },
    'Agonist': {
        'Z': 3.4,
        'k': 3,
        'fingerprint_type': 'Morgan',
        'mean_similarity': 0.08409199107685085,
        'std_similarity': 0.06225916599159844
    }
}

# Load training molecules and fingerprints
def load_training_data(model_type):
    """Load training data and generate fingerprints"""
    # Load training set based on model type
    train_file_path = f'data/train_{model_type.lower()}.xlsx'
    train_df = pd.read_excel(train_file_path)
    train_smiles_list = train_df['SMILES'].tolist()
    
    # Generate training set molecule objects
    train_molecules = [Chem.MolFromSmiles(smiles) for smiles in train_smiles_list if Chem.MolFromSmiles(smiles)]
    
    # Choose fingerprint type based on model type
    fingerprint_type = AD_PARAMS[model_type]['fingerprint_type']
    if fingerprint_type == 'Morgan':
        train_fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in train_molecules]
    else:  # RDKit
        train_fingerprints = [Chem.RDKFingerprint(mol) for mol in train_molecules]
    
    return train_fingerprints

# Calculate applicability domain
def calculate_applicability_domain(smiles_list, model_type):
    """Determine if molecules are within applicability domain"""
    # Get AD parameters
    params = AD_PARAMS[model_type]
    Z = params['Z']
    k = params['k']
    mean_similarity = params['mean_similarity']
    std_similarity = params['std_similarity']
    fingerprint_type = params['fingerprint_type']
    
    # Calculate threshold
    DT = mean_similarity + Z * std_similarity
    
    # Load training set fingerprints
    train_fingerprints = load_training_data(model_type)
    
    # Generate test molecule fingerprints
    test_molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list if Chem.MolFromSmiles(smiles)]
    
    # Ensure molecule list and SMILES list lengths match
    if len(test_molecules) != len(smiles_list):
        valid_indices = [i for i, smiles in enumerate(smiles_list) if Chem.MolFromSmiles(smiles)]
        _ = [smiles_list[i] for i in valid_indices] 
    else:
        valid_indices = list(range(len(smiles_list)))
        _ = smiles_list
    
    # Generate test molecule fingerprints
    if fingerprint_type == 'Morgan':
        test_fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in test_molecules]
    else:  # RDKit
        test_fingerprints = [Chem.RDKFingerprint(mol) for mol in test_molecules]
    
    # Determine if each molecule is within AD
    domain_status = []
    for i, test_fp in enumerate(test_fingerprints):
        # Calculate similarities with training set
        similarities = [TanimotoSimilarity(test_fp, train_fp) for train_fp in train_fingerprints]
        most_similar_indices = np.argsort(similarities)[-k:]  # Find k most similar 
        most_similar_values = [similarities[idx] for idx in most_similar_indices]
        
        # Check if any similarity is below DT
        if np.any(np.array(most_similar_values) < DT):  # Outside AD if any below DT
            domain_status.append('Outside AD')
        else:
            domain_status.append('Inside AD')
    
    # Create results dictionary for all SMILES 
    results = {}
    for i, smiles in enumerate(smiles_list):
        if i in valid_indices:
            idx = valid_indices.index(i)
            results[smiles] = domain_status[idx]
        else: # 对于无效的SMILES
            results[smiles] = 'Invalid SMILES'
    
    return results