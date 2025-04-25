# ERRα Prediction Web Application
import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import json
from joblib import load
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import inchi
import deepchem as dc
import sys
import warnings
import io
from PIL import Image

# Add Ketcher import
from streamlit_ketcher import st_ketcher

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gensim.similarities")

# Import cleaning functions
from clean import choose_standardize

# Add project root directory to path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import mol_to_graph_data_obj_simple
from model import GraphTransformerModel, GINModel, GCNModel
from torch_geometric.nn.models import AttentiveFP

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load best hyperparameters
with open('../best_hyperparameters.json', 'r') as f:
    best_params = json.load(f)

# Generate InChI Key from SMILES
def smiles_to_inchi_key(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        inchi_key = inchi.MolToInchiKey(mol)
        return inchi_key
    except Exception as e:
        st.error(f"Error processing SMILES {smiles}: {e}")
        return None

# Load database
@st.cache_data
def load_database():
    try:
        db_path = "data/label_data_with_inchikey.xlsx"
        df = pd.read_excel(db_path)
        # Ensure database has required columns
        if 'SMILES' not in df.columns or 'InChI_Key' not in df.columns or 'Category' not in df.columns:
            st.error("Invalid database format. Must contain 'SMILES', 'InChI_Key' and 'Category' columns")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return None

# Search database by InChI Key
def search_by_inchi_key(inchi_key, database):
    if inchi_key is None:
        return None
    
    # Find matching records
    matches = database[database['InChI_Key'] == inchi_key]
    return matches if not matches.empty else None

# Load ML model    
def load_ml_model(model_path):
    return load(model_path)

# Get ML model predictions
def get_ml_predictions(model, X):
    return model.predict_proba(X)[:, 1]

# Initialize model
def initialize_model(model_class, params, model_type):
    # Print initialization info
    print(f"Initializing {model_type} prediction model")

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

# Load DL model weights
def load_model_weights(model, model_name, model_type):
    model_path = f'../graph_models/{model_type}/ros/{model_name}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Evaluate DL model
def evaluate_model(model, data_loader):
    model.to(device)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out = model(data.x.float(), data.edge_index.long(), data.edge_attr.float(), data.batch.long())
            prob = torch.sigmoid(out).squeeze().cpu().numpy()
            
            # Handle single prediction case (scalar result)
            if np.isscalar(prob):
                prob = np.array([prob])
            # Handle batch prediction case (ensure array)
            elif isinstance(prob, np.ndarray) and prob.ndim == 0:
                prob = np.array([prob.item()])
                
            all_probs.append(prob)
    
    # Return directly if single batch and single prediction
    if len(all_probs) == 1 and len(all_probs[0]) == 1:
        return all_probs[0]
    
    # Otherwise concatenate all batch results
    return np.concatenate(all_probs)

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
        try:
            mol2vec_fp = featurizer.featurize(smiles)
            mol2vec_fp_list = mol2vec_fp[0].tolist()
            mol2vec_fps.append(mol2vec_fp_list)
        except:
            # Add zero vector if SMILES is invalid
            mol2vec_fps.append([0] * 300)  # Mol2Vec is typically 300-dimensional
    return mol2vec_fps

# Add SMILES and labels to feature dataframe
def add_smiles_labels(features, smiles, feature_names):
    df_features = pd.DataFrame(features, columns=feature_names)
    df_features.insert(0, 'SMILES', smiles)
    return df_features

# Clean SMILES strings
def clean_smiles_list(smiles_list):
    cleaned_smiles = []
    for smiles in smiles_list:
        try:
            # Use choose_standardize function for SMILES cleaning
            cleaned = choose_standardize(smiles)
            cleaned_smiles.append(cleaned)
        except Exception as e:
            st.warning(f"Error cleaning SMILES: {smiles}, Error: {e}")
            cleaned_smiles.append(smiles)  # Use original SMILES if cleaning fails
    return cleaned_smiles

# Comprehensive data preparation for prediction
def prepare_data_for_prediction(smiles_list, pretrain_model_path):
    # Ensure smiles_list is a list even for single SMILES
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]
    
    # First clean SMILES
    cleaned_smiles = clean_smiles_list(smiles_list)

    # Calculate descriptors and Mol2Vec features
    descriptors = calculate_descriptors(cleaned_smiles)
    mol2vec_fps = calculate_mol2vec(cleaned_smiles, pretrain_model_path)
    
    # Create dataframes
    desc_df = add_smiles_labels(descriptors, cleaned_smiles, [name for name, _ in Descriptors.descList])
    mol2vec_df = add_smiles_labels(mol2vec_fps, cleaned_smiles, [f'Mol2Vec_{i}' for i in range(len(mol2vec_fps[0]))])
    
    # Prepare graph data
    graph_data_list = []
    for smiles in cleaned_smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                graph_data = mol_to_graph_data_obj_simple(mol)
                graph_data_list.append(graph_data)
        except:
            st.error(f"Cannot process SMILES: {smiles}")
    
    return desc_df, mol2vec_df, graph_data_list, cleaned_smiles

# Predict agonist code
def predict_agonist(smiles_list):
    pretrain_model_path = '../model_300dim.pkl'
    
    # Prepare data
    desc_df, mol2vec_df, graph_data_list, cleaned_smiles = prepare_data_for_prediction(smiles_list, pretrain_model_path)
    
    # Prepare ML model input
    X_mol2vec = mol2vec_df.iloc[:, 1:]
    X_desc = desc_df.iloc[:, 1:]
    
    # Load ML models
    svm_model = load_ml_model('../ml_final_models/agonist/ros/SVM+mol2vec.joblib')
    rf_model = load_ml_model('../ml_final_models/agonist/ros/RF+descriptors.joblib')
    
    # Get ML model predictions
    svm_probs = get_ml_predictions(svm_model, X_mol2vec)
    rf_probs = get_ml_predictions(rf_model, X_desc)
    
    # Load DL models
    selected_models = [
        (GraphTransformerModel, "GT"),
        (GCNModel, "GCN"),
        (AttentiveFP, "AFP")
    ]
    
    dl_models = []
    batch_size = 32
    
    for model_class, model_name in selected_models:
        study_name = f"agonist_ros_{model_name}"
        params = best_params[study_name]
        model = initialize_model(model_class, params, "agonist")
        model = load_model_weights(model, model_name, "agonist")
        model.to(device)
        dl_models.append(model)
    
    # Create data loader
    data_loader = DataLoader(graph_data_list, batch_size=batch_size, shuffle=False) 

    # Get DL model predictions
    dl_probs = [evaluate_model(model, data_loader) for model in dl_models]
    
    # Merge all predictions
    all_probs = [svm_probs, rf_probs] + dl_probs
    all_probs = np.array(all_probs).T
    
    # Calculate ensemble predictions
    consensus_probs = np.mean(all_probs, axis=1)
    
    # Apply threshold
    threshold = 0.4
    predictions = (consensus_probs >= threshold).astype(int)
    
    return predictions, consensus_probs, cleaned_smiles

# Predict antagonist code
def predict_antagonist(smiles_list):
    pretrain_model_path = '../model_300dim.pkl'
    
    # Prepare data
    desc_df, mol2vec_df, graph_data_list, cleaned_smiles = prepare_data_for_prediction(smiles_list, pretrain_model_path)
    
    # Prepare ML model input
    X_mol2vec = mol2vec_df.iloc[:, 1:]
    X_desc = desc_df.iloc[:, 1:]
    
    # Load ML models
    ml_models = [
        ('../ml_final_models/antagonist/rus/lgb+mol2vec.joblib', X_mol2vec),
        ('../ml_final_models/antagonist/rus/RF+mol2vec.joblib', X_mol2vec),
        ('../ml_final_models/antagonist/rus/lgb+descriptors.joblib', X_desc),
        ('../ml_final_models/antagonist/rus/RF+descriptors.joblib', X_desc),
        ('../ml_final_models/antagonist/ros/RF+mol2vec.joblib', X_mol2vec),
        ('../ml_final_models/antagonist/ros/RF+descriptors.joblib', X_desc)
    ]
    
    # Get ML model predictions
    ml_probs = [get_ml_predictions(load_ml_model(path), X) for path, X in ml_models]
    
    # Load DL models
    selected_dl_models = [
        (GINModel, "GIN"),
        (GCNModel, "GCN"),
        (AttentiveFP, "AFP")
    ]
    
    dl_models = []
    batch_size = 32
    
    for model_class, model_name in selected_dl_models:
        study_name = f"antagonist_ros_{model_name}"
        params = best_params[study_name]
        model = initialize_model(model_class, params, "antagonist")
        model = load_model_weights(model, model_name, "antagonist")
        model.to(device)
        dl_models.append(model)
    
    # Create data loader
    data_loader = DataLoader(graph_data_list, batch_size=batch_size, shuffle=False)
    
    # Get DL model predictions
    dl_probs = [evaluate_model(model, data_loader) for model in dl_models]
    
    # Merge all predictions
    all_probs = np.array(ml_probs + dl_probs).T
    
    # Calculate ensemble predictions
    consensus_probs = np.mean(all_probs, axis=1)
    
    # Apply threshold
    threshold = 0.55
    predictions = (consensus_probs >= threshold).astype(int)
    
    return predictions, consensus_probs, cleaned_smiles

def predict_binder(smiles_list):
    pretrain_model_path = '../model_300dim.pkl'
    
    # Prepare data
    desc_df, mol2vec_df, graph_data_list, cleaned_smiles = prepare_data_for_prediction(smiles_list, pretrain_model_path)
    
    # Prepare ML model input
    X_mol2vec = mol2vec_df.iloc[:, 1:]
    X_desc = desc_df.iloc[:, 1:]
    
    # Load ML models
    ml_models = [
        ('../ml_final_models/binder/rus/lgb+mol2vec.joblib', X_mol2vec),
        ('../ml_final_models/binder/rus/RF+mol2vec.joblib', X_mol2vec),
        ('../ml_final_models/binder/rus/lgb+descriptors.joblib', X_desc),
        ('../ml_final_models/binder/rus/RF+descriptors.joblib', X_desc),
        ('../ml_final_models/binder/ros/RF+mol2vec.joblib', X_mol2vec),
        ('../ml_final_models/binder/ros/RF+descriptors.joblib', X_desc)
    ]
    
    # Get ML model predictions
    ml_probs = [get_ml_predictions(load_ml_model(path), X) for path, X in ml_models]
    
    # Load DL models
    selected_dl_models = [
        (GINModel, "GIN"),
        (GraphTransformerModel, "GT"),
        (AttentiveFP, "AFP")
    ]
    
    dl_models = []
    batch_size = 32
    
    for model_class, model_name in selected_dl_models:
        study_name = f"binder_ros_{model_name}"
        params = best_params[study_name]
        model = initialize_model(model_class, params, "binder")
        model = load_model_weights(model, model_name, "binder")
        model.to(device)
        dl_models.append(model)
    
    # Create data loader
    data_loader = DataLoader(graph_data_list, batch_size=batch_size, shuffle=False)
    
    # Get DL model predictions
    dl_probs = [evaluate_model(model, data_loader) for model in dl_models]
    
    # Merge all predictions
    all_probs = np.array(ml_probs + dl_probs).T
    
    # Calculate ensemble predictions
    consensus_probs = np.mean(all_probs, axis=1)
    
    # Apply threshold
    threshold = 0.5
    predictions = (consensus_probs >= threshold).astype(int)
    
    return predictions, consensus_probs, cleaned_smiles

# Database search function
def search_database(smiles_list):
    # Ensure smiles_list is a list
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]
    
    # Clean SMILES
    cleaned_smiles = clean_smiles_list(smiles_list)
    
    # Load database
    database = load_database()
    if database is None:
        return None
    
    results = []
    for smiles in cleaned_smiles:
        # Generate InChI Key
        inchi_key = smiles_to_inchi_key(smiles)
        if inchi_key:
            # Search in database
            matches = search_by_inchi_key(inchi_key, database)
            if matches is not None:
                for _, row in matches.iterrows():
                    results.append({
                        'Original SMILES': smiles,
                        'Cleaned SMILES': smiles,
                        'Database SMILES': row['SMILES'],
                        'Category': row['Category']
                    })
            else:
                results.append({
                    'Original SMILES': smiles,
                    'Cleaned SMILES': smiles,
                    'Database SMILES': 'Not Found',
                    'Category': 'Unknown'
                })
        else:
            results.append({
                'Original SMILES': smiles,
                'Cleaned SMILES': smiles,
                'Database SMILES': 'Invalid SMILES',
                'Category': 'Unknown'
            })
    
    return pd.DataFrame(results)

# Display molecular structure
def display_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol, size=(400, 300))
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return buf.getvalue()
        return None
    except:
        return None

# Streamlit application
def main():
    st.title("ERRα-Predictor")
    
    # Display schematic diagram
    st.image("Schematic diagram.png", caption="Schematic diagram of ERRα-Predictor", use_column_width=True)
    
    # Sidebar options
    st.sidebar.title("Function Selection")
    app_mode = st.sidebar.selectbox("Select Function", ["Prediction", "Database Search"])
    
    # Prediction and search, default is prediction
    if app_mode == "Prediction":
        prediction_type = st.sidebar.radio("Select Prediction Type", ["Single SMILES", "Multiple SMILES"])
        model_type = st.sidebar.selectbox("Select ERRα Type", ["Binder", "Antagonist", "Agonist"])
        
        # Single SMILES prediction
        if prediction_type == "Single SMILES":
            st.header("Single SMILES Prediction")
            
            # Add input method selection
            input_method = st.radio("Choose input method", ["Draw molecule", "Enter SMILES"])
            
            if input_method == "Draw molecule":
                # Use Ketcher editor
                st.subheader("Draw your molecule")
                selected = ""  # Default empty structure
                selected = st_ketcher(selected)  # Display Ketcher editor
                
                # If user draws a molecule, convert to SMILES
                if selected:
                    smiles_input = selected
                    st.write(f"Generated SMILES: {smiles_input}")
                else:
                    smiles_input = ""
            else:
                # Original SMILES input box
                smiles_input = st.text_input("Enter SMILES string", "")
            
            if st.button("Predict"):
                if smiles_input:
                    try:
                        mol = Chem.MolFromSmiles(smiles_input)
                        if mol:  # If SMILES is valid
                            with st.spinner("Predicting..."):
                                if model_type == "Agonist":
                                    predictions, probs, cleaned_smiles = predict_agonist([smiles_input])
                                elif model_type == "Antagonist":
                                    predictions, probs, cleaned_smiles = predict_antagonist([smiles_input])
                                else:  # Binder
                                    predictions, probs, cleaned_smiles = predict_binder([smiles_input])
                            
                            # Display results
                            st.subheader("Prediction Results")
                            result = "Active" if predictions[0] == 1 else "Inactive"
                            st.write(f"Predicted Class: {result}")
                            st.write(f"Prediction Probability: {probs[0]:.4f}")
                            
                            # Display original and cleaned SMILES
                            if smiles_input != cleaned_smiles[0]:
                                st.write(f"Original SMILES: {smiles_input}")
                                st.write(f"Cleaned SMILES: {cleaned_smiles[0]}")
                            
                            # Visualize molecule
                            st.subheader("Molecular Structure")
                            mol_img = display_molecule(cleaned_smiles[0])
                            if mol_img:
                                st.image(mol_img)
                        else:
                            st.error("Invalid SMILES string, please check your input")
                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
                else:
                    st.warning("Please enter a SMILES string or draw a molecule") 
        
        # Multiple SMILES prediction
        else:  # Multiple SMILES prediction
            st.header("Multiple SMILES Prediction")
            
            # Provide two input methods
            input_method = st.radio("Select Input Method", ["Text Input", "CSV File Upload"])
            
            if input_method == "Text Input":
                smiles_text = st.text_area("Enter multiple SMILES strings (one per line)", "")
                
                if st.button("Predict"):
                    if smiles_text:
                        smiles_list = [s.strip() for s in smiles_text.split("\n") if s.strip()]
                        if smiles_list:
                            with st.spinner("Predicting..."):
                                try:
                                    if model_type == "Agonist":
                                        predictions, probs, cleaned_smiles = predict_agonist(smiles_list)
                                    elif model_type == "Antagonist":
                                        predictions, probs, cleaned_smiles = predict_antagonist(smiles_list)
                                    else:  # Binder
                                        predictions, probs, cleaned_smiles = predict_binder(smiles_list)
                                    
                                    # Display results
                                    results_df = pd.DataFrame({
                                        "Original SMILES": smiles_list,
                                        "Cleaned SMILES": cleaned_smiles,
                                        "Predicted Class": ["Active" if p == 1 else "Inactive" for p in predictions],
                                        "Prediction Probability": probs
                                    })
                                    st.dataframe(results_df)
                                    
                                    # Provide download option
                                    csv = results_df.to_csv(index=False)
                                    st.download_button(
                                        label="Download Predictions",
                                        data=csv,
                                        file_name=f"ERRa_{model_type}_predictions.csv",
                                        mime="text/csv"
                                    )
                                except Exception as e:
                                    st.error(f"Error during processing: {str(e)}")
                        else:
                            st.warning("Please enter at least one SMILES string")
                    else:
                        st.warning("Please enter SMILES strings")
            
            # Upload CSV file prediction
            else:  # CSV file upload
                st.write("Upload a CSV file containing SMILES (must have a 'SMILES' column)")
                uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
                
                if uploaded_file is not None:
                    try:
                        df = pd.read_csv(uploaded_file)
                        if "SMILES" in df.columns:
                            st.write(f"Loaded {len(df)} compounds")
                            st.dataframe(df.head())
                            
                            if st.button("Predict"):
                                smiles_list = df["SMILES"].tolist()
                                with st.spinner("Predicting..."):
                                    try:
                                        if model_type == "Agonist":
                                            predictions, probs, cleaned_smiles = predict_agonist(smiles_list)
                                        elif model_type == "Antagonist":
                                            predictions, probs, cleaned_smiles = predict_antagonist(smiles_list)
                                        else:  # Binder
                                            predictions, probs, cleaned_smiles = predict_binder(smiles_list)
                                        
                                        # Add results to original dataframe
                                        df["Cleaned SMILES"] = cleaned_smiles
                                        df["Predicted Class"] = ["Active" if p == 1 else "Inactive" for p in predictions]
                                        df["Prediction Probability"] = probs
                                        
                                        # Display results
                                        st.dataframe(df)
                                        
                                        # Provide download option
                                        csv = df.to_csv(index=False)
                                        st.download_button(
                                            label="Download Predictions",
                                            data=csv,
                                            file_name=f"ERRa_{model_type}_predictions.csv",
                                            mime="text/csv"
                                        )
                                    except Exception as e:
                                        st.error(f"Error during processing: {str(e)}")
                        else:
                            st.error("CSV file does not contain a 'SMILES' column")
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
    
    # Data Retrieval Function
    else:  # Data Retrieval Function
        st.header("Data Retrieval Function")
        search_type = st.radio("Select Search Type", ["Single SMILES Search", "Multiple SMILES Search"])
        
        # Single SMILES search
        if search_type == "Single SMILES Search":
            st.header("Single SMILES Search")
            
            # Add input method selection
            input_method = st.radio("Choose input method", ["Draw molecule", "Enter SMILES"])
            
            if input_method == "Draw molecule":
                # Use Ketcher editor
                st.subheader("Draw your molecule")
                selected = ""  # Default empty structure
                selected = st_ketcher(selected)  # Display Ketcher editor
                
                # If user draws a molecule, convert to SMILES
                if selected:
                    smiles_input = selected
                    st.write(f"Generated SMILES: {smiles_input}")
                else:
                    smiles_input = ""
            else:
                # Original SMILES input box
                smiles_input = st.text_input("Enter SMILES string", "")
            
            if st.button("Search"):
                if smiles_input:
                    try:
                        mol = Chem.MolFromSmiles(smiles_input)
                        if mol:  # If SMILES is valid
                            with st.spinner("Searching..."):
                                results = search_database(smiles_input)
                            
                            if results is not None and not results.empty:
                                st.subheader("Search Results")
                                st.dataframe(results)

                                # Tell user if molecule was found in database
                                match_found = any(row['Database SMILES'] != 'Not Found' and row['Database SMILES'] != 'Invalid SMILES' for _, row in results.iterrows())
                                
                                if match_found:
                                    st.success("✅ Match found! The molecule was found in the database.")
                                else:
                                    st.warning("⚠️ No match found! The molecule does not exist in the database.")
                                
                                # Display molecular structure
                                st.subheader("Molecular Structure")
                                
                                # Display input molecule
                                mol_img = display_molecule(smiles_input)
                                if mol_img:
                                    st.image(mol_img)
                            else:
                                st.info("No matching molecules found in database")
                        else:
                            st.error("Invalid SMILES string, please check your input")
                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
                else:
                    st.warning("Please enter a SMILES string or draw a molecule")

        # Multiple SMILES search
        else:  # Multiple SMILES search
            # Provide two input methods
            input_method = st.radio("Select Input Method", ["Text Input", "CSV File Upload"])
            
            if input_method == "Text Input":
                smiles_text = st.text_area("Enter multiple SMILES strings (one per line)", "")
                
                if st.button("Search"):
                    if smiles_text:
                        smiles_list = [s.strip() for s in smiles_text.split("\n") if s.strip()]
                        if smiles_list:
                            with st.spinner("Searching..."):
                                try:
                                    results = search_database(smiles_list)
                                    
                                    if results is not None and not results.empty:
                                        st.subheader("Search Results")
                                        st.dataframe(results)
                                        
                                        # Provide download option
                                        csv = results.to_csv(index=False)
                                        st.download_button(
                                            label="Download Search Results",
                                            data=csv,
                                            file_name="database_search_results.csv",
                                            mime="text/csv"
                                        )
                                        
                                        # Display number of found molecules
                                        found_count = sum(1 for _, row in results.iterrows() if row['Database SMILES'] != 'Not Found' and row['Database SMILES'] != 'Invalid SMILES')
                                        st.write(f"Found {found_count} matching molecules in database")
                                    else:
                                        st.info("No matching molecules found in database")
                                except Exception as e:
                                    st.error(f"Error during processing: {str(e)}")
                        else:
                            st.warning("Please enter at least one SMILES string")
                    else:
                        st.warning("Please enter SMILES strings")
            
            # Upload CSV file search
            else:  # CSV file upload
                st.write("Upload a CSV file containing SMILES (must have a 'SMILES' column)")
                uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
                
                if uploaded_file is not None:
                    try:
                        df = pd.read_csv(uploaded_file)
                        if "SMILES" in df.columns:
                            st.write(f"Loaded {len(df)} compounds")
                            st.dataframe(df.head())
                            if st.button("Search"):
                                smiles_list = df["SMILES"].tolist()
                                with st.spinner("Searching..."):
                                    try:
                                        results = search_database(smiles_list)
                                        
                                        if results is not None and not results.empty:
                                            # Merge results with original dataframe
                                            # Create a mapping dictionary from original SMILES to search results
                                            result_dict = {}
                                            for _, row in results.iterrows():
                                                orig_smiles = row['Original SMILES']
                                                if orig_smiles not in result_dict:
                                                    result_dict[orig_smiles] = []
                                                result_dict[orig_smiles].append({
                                                    'Database SMILES': row['Database SMILES'],
                                                    'Category': row['Category']
                                                })
                                            
                                            # Add new columns to original dataframe
                                            df['Database SMILES'] = df['SMILES'].apply(
                                                lambda s: result_dict.get(s, [{}])[0].get('Database SMILES', 'Not Found') if s in result_dict else 'Not Found'
                                            )
                                            df['Category'] = df['SMILES'].apply(
                                                lambda s: result_dict.get(s, [{}])[0].get('Category', 'Unknown') if s in result_dict else 'Unknown'
                                            )
                                            
                                            # Display results
                                            st.subheader("Search Results")
                                            st.dataframe(df)
                                            
                                            # Provide download option
                                            csv = df.to_csv(index=False)
                                            st.download_button(
                                                label="Download Search Results",
                                                data=csv,
                                                file_name="database_search_results.csv",
                                                mime="text/csv"
                                            )
                                            
                                            # Display number of found molecules
                                            found_count = sum(1 for status in df['Database SMILES'] if status != 'Not Found' and status != 'Invalid SMILES')
                                            st.write(f"Found {found_count} matching molecules in database")
                                        else:
                                            st.info("No matching molecules found in database")
                                    except Exception as e:
                                        st.error(f"Error during processing: {str(e)}")
                        else:
                            st.error("CSV file does not contain a 'SMILES' column")
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")

    # Add About information
    st.sidebar.markdown("---") # 默认是蓝色的样式
    st.sidebar.subheader("About")
    # Display different about information based on current function
    if app_mode == "Prediction":
        st.sidebar.info(
            """
            This application uses ensemble models of machine learning and deep learning to predict compound interactions with estrogen-related receptor α (ERRα).
            
            - **Binder**: Predicts whether compounds can bind to ERRα
            - **Antagonist**: Predicts whether compounds are ERRα antagonists
            - **Agonist**: Predicts whether compounds are ERRα agonists
            """
        )
    else:  # Database Search
        st.sidebar.info(
            """
            The database search function allows searching for compound activity information in the known database.

            Compounds in the database are classified into 8 categories :
            - **both_active**
            - **ago_active_ant_inactive**
            - **ago_active_ant_none**
            - **ago_inactive_ant_active**
            - **both_inactive**
            - **ago_inactive_ant_none**
            - **ago_none_ant_active**
            - **ago_none_ant_inactive**
            """
        )

if __name__ == "__main__":
    main()

            # In this context:
            # - **ant** represents antagonist
            # - **ago** represents agonist

# streamlit run app.py