from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import pandas as pd

def standardize(smiles: str, stereo=True, canon_tautomer=True) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)

        # rdMolStandardize.Cleanup(mol) is equivalent to the molvs.Standardizer().standardize(mol) function
        # Remove H atoms, Sanitize, Disconnect metal bonds, Normalize, Reionize, Assign stereochemistry
        clean_mol = rdMolStandardize.Cleanup(mol)

        # Get the actual mol we are interested in (e.g., remove mixtures or salts, etc.)
        clean_mol = rdMolStandardize.FragmentParent(clean_mol)

        # Neutralize 去电荷
        uncharger = rdMolStandardize.Uncharger()
        clean_mol = uncharger.uncharge(clean_mol)

        # Remove isotope
        for atom in clean_mol.GetAtoms():
            atom.SetIsotope(0)
    
        # Remove stereochemistry 是否移除立体化学信息 如果 stereo 为 False，那么代码会执行移除立体化学信息的操作
        if not stereo:
            Chem.RemoveStereochemistry(clean_mol)

        # Get the canonical tautomer 互变异构体
        if canon_tautomer:
            te = rdMolStandardize.TautomerEnumerator()
            clean_mol = te.Canonicalize(clean_mol)

        return Chem.MolToSmiles(clean_mol)
    
    except Exception as e:
        print(f"{smiles}; {e}")
        # Return the original SMILES for further inspection or return None
        return smiles
    
def choose_standardize(smiles):
    if any(symbol in smiles for symbol in ['@', '\\', '/']):
        return standardize(smiles,canon_tautomer=False)
    else:
        return standardize(smiles,canon_tautomer=True)


