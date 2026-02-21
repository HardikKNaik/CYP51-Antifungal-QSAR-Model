import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import os

def run_step1_feature_generation():
    print("--- CYP-SENSE QSAR: Step 1 (Feature Generation) ---")
    
    # 1. Interactive Inputs
    active_path = input("Enter path for ACTIVES CSV (e.g., actives_cure.csv): ").strip()
    inactive_path = input("Enter path for INACTIVES CSV (e.g., inactives.csv): ").strip()
    output_name = input("Enter name for output file (without .csv): ").strip()

    if not os.path.exists(active_path) or not os.path.exists(inactive_path):
        print("Error: One or both input files not found.")
        return

    # 2. Load and Label
    print("\nLoading datasets...")
    df_act = pd.read_csv(active_path)
    df_inact = pd.read_csv(inactive_path)
    
    # Standardize Column Names (Assuming SMILES_Standard for actives and SMILES for inactives)
    # We will try to find the SMILES column automatically
    def find_smiles(df):
        return next((c for c in df.columns if 'smiles' in c.lower()), None)

    act_smi = find_smiles(df_act)
    inact_smi = find_smiles(df_inact)

    df_act = df_act[[act_smi]].copy().rename(columns={act_smi: 'SMILES'})
    df_act['Label'] = 1
    
    df_inact = df_inact[[inact_smi]].copy().rename(columns={inact_smi: 'SMILES'})
    df_inact['Label'] = 0

    master_df = pd.concat([df_act, df_inact], ignore_index=True)
    print(f"Total Combined Samples: {len(master_df)} (1: {len(df_act)}, 0: {len(df_inact)})")

    # 3. MACCS Fingerprint Generation
    print("Generating 166-bit MACCS Fingerprints...")
    features = []
    valid_indices = []

    for idx, smi in enumerate(master_df['SMILES']):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            # MACCS keys are 167 bits (index 0 is a dummy bit)
            fp = list(MACCSkeys.GenMACCSKeys(mol).ToBitString())
            features.append([int(b) for b in fp])
            valid_indices.append(idx)
        else:
            print(f"Warning: Invalid SMILES skipped at index {idx}")

    # 4. Save to CSV
    # Only keep rows where SMILES were valid
    final_df = master_df.iloc[valid_indices].reset_index(drop=True)
    
    maccs_cols = [f"MACCS_{i}" for i in range(167)]
    fp_df = pd.DataFrame(features, columns=maccs_cols)
    
    output_df = pd.concat([final_df, fp_df], axis=1)
    output_df.to_csv(f"{output_name}.csv", index=False)
    
    print(f"\nSUCCESS: Feature file saved as '{output_name}.csv'")
    print(f"Dimensions: {output_df.shape[0]} samples x {output_df.shape[1]} columns")

if __name__ == "__main__":
    run_step1_feature_generation()