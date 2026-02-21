import pandas as pd
import numpy as np
import joblib
import os
from rdkit import Chem
from rdkit.Chem import MACCSkeys, DataStructs

def run_oecd_final_prediction():
    print("--- CYP-SENSE QSAR: OECD Final Prediction & Feature Importance ---")
    
    # 1. Load the OECD Champion Model
    model_path = input("Enter your Champion Model path (e.g., m1_new.joblib): ").strip()
    if not os.path.exists(model_path):
        return print("Error: Model file not found.")
    
    ensemble = joblib.load(model_path)
    
    # 2. EXTRACT KNOWLEDGE (FEATURE IMPORTANCE)
    # The first estimator [0] in your ensemble is the Random Forest
    rf_component = ensemble.estimators_[0] 
    importances = rf_component.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n--- Structural Drivers of Activity (Top 5 MACCS Bits) ---")
    feature_results = []
    for i in range(5):
        bit = indices[i]
        score = importances[bit]
        print(f"Rank {i+1}: MACCS Bit {bit} (Score: {score:.4f})")
        feature_results.append([f"Rank {i+1}", f"MACCS_{bit}", score])

    # 3. Load Training Data for Similarity Check (Reliability)
    train_path = input("\nEnter Training Feature CSV (to calculate Similarity): ").strip()
    df_train = pd.read_csv(train_path)
    train_mols = [Chem.MolFromSmiles(s) for s in df_train['SMILES']]
    train_fps = [MACCSkeys.GenMACCSKeys(m) for m in train_mols if m]

    # 4. Predict New Compounds
    predict_csv = input("Enter CSV for 3A1, 3A2, 3A3 (must have 'ID' and 'SMILES'): ").strip()
    if os.path.exists(predict_csv):
        df_new = pd.read_csv(predict_csv)
        results = []
        
        print("\nProcessing Authenticated Predictions...")
        for _, row in df_new.iterrows():
            mol = Chem.MolFromSmiles(row['SMILES'])
            if mol:
                # Prediction Logic
                fp_bits = list(MACCSkeys.GenMACCSKeys(mol).ToBitString())
                feat = np.array([int(b) for b in fp_bits]).reshape(1, -1)
                prob = ensemble.predict_proba(feat)[0][1]
                
                # Similarity Logic (OECD Applicability Domain)
                target_fp = MACCSkeys.GenMACCSKeys(mol)
                sims = [DataStructs.TanimotoSimilarity(target_fp, tfp) for tfp in train_fps]
                max_sim = max(sims)
                
                reliability = "Reliable" if max_sim > 0.6 else "Novel (Outside AD)"
                results.append([row['ID'], "ACTIVE" if prob > 0.5 else "INACTIVE", 
                                f"{prob*100:.1f}%", f"{max_sim:.2f}", reliability])
        
        # Save Predictions
        res_df = pd.DataFrame(results, columns=['ID', 'Prediction', 'Confidence', 'Similarity', 'Status'])
        print("\n", res_df)
        res_df.to_csv("OECD_Final_Validated_Results.csv", index=False)
        
        # Save Features
        feat_df = pd.DataFrame(feature_results, columns=['Rank', 'Feature', 'Importance_Score'])
        feat_df.to_csv("OECD_Feature_Importance.csv", index=False)
        print("\nSUCCESS: Results saved to 'OECD_Final_Validated_Results.csv' and 'OECD_Feature_Importance.csv'")

if __name__ == "__main__":
    run_oecd_final_prediction()