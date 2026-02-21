import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix, recall_score, precision_score

def run_oecd_optimizer():
    print("--- CYP-SENSE QSAR: OECD-Compliant Grid Optimizer ---")
    
    input_path = input("Enter the path for your FEATURE CSV: ").strip()
    if not os.path.exists(input_path): return print("Error: File not found.")
    
    df = pd.read_csv(input_path)
    y = df['Label'].values
    X = df.filter(like='MACCS_').values
    feature_names = df.filter(like='MACCS_').columns.tolist()

    print("\n--- Define Grid Search Ranges ---")
    depth_list = [int(d.strip()) if d.strip().lower() != 'none' else None for d in input("Depths (e.g. 3, 5, None): ").split(',')]
    c_list = [float(c.strip()) for c in input("SVM C values (e.g. 0.01, 1, 10): ").split(',')]
    voting_list = input("Voting (soft, hard): ").lower().replace(' ', '').split(',')

    best_mcc = -1
    best_params = {}

    loo = LeaveOneOut()

    for depth in depth_list:
        for c_val in c_list:
            for v_type in voting_list:
                y_true, y_pred = [], []
                
                rf = RandomForestClassifier(n_estimators=200, max_depth=depth, random_state=42)
                et = ExtraTreesClassifier(n_estimators=200, max_depth=depth, random_state=42)
                svc = SVC(probability=True, kernel='rbf', C=c_val, random_state=42)
                
                ensemble = VotingClassifier(estimators=[('rf', rf), ('et', et), ('svc', svc)], voting=v_type)

                for train_index, test_index in loo.split(X):
                    ensemble.fit(X[train_index], y[train_index])
                    y_true.append(y[test_index][0])
                    y_pred.append(ensemble.predict(X[test_index])[0])
                
                mcc = matthews_corrcoef(y_true, y_pred)
                acc = accuracy_score(y_true, y_pred)
                
                if mcc > best_mcc:
                    best_mcc = mcc
                    best_params = {'depth': depth, 'c': c_val, 'vote': v_type}
                    print(f" -> New Best MCC: {mcc:.4f} (Acc: {acc:.2f})")

    # Final training and expanded metrics
    print(f"\nFinalizing Champion Model (MCC: {best_mcc:.4f})...")
    out_name = input("Enter name for output (e.g., cyp_champion): ").strip()
    
    final_model = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=500, max_depth=best_params['depth'], random_state=42)),
            ('et', ExtraTreesClassifier(n_estimators=500, max_depth=best_params['depth'], random_state=42)),
            ('svc', SVC(probability=True, kernel='rbf', C=best_params['c'], random_state=42))
        ], voting=best_params['vote'])
    
    final_model.fit(X, y)
    
    # Calculate final OECD metrics via the same LOOCV logic for the log
    y_true, y_pred = [], []
    for tr_idx, te_idx in loo.split(X):
        final_model.fit(X[tr_idx], y[tr_idx])
        y_true.append(y[te_idx][0]); y_pred.append(final_model.predict(X[te_idx])[0])
    
    cm = confusion_matrix(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred) # True Positive Rate
    specificity = cm[0,0] / (cm[0,0] + cm[0,1]) # True Negative Rate
    precision = precision_score(y_true, y_pred)

    joblib.dump(final_model, f"{out_name}.joblib")
    
    with open(f"{out_name}_OECD_REPORT.txt", "w") as f:
        f.write(f"--- OECD QSAR VALIDATION REPORT ---\n")
        f.write(f"Parameters: {best_params}\n")
        f.write(f"1. Accuracy (Overall Power): {acc:.4f}\n")
        f.write(f"2. MCC (Robustness): {best_mcc:.4f}\n")
        f.write(f"3. Sensitivity (Active Recall): {sensitivity:.4f}\n")
        f.write(f"4. Specificity (Inactive Precision): {specificity:.4f}\n")
        f.write(f"5. Precision: {precision:.4f}\n")
        f.write(f"\nConfusion Matrix: TN:{cm[0][0]} FP:{cm[0][1]} / FN:{cm[1][0]} TP:{cm[1][1]}\n")

    print(f"OECD Report and Model saved as {out_name}.")

if __name__ == "__main__":
    run_oecd_optimizer()