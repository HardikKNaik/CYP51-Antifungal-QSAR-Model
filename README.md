# CYP51-Antifungal-QSAR-Model
This repository contains a validated Machine Learning QSAR pipeline developed to predict the antifungal activity of novel 4-thiazolidinone derivatives against *Candida albicans* CYP51.

## 📊 Model Performance & Reliability
The model was validated using **Leave-One-Out Cross-Validation (LOOCV)** to ensure maximum reliability for medicinal chemistry datasets.

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **91.43%** | High overall predictive power |
| **MCC** | **0.7485** | Strong correlation (Gold Standard for QSAR) |
| **Sensitivity** | **96.30%** | Excellent at identifying true actives |
| **Specificity** | **75.00%** | Robust discrimination of inactives |

## 🛠️ Methodology
- **Descriptors:** 166-bit MACCS Molecular Fingerprints.
- **Algorithm:** Consensus Ensemble (Soft-Voting) consisting of Random Forest, Extra Trees, and Support Vector Classifier (SVC).
- **Validation:** Fully compliant with the 5 OECD Principles for QSAR validation.
- **Applicability Domain:** Quantified via Tanimoto Similarity assessment to ensure prospective reliability.

## 📂 Folder Structure
- `/dataset`: Training data and screening library (including 3A series).
- `/model`: The trained `.joblib` champion model and the detailed OECD Validation Report.
- `/scripts`: The Python workflow (Feature Generation -> Training -> Prediction).

## 🚀 Scientific Impact
In prospective screening, the model identified the novel series (**3A1, 3A2, 3A3**) as **Active** with high confidence scores (>87.8%). The feature importance analysis confirmed the activity drivers.
