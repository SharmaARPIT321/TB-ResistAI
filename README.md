# TB-Drug-Resistance-ML (TB-ResistAI)

Machine-learning research project for predicting tuberculosis drug resistance from selected mutation features.

## Final models
| Drug | Model | Features | Threshold |
|---|---|---:|---:|
| Rifampicin | Random Forest | 30 | 0.72 |
| Isoniazid | Logistic Regression | 2 | 0.50 |
| Ethambutol | Random Forest | 8 | 0.32 |

## External validation — final results
| Drug | ROC-AUC | 95% CI | Accuracy | Precision | Recall | F1 |
|---|---:|---|---:|---:|---:|---:|
| Rifampicin | **94.76%** | 93.16–96.17% | 93.20% | 98.34% | 87.18% | 92.43% |
| Isoniazid | 87.91% | 85.93–89.84% | 87.90% | 97.02% | 78.20% | 86.60% |
| Ethambutol | 87.10% | 84.47–89.60% | 85.40% | 70.06% | 86.11% | 77.26% |

The detailed external table stores RIF ROC-AUC as 94.758925%, which rounds to 94.76%.

## Structure
```text
TB-Drug-Resistance-ML/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── rifampicin/{model.joblib,features.json}
│   ├── isoniazid/{model.joblib,features.json}
│   └── ethambutol/{model.joblib,features.json}
├── results/
│   ├── internal/
│   ├── external/
│   └── figures/
├── notebooks/
└── data/
    └── README.md
```

## Run the website
```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads the already-fitted models; it does not retrain them. It expects binary mutation-presence features matching the training representation.

## Reproducibility and data
Raw/source datasets are kept out of this public-ready repository until redistribution permissions are verified. The uploaded complete backup remains the private master copy. Add the original Colab `.ipynb` to `notebooks/` before publishing.

## Disclaimer
Research prototype only. Do not use model predictions for clinical diagnosis or treatment decisions.
