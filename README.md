# TB-ResistAI

Machine-learning research project for predicting tuberculosis
drug resistance using selected mutation features.

## Final selected models

| Drug | Model | Features | Decision threshold |
|---|---|---:|---:|
| Rifampicin (RIF) | Random Forest | 30 | 0.72 |
| Isoniazid (INH) | Logistic Regression | 2 | 0.50 |
| Ethambutol (EMB) | Random Forest | 8 | 0.32 |

## Project contents

- models/: exported fitted models and model manifest
- results/tables/: DataFrames available in the Colab runtime, saved as CSV
- results/figures/: Matplotlib figures open at backup time
- metadata/: environment details and saved-table inventory

## Important

The backup contains the tables, figures, and models available in the
current runtime. It cannot recover variables or figures that have
already been cleared, nor can it automatically include the notebook
source itself.

Verify the final external-validation ROC-AUC directly against the
saved results table before reporting it in the README or publication.

Research prototype only; not intended for clinical diagnosis.
