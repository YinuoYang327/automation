# Drug-Target Interaction Prediction

This repository contains a **generic deep learning pipeline** for
paired drug–target.


## Files

- `models.py` – CNN-based encoders and a `SequencePairRegressor` model
- `data.py` – synthetic data generation and `SequencePairDataset`
- `utils.py` – training and evaluation utilities
- `train.py` – end-to-end training script
- `requirements.txt` – Python dependencies

## Requirements

- Python >= 3.8  
- PyTorch >= 1.12  
- NumPy  
- scikit-learn  

Install with:

```bash
pip install -r requirements.txt
