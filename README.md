# Sequence Pair Deep Learning Demo

This repository contains a **generic deep learning pipeline** for
paired biological sequences (e.g., ligand–protein, drug–target).
It is adapted from a Jupyter notebook used in a previous research project,
but it uses **synthetic data only** and contains **no proprietary code or datasets**.

The goal is to illustrate my typical coding style for:

- Model architecture definition (PyTorch)
- Data pipeline & `Dataset`/`DataLoader` usage
- Training / evaluation loops with basic metrics
- Modular, readable code organization

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
