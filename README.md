# Credit-Fraud-Detection(autoencoders)
This script builds an unsupervised anomaly detection model using a deep learning autoencoder in TensorFlow.  
The idea:  Train the model only on normal transactions Let it learn how “normal” data looks Flag transactions as fraud when reconstruction error is high

## 🧠 Model Architecture
- Dense layers with bottleneck
- L1 + L2 regularization
- Dropout for robustness

## ⚙️ Tech Stack
- TensorFlow / Keras
- Scikit-learn
- Pandas / NumPy

## 🚀 How to Run
```bash
git clone https://github.com/Sivacharan-ML007/Credit-Fraud-Detection.git
cd credit-card-fraud-autoencoder

pip install -r requirements.txt
/home/sivacharan/miniconda3/envs/myenv/bin/python model.py
```

## Dataset Files
- `data/sample.csv` is the demo dataset and should be used for default runs.
- `data/creditcard.csv` in this repository is a placeholder note, not transaction CSV content.
- To use your own data, replace a file under `data/` and keep the same columns (including `Class`).

## Default Data Path
Set `DATA_PATH` in `model.py` to:

```python
DATA_PATH = "data/sample.csv"
```
