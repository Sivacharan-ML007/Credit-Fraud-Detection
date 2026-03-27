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
git clone https://github.com/sivacharan-ML007/Credit-Fraud-Detection(autoencoders).git
cd credit-card-fraud-autoencoder

pip install -r requirements.txt
