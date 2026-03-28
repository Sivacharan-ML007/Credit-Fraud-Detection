import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from utils import load_data, scale_data, reconstruction_error

DATA_PATH = "data/sample.csv"  #change dataset name for custom use
MODEL_PATH = "model/autoencoder.h5"
MODE = "both"   # "fraud" or "both"

# ===================== LOAD =====================
normal,fraud=load_data(DATA_PATH)
x_normal,x_fraud,scaler=scale_data(normal, fraud)

x_train, x_test=train_test_split(
    x_normal, test_size=0.2, random_state=42
)

model=tf.keras.models.load_model(MODEL_PATH)

# ===================== RECONSTRUCTION =====================
recon_test=model.predict(x_test)
recon_fraud=model.predict(x_fraud)

error_test=reconstruction_error(x_test, recon_test)
error_fraud=reconstruction_error(x_fraud, recon_fraud)

threshold=np.percentile(error_test, 99.1)

print(f"\nThreshold: {threshold:.6f}")


# ===================== FRAUD ONLY =====================
if MODE == "fraud":
    y_true = np.ones(len(x_fraud))
    y_pred = (error_fraud > threshold).astype(int)

    recall = recall_score(y_true, y_pred)

    print("\n===== FRAUD ONLY =====")
    print(f"Recall: {recall:.4f}")


# ===================== BOTH =====================
elif MODE=="both":
    x_all=np.concatenate([x_test, x_fraud])

    y_true = np.concatenate([
        np.zeros(len(x_test)),
        np.ones(len(x_fraud))
    ])

    error_all=np.concatenate([error_test,error_fraud])
    y_pred=(error_all > threshold).astype(int)

    precision=precision_score(y_true,y_pred)
    recall=recall_score(y_true,y_pred)
    auc=roc_auc_score(y_true,error_all)

    print("\n===== COMBINED =====")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"AUC       : {auc:.4f}")
