import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from utils import load_data, scale_data, reconstruction_error

DATA_PATH = "data/sample.csv"  #change dataset name for custom use
MODEL_PATH = "model/autoencoder.h5"
MODE = "fraud"   # "fraud" or "both"
MODE = "both"   # "fraud" or "both"

# ===================== LOAD =====================
normal,fraud=load_data(DATA_PATH)
x_normal,x_fraud,scaler=scale_data(normal, fraud)

x_train, x_test=train_test_split(
    x_normal, test_size=0.2, random_state=42
)

# Load without compile state to avoid legacy metric/loss deserialization issues in newer Keras.
model=tf.keras.models.load_model(MODEL_PATH, compile=False)

# ===================== RECONSTRUCTION =====================
recon_test=model.predict(x_test)
recon_fraud=model.predict(x_fraud)

error_test=reconstruction_error(x_test, recon_test)
error_fraud=reconstruction_error(x_fraud, recon_fraud)

threshold=np.percentile(error_test, 99.1)

print(f"\nThreshold: {threshold:.6f}")

def print_predictions(y_true, y_pred, n=20):
    """Print first `n` samples showing true label and predicted label.

    Labels are shown as 'Fraud' or 'Not Fraud'.
    """
    total = min(n, len(y_true))
    print(f"\nShowing first {total} predictions (True -> Predicted):")
    for i in range(total):
        t = "Fraud" if y_true[i] == 1 else "Not Fraud"
        p = "Fraud" if y_pred[i] == 1 else "Not Fraud"
        print(f"{i+1:3d}: {t} -> {p}")


def print_fraud_legit_metrics(y_true, y_pred, scores):
    """Print class-wise metrics for Legit (0) and Fraud (1)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    legit_precision = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    legit_recall = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    fraud_precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    fraud_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)

    def f1(p, r):
        return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)

    legit_f1 = f1(legit_precision, legit_recall)
    fraud_f1 = f1(fraud_precision, fraud_recall)

    print("\nClass Metrics")
    print(f"Legit  - Precision: {legit_precision:.4f} | Recall: {legit_recall:.4f} | F1: {legit_f1:.4f}")
    print(f"Fraud  - Precision: {fraud_precision:.4f} | Recall: {fraud_recall:.4f} | F1: {fraud_f1:.4f}")

    print("\nGlobal Metrics")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"PR-AUC  : {pr_auc:.4f}")

    print("\nConfusion Matrix [ [TN, FP], [FN, TP] ]")
    print(np.array([[tn, fp], [fn, tp]]))


# ===================== FRAUD ONLY =====================
if MODE == "fraud":
    y_true = np.ones(len(x_fraud))
    y_pred = (error_fraud > threshold).astype(int)

    recall = recall_score(y_true, y_pred)

    print("\n===== FRAUD ONLY =====")
    print(f"Recall: {recall:.4f}")
    print_predictions(y_true, y_pred, n=20)


# ===================== BOTH =====================
elif MODE=="both":
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
    print_fraud_legit_metrics(y_true, y_pred, error_all)
    print_predictions(y_true, y_pred, n=20)
