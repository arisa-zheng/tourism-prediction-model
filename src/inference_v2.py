# ==========================================
# 03_inference.py
# Loads Keras model and runs inference on data/test_data.csv
# Saves:
#   - data/output/03_inference/metrics_test.txt
#   - data/output/03_inference/confusion_matrix_test.png
#   - data/output/03_inference/predictions.csv
# ==========================================

import os
import sys
import joblib
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score

# =========================
# CONFIG
# =========================
TEST_PATH = "data/test_data.csv"
TARGET_COL = "ProdTaken"

MODEL_PATH = "examples/tourism_nn_model.h5"
COLS_PATH  = "examples/feature_columns.joblib"
SCALER_PATH = "examples/scaler.joblib"

OUTPUT_DIR = "data/output/03_inference_v2"
THRESHOLD = 0.4  

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOGGER
# =========================
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(os.path.join(OUTPUT_DIR, "complete_output.txt"))

def save_confusion_matrix_png(cm, out_path, title):

    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / np.sum(cm)

    plt.figure(figsize=(8, 6))

    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=["no", "yes"],
        yticklabels=["no", "yes"]
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Metrics text box (right side)
    textstr = (
        f"True Negatives: {tn}\n"
        f"False Positives: {fp}\n"
        f"False Negatives: {fn}\n"
        f"True Positives: {tp}\n\n"
        f"Accuracy: {accuracy:.4f}"
    )

    plt.gcf().text(
        1.02, 0.5, textstr,
        fontsize=11,
        verticalalignment='center',
        bbox=dict(boxstyle="round", facecolor="beige", alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

# =========================
# Custom Loss (must match training)
# =========================
# def combined_bce_l1_weights_loss(model, alpha=1.0, beta=0.001):
#     def loss(y_true, y_pred):
#         y_true_f = tf.cast(y_true, tf.float32)
#         bce = tf.keras.losses.binary_crossentropy(y_true_f, y_pred)
#         bce_loss = tf.reduce_mean(bce)
#         l1_reg = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in model.trainable_weights])
#         return alpha * bce_loss + beta * l1_reg
#     return loss

# =========================
# Weighted Binary Crossentropy
# =========================
def make_weighted_binary_crossentropy(pos_weight):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        weights = y_true * (pos_weight - 1.0) + 1.0
        return tf.reduce_mean(bce * weights)
    return loss

print("=" * 70)
print("COMPONENT 3: INFERENCE STARTED (Keras NN)")
print("=" * 70)

# =========================
# 1) Load model
# =========================
# print("\nSTEP 1: LOAD MODEL")
# model = keras.models.load_model(MODEL_PATH, compile=False)

# custom_loss = combined_bce_l1_weights_loss(model, alpha=1.0, beta=0.001)
# model.compile(
#     optimizer="adam",
#     loss=custom_loss,
#     metrics=["accuracy", keras.metrics.AUC(name="auc")]
# )

# Use SAME pos_weight value used in training

print("\nSTEP 1: LOAD MODEL")

model = keras.models.load_model(MODEL_PATH, compile=False)
pos_weight = 2.5   # <-- or your computed value

loss_fn = make_weighted_binary_crossentropy(pos_weight)

model.compile(
    optimizer="adam",
    loss=loss_fn,
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
)

print("Loaded model from:", MODEL_PATH)

# =========================
# 2) Load preprocessing artifacts
# =========================
print("\nSTEP 2: LOAD PREPROCESSING ARTIFACTS")
feature_columns = joblib.load(COLS_PATH)
scaler = joblib.load(SCALER_PATH)

print("Loaded feature columns:", len(feature_columns))
print("Loaded scaler:", type(scaler).__name__)

# =========================
# 3) Load test data
# =========================
print("\nSTEP 3: LOAD TEST DATA")
df = pd.read_csv(TEST_PATH)
print("Test file shape:", df.shape)

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found in {TEST_PATH}")

df = df.dropna(subset=[TARGET_COL])

X_raw = df.drop(columns=[TARGET_COL])
y_true = df[TARGET_COL].astype(int).values  # assume 0/1

# =========================
# 4) One-hot encode + align columns + scale
# =========================
print("\nSTEP 4: ENCODE + ALIGN + SCALE")
categorical_columns = X_raw.select_dtypes(include=["object"]).columns.tolist()
X_enc = pd.get_dummies(X_raw, columns=categorical_columns, drop_first=True)

# Align to training columns
X_enc = X_enc.reindex(columns=feature_columns, fill_value=0)

X_scaled = scaler.transform(X_enc)

# =========================
# 5) Predict + Evaluate
# =========================
print("\nSTEP 5: PREDICT + EVALUATE")
proba = model.predict(X_scaled).reshape(-1)
pred = (proba > THRESHOLD).astype(int)

acc = accuracy_score(y_true, pred)
auc = roc_auc_score(y_true, proba)
cm = confusion_matrix(y_true, pred)
report = classification_report(y_true, pred, digits=4)

print("Test Accuracy:", acc)
print("Test AUC:", auc)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

with open(os.path.join(OUTPUT_DIR, "metrics_test.txt"), "w") as f:
    f.write(f"THRESHOLD = {THRESHOLD}\n")
    f.write(f"Accuracy = {acc}\n")
    f.write(f"AUC = {auc}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n")

save_confusion_matrix_png(
    cm,
    os.path.join(OUTPUT_DIR, "confusion_matrix_test.png"),
    title="Test Confusion Matrix"
)

# =========================
# 6) Save predictions
# =========================
print("\nSTEP 6: SAVE PREDICTIONS")
pred_out = df.copy()
pred_out["Pred_Prob"] = proba
pred_out["Pred_Label"] = pred
pred_out.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)

print("Saved predictions to:", os.path.join(OUTPUT_DIR, "predictions.csv"))

print("\n" + "=" * 70)
print("COMPONENT 3: INFERENCE COMPLETE")
print("=" * 70)

# Close logger
sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
