# ==========================================
# 02_training.py
# Train a Keras NN model 
# Saves:
#   - examples/tourism_nn_model.h5
#   - examples/feature_columns.joblib
#   - examples/scaler.joblib
#   - data/output/02_training/metrics_test.txt
#   - data/output/02_training/confusion_matrix_test.png
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

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# =========================
# CONFIG
# =========================
TRAIN_PATH = "data/train_data.csv"
TEST_PATH  = "data/test_data.csv"
TARGET_COL = "ProdTaken"

OUTPUT_DIR = "data/output/02_training"
MODEL_DIR = "examples"
MODEL_PATH = os.path.join(MODEL_DIR, "tourism_nn_model.h5")
COLS_PATH  = os.path.join(MODEL_DIR, "feature_columns.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

RANDOM_STATE = 42
THRESHOLD = 0.6  

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

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

# =========================
# CONFUSION MATRIX FUNCTION
# =========================

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
# CORRELATION HEATMAP FUNCTION
# =========================
def save_correlation_heatmap(df_features, out_path, title="Feature Correlation Matrix", top_n=20):

    num_df = df_features.select_dtypes(include=[np.number])

    if num_df.shape[1] > top_n:
        variances = num_df.var().sort_values(ascending=False)
        selected_cols = variances.head(top_n).index
        num_df = num_df[selected_cols]

    corr = num_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "Correlation"}
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_training_curves(history, out_path):
    """
    Saves training vs validation Accuracy and Loss in one image (2 panels)
    like the example screenshot.
    """
    hist = history.history

    # ---- Accuracy keys can vary depending on TF/Keras version ----
    acc_key = "accuracy" if "accuracy" in hist else ("acc" if "acc" in hist else None)
    val_acc_key = "val_accuracy" if "val_accuracy" in hist else ("val_acc" if "val_acc" in hist else None)

    loss = hist.get("loss", [])
    val_loss = hist.get("val_loss", [])

    plt.figure(figsize=(12, 4))

    # ===== Left plot: Accuracy =====
    plt.subplot(1, 2, 1)
    if acc_key and val_acc_key:
        plt.plot(hist[acc_key], label="Train Accuracy")
        plt.plot(hist[val_acc_key], label="Val Accuracy")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, "Accuracy history not available", ha="center", va="center")
        plt.axis("off")

    # ===== Right plot: Loss =====
    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# Custom Loss
# =========================
def compile_model_with_custom_loss(model, loss_function, optimizer="adam", metrics=None):
    if metrics is None:
        metrics = ["accuracy", keras.metrics.AUC(name="auc")]
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
    return model

def combined_bce_l1_weights_loss(model, alpha=1.0, beta=0.001):
    """
    Combined loss: alpha * BCE + beta * L1_regularization on model weights
    """
    def loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)

        bce = tf.keras.losses.binary_crossentropy(y_true_f, y_pred)
        bce_loss = tf.reduce_mean(bce)

        l1_reg = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in model.trainable_weights])
        total_loss = alpha * bce_loss + beta * l1_reg
        return total_loss
    return loss

print("=" * 70)
print("COMPONENT 2: TRAINING STARTED (Keras NN)")
print("=" * 70)

# =========================
# 1) Load prepared data
# =========================
print("\nSTEP 1: LOAD TRAIN/TEST DATA")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)

# Drop missing target (safety)
train_df = train_df.dropna(subset=[TARGET_COL])
test_df  = test_df.dropna(subset=[TARGET_COL])

X_train_raw = train_df.drop(columns=[TARGET_COL])
y_train_raw = train_df[TARGET_COL]

X_test_raw = test_df.drop(columns=[TARGET_COL])
y_test_raw = test_df[TARGET_COL]

print("\nTrain target distribution:")
print(y_train_raw.value_counts(normalize=True))

print("\nTest target distribution:")
print(y_test_raw.value_counts(normalize=True))

# Encode target (even if already 0/1)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw.astype(str))
y_test  = label_encoder.transform(y_test_raw.astype(str))

print("\nLabel classes:", list(label_encoder.classes_))

# =========================
# 2) One-hot encode categoricals 
# =========================
print("\nSTEP 2: ONE-HOT ENCODE (pd.get_dummies, drop_first=True)")

categorical_columns = X_train_raw.select_dtypes(include=["object"]).columns.tolist()
print("Categorical columns:", categorical_columns)

X_train_enc = pd.get_dummies(X_train_raw, columns=categorical_columns, drop_first=True)
X_test_enc  = pd.get_dummies(X_test_raw,  columns=categorical_columns, drop_first=True)

# IMPORTANT: align test columns to train columns (keeps architecture consistent at inference)
feature_columns = X_train_enc.columns
X_test_enc = X_test_enc.reindex(columns=feature_columns, fill_value=0)

print("Features after encoding:", X_train_enc.shape[1])

# Save feature columns for inference
joblib.dump(list(feature_columns), COLS_PATH)
print("Saved feature columns to:", COLS_PATH)

# =========================
# CORRELATION HEATMAP
# =========================
save_correlation_heatmap(
    X_train_enc,
    out_path=os.path.join(OUTPUT_DIR, "correlation_heatmap.png"),
    title="Correlation Matrix - Tourism Features",
    top_n=20
)

print("Saved correlation heatmap.")

# =========================
# 3) Scale (fit on train, transform test)
# =========================
print("\nSTEP 3: SCALE (StandardScaler)")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_enc)
X_test_scaled = scaler.transform(X_test_enc)

joblib.dump(scaler, SCALER_PATH)
print("Saved scaler to:", SCALER_PATH)

# =========================
# 4) Build NN architecture
# =========================
print("\nSTEP 4: BUILD MODEL (same hidden layers/activations/dropouts)")

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

model = keras.Sequential([
    keras.layers.Dense(32, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(8, activation="tanh"),

    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(1, activation="sigmoid")
])

model.summary()

# Compile with custom loss
custom_loss = combined_bce_l1_weights_loss(model, alpha=1.0, beta=0.001)
compile_model_with_custom_loss(model, custom_loss, optimizer="adam")

# =========================
# 5) Train (same style: validation_split + EarlyStopping)
# =========================
print("\nSTEP 5: TRAIN MODEL")
history = model.fit(
    X_train_scaled, y_train,
    epochs=300,
    batch_size=16,
    validation_split=0.2,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]
)

save_training_curves(
    history,
    out_path=os.path.join(OUTPUT_DIR, "training_curves.png")
)
print("Saved training curves to:", os.path.join(OUTPUT_DIR, "training_curves.png"))


# =========================
# 6) Evaluate on HOLDOUT test_data.csv (your 25%)
# =========================
print("\nSTEP 6: EVALUATE ON TEST SET (holdout)")

test_pred_proba = model.predict(X_test_scaled).reshape(-1)
test_pred = (test_pred_proba > THRESHOLD).astype(int)

acc = accuracy_score(y_test, test_pred)
auc = roc_auc_score(y_test, test_pred_proba)
cm = confusion_matrix(y_test, test_pred)
report = classification_report(y_test, test_pred, digits=4)

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
# 7) Save model 
# =========================
print("\nSTEP 7: SAVE MODEL")
model.save(MODEL_PATH)
print("Saved model to:", MODEL_PATH)

print("\n" + "=" * 70)
print("COMPONENT 2: TRAINING COMPLETE")
print("=" * 70)

# Close logger
sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
