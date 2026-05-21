import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

# =========================================
# PROJECT ROOT
# =========================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# =========================================
# PATHS
# =========================================

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data_collection",
    "data",
    "processed"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================
# LOAD DATASET
# =========================================

print("\n=================================")
print("LOADING DATASET")
print("=================================\n")

X = np.load(
    os.path.join(DATA_DIR, "X.npy")
)

y = np.load(
    os.path.join(DATA_DIR, "y.npy")
)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# =========================================
# USE SUBSET FOR FASTER TRAINING
# =========================================

MAX_SAMPLES = 30000

if len(X) > MAX_SAMPLES:

    print(f"\nUsing subset: {MAX_SAMPLES} samples")

    X = X[:MAX_SAMPLES]
    y = y[:MAX_SAMPLES]

print("\nFinal dataset size:")
print(X.shape)
print(y.shape)

# =========================================
# TRAIN / VALIDATION SPLIT
# =========================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n=================================")
print("TRAIN / VALIDATION SPLIT")
print("=================================\n")

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")

print(f"\nX_val: {X_val.shape}")
print(f"y_val: {y_val.shape}")

# =========================================
# BUILD MODEL
# =========================================

print("\n=================================")
print("BUILDING MODEL")
print("=================================\n")

model = Sequential([

    # -------------------------------------
    # First LSTM Layer
    # -------------------------------------
    LSTM(
        128,
        return_sequences=True,
        input_shape=(30, 126)
    ),

    Dropout(0.2),

    # -------------------------------------
    # Second LSTM Layer
    # -------------------------------------
    LSTM(64),

    Dropout(0.2),

    # -------------------------------------
    # Dense Layers
    # -------------------------------------
    Dense(64, activation="relu"),

    Dense(32, activation="relu"),

    # -------------------------------------
    # Output Layer
    # steering, throttle, brake
    # -------------------------------------
    Dense(3, activation="linear")

])

# =========================================
# COMPILE MODEL
# =========================================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# =========================================
# CALLBACKS
# =========================================

checkpoint = ModelCheckpoint(

    filepath=os.path.join(
        MODEL_DIR,
        "gesture_driver.keras"
    ),

    monitor="val_loss",

    save_best_only=True,

    verbose=1
)

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True,

    verbose=1
)

# =========================================
# TRAIN MODEL
# =========================================

print("\n=================================")
print("STARTING TRAINING")
print("=================================\n")

history = model.fit(

    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=30,

    batch_size=64,

    callbacks=[
        checkpoint,
        early_stopping
    ],

    verbose=1
)

# =========================================
# SAVE FINAL MODEL
# =========================================

final_model_path = os.path.join(
    MODEL_DIR,
    "gesture_driver_final.keras"
)

model.save(final_model_path)

print("\n=================================")
print("MODEL SAVED")
print("=================================\n")

print(final_model_path)

# =========================================
# PLOT TRAINING CURVES
# =========================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Train Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Training Loss Curve")

plt.legend()

plot_path = os.path.join(
    MODEL_DIR,
    "training_curve.png"
)

plt.savefig(plot_path)

print("\nTraining curve saved:")
print(plot_path)

plt.show()

# =========================================
# FINAL MESSAGE
# =========================================

print("\n=================================")
print("TRAINING COMPLETE")
print("=================================")