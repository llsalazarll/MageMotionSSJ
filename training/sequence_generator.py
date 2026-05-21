import os
import numpy as np

# =========================================
# PROJECT ROOT
# =========================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# =========================================
# PATHS
# =========================================

RECORDINGS_DIR = os.path.join(
    PROJECT_ROOT,
    "data_collection",
    "data",
    "recordings"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data_collection",
    "data",
    "processed"
)

SEQUENCE_LENGTH = 30

# Create processed directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================
# LOAD RECORDINGS
# =========================================

all_frames = []

recording_files = sorted([
    f for f in os.listdir(RECORDINGS_DIR)
    if f.endswith(".npy")
])

print("\n=================================")
print("LOADING RECORDINGS")
print("=================================")

print(f"\nFound {len(recording_files)} recording files\n")

for file in recording_files:

    path = os.path.join(RECORDINGS_DIR, file)

    data = np.load(
        path,
        allow_pickle=True
    )

    print(f"Loaded {file} -> {len(data)} frames")

    all_frames.extend(data)

print("\n=================================")
print(f"TOTAL FRAMES: {len(all_frames)}")
print("=================================\n")

# =========================================
# BUILD SEQUENCES
# =========================================

X = []
y = []

print("Generating sequences...\n")

for i in range(len(all_frames) - SEQUENCE_LENGTH):

    sequence = []

    # -------------------------------------
    # Build 30-frame sequence
    # -------------------------------------
    for j in range(SEQUENCE_LENGTH):

        frame = all_frames[i + j]

        left_hand = np.array(
            frame["left_hand"],
            dtype=np.float32
        )

        right_hand = np.array(
            frame["right_hand"],
            dtype=np.float32
        )

        # Combine hands
        features = np.concatenate([
            left_hand,
            right_hand
        ])

        sequence.append(features)

    # -------------------------------------
    # Target output
    # -------------------------------------
    target_frame = all_frames[i + SEQUENCE_LENGTH]

    target = np.array([
        target_frame["steering"],
        target_frame["throttle"],
        target_frame["brake"]
    ], dtype=np.float32)

    X.append(sequence)
    y.append(target)

# =========================================
# CONVERT TO NUMPY
# =========================================

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# =========================================
# SAVE DATASET
# =========================================

X_path = os.path.join(OUTPUT_DIR, "X.npy")
y_path = os.path.join(OUTPUT_DIR, "y.npy")

np.save(X_path, X)
np.save(y_path, y)

# =========================================
# SUMMARY
# =========================================

print("\n=================================")
print("DATASET GENERATED SUCCESSFULLY")
print("=================================\n")

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print("\nSaved Files:")
print(X_path)
print(y_path)

print("\n=================================")
print("DONE")
print("=================================")