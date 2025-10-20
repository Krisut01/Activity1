# ONE FILE: Kaggle Digit Recognizer (MNIST CSV) end-to-end
# - Loads train.csv & test.csv
# - Trains a Keras MLP with Softmax
# - Shows sample digits & predictions
# - Writes submission.csv for Kaggle
# - Creates test1.csv from your handwritten digits
# Run this where train.csv and test.csv are present.

import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image, ImageOps

# -----------------------
# 0) Reproducibility
# -----------------------
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------
# 1) Load data
# -----------------------
train_df = pd.read_csv("train.csv")   # columns: label, pixel0..pixel783
test_df  = pd.read_csv("test.csv")    # columns: pixel0..pixel783

y_raw = train_df["label"].values
X_raw = train_df.drop(columns=["label"]).values
X_test_raw = test_df.values

# -----------------------
# 2) Preprocess
# -----------------------
X = (X_raw.astype("float32")) / 255.0
X_test = (X_test_raw.astype("float32")) / 255.0
y = to_categorical(y_raw, num_classes=10)

# Train/validation split (10% holdout)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=SEED, stratify=y_raw
)

# -----------------------
# 3) Visualize sample digits (optional)
# -----------------------
def show_digits(matrix, n=16, title=None, save_path=None):
    n = min(n, matrix.shape[0])
    side = math.ceil(math.sqrt(n))
    plt.figure(figsize=(5,5))
    for i in range(n):
        plt.subplot(side, side, i+1)
        plt.imshow(matrix[i].reshape(28,28), cmap=plt.cm.binary)
        plt.axis("off")
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    # Save instead of blocking show, so the script continues
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = save_path or os.path.join(script_dir, "digits_preview.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved preview: {out_path}")
    except Exception as ex:
        print(f"Failed to save preview image: {ex}")
    finally:
        plt.close()

# -----------------------
# 4) Build MLP model
# -----------------------
model = Sequential([
    Dense(512, activation="relu", input_shape=(784,)),
    Dropout(0.2),
    Dense(256, activation="relu"),
    Dropout(0.2),
    Dense(10, activation="softmax")  # use softmax for multi-class
])

model.compile(
    optimizer="sgd",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------
# 5) Train
# -----------------------
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

# -----------------------
# 6) Evaluate
# -----------------------
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation accuracy: {val_acc:.4f}")

# -----------------------
# 7) Predict test set & save submission
# -----------------------
probs = model.predict(X_test, verbose=0)
pred_labels = np.argmax(probs, axis=1)
submission = pd.DataFrame({
    "ImageId": np.arange(1, len(pred_labels)+1),
    "Label": pred_labels
})
submission.to_csv("submission.csv", index=False)
print("Saved: submission.csv")

model.save("model.h5")
print("Saved trained model: model.h5")

# -----------------------
# 8) Show first 100 test digits
# -----------------------
def show_test_preds(Xt, preds, n=100, save_path=None):
    n = min(n, Xt.shape[0])
    side = math.ceil(math.sqrt(n))
    plt.figure(figsize=(5,5))
    for i in range(n):
        plt.subplot(side, side, i+1)
        plt.imshow(Xt[i].reshape(28,28), cmap=plt.cm.binary)
        plt.title(int(preds[i]))
        plt.axis("off")
    plt.tight_layout()
    # Save instead of blocking show, so the script continues
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = save_path or os.path.join(script_dir, "test_predictions_preview.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved preview: {out_path}")
    except Exception as ex:
        print(f"Failed to save preview image: {ex}")
    finally:
        plt.close()

show_test_preds(X_test, pred_labels, n=100)

# -----------------------
# 9) Load handwritten digits & create test1.csv
# -----------------------
def load_and_prepare_my_digits(folder_path="my_digits"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = folder_path if os.path.isabs(folder_path) else os.path.join(script_dir, folder_path)
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return None, None

    print("Files detected in my_digits:", os.listdir(folder_path))
    desired_names = [f"{d}.png" for d in range(9, -1, -1)]
    files_in_dir = {name.lower(): os.path.join(folder_path, name) for name in os.listdir(folder_path)}

    ordered_paths = []
    for name in desired_names:
        match = files_in_dir.get(name.lower())
        if match:
            ordered_paths.append(match)
        else:
            print(f"Warning: {name} not found in folder. Skipping.")

    if not ordered_paths:
        print("No images found. Skipping custom digits.")
        return None, None

    processed, descs = [], []
    # Determine a robust LANCZOS resample filter (compat with older Pillow)
    try:
        lanczos_filter = Image.Resampling.LANCZOS  # Pillow >= 9.1
    except Exception:
        lanczos_filter = Image.LANCZOS  # fallback

    for p in ordered_paths:
        try:
            img = Image.open(p).convert("L")
            img = img.resize((28,28), lanczos_filter)
            arr = np.array(img, dtype=np.uint8)
            if arr.mean() > 127:
                arr = np.array(ImageOps.invert(Image.fromarray(arr)), dtype=np.uint8)
            arr = arr.astype("float32") / 255.0
            processed.append(arr.flatten())
            descs.append(os.path.basename(p))
        except Exception as ex:
            print(f"Failed to process '{p}': {ex}")

    if not processed:
        print("No valid images processed. Skipping custom digits.")
        return None, None

    return np.stack(processed, axis=0), descs

def export_test1_csv(pixels_2d, csv_path="test1.csv"):
    if pixels_2d is None:
        print(f"No pixels to export. Skipping {csv_path}")
        return
    pixels_255 = (pixels_2d * 255.0).round().astype(np.uint8)
    cols = [f"pixel{i}" for i in range(28*28)]
    df = pd.DataFrame(pixels_255, columns=cols)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = csv_path if os.path.isabs(csv_path) else os.path.join(script_dir, csv_path)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

# Prepare & predict handwritten digits
my_pixels, my_descs = load_and_prepare_my_digits("my_digits")

if my_pixels is not None:
    export_test1_csv(my_pixels, csv_path="test1.csv")
    print("Shape of my_pixels:", my_pixels.shape)

    my_probs = model.predict(my_pixels, verbose=0)
    my_preds = np.argmax(my_probs, axis=1)
    results_df = pd.DataFrame({
        "Source": my_descs,
        "PredictedLabel": my_preds
    })
    results_df.to_csv("my_digits_predictions.csv", index=False)
    print("Saved: my_digits_predictions.csv")
    print(results_df)

    # Reload test1.csv from disk and predict again to demonstrate feeding CSV to the model
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        test1_path = os.path.join(script_dir, "test1.csv")
        print(f"Reloading CSV from: {test1_path}")
        test1_df = pd.read_csv(test1_path)
        test1_pixels = test1_df.values.astype("float32") / 255.0
        csv_probs = model.predict(test1_pixels, verbose=0)
        csv_preds = np.argmax(csv_probs, axis=1)
        csv_results_df = pd.DataFrame({
            "Source": my_descs[: len(csv_preds)],
            "PredictedLabel": csv_preds
        })
        out_csv_from_csv = os.path.join(script_dir, "my_digits_predictions_from_csv.csv")
        csv_results_df.to_csv(out_csv_from_csv, index=False)
        print(f"Saved: {out_csv_from_csv}")

        # Quick consistency check
        if len(csv_preds) == len(my_preds) and np.all(csv_preds == my_preds):
            print("CSV prediction matches in-memory prediction for all rows.")
        else:
            print("Note: CSV prediction differs from in-memory prediction or lengths differ.")
    except Exception as ex:
        print(f"Failed to reload and predict from test1.csv: {ex}")
else:
    print("No handwritten digits processed. test1.csv not created.")
