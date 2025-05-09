import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from glob import glob
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import Xception, ResNet50
from sklearn.metrics import log_loss

# ─────────── CONFIG ───────────
CONFIG = {
    "DATA_DIR": "./DatasetSampled" ,         # Path where the dataset folder is placed
    "IMG_SIZE": 128,
    "BATCH_SIZE": 32,
    "EPOCHS": 10,
    "LEARNING_RATE": 1e-4,
    "SEED": 42,
    "OUTPUT_MODEL_PATH": "./best_model.h5"
}

# Set reproducibility
tf.random.set_seed(CONFIG["SEED"])
np.random.seed(CONFIG["SEED"])

# ─────────── DATA EXPLORATION ───────────
def analyze_image_sizes():
    """
    Analyze and display the actual image sizes present in the dataset.
    Scans up to 1000 images per class in Train, Validation, and Test folders, collects their dimensions,
    prints the most common sizes, and plots histograms of widths and heights.
    """
    splits = ["Train", "Validation", "Test"]
    sizes = []
    max_images_per_class = 1000  # Limit for faster preview
    for split in splits:
        split_path = os.path.join(CONFIG["DATA_DIR"], split)
        if not os.path.exists(split_path):
            continue
        classes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        for cls in classes:
            img_files = glob(os.path.join(split_path, cls, '*'))[:max_images_per_class]
            for img_path in img_files:
                try:
                    with Image.open(img_path) as img:
                        sizes.append(img.size)  # (width, height)
                except Exception as e:
                    print(f"Error reading {img_path}: {e}")
    if not sizes:
        print("No images found to analyze sizes.")
        return
    # Print most common sizes
    size_counts = Counter(sizes)
    print("\nMost common image sizes (width x height):")
    for size, count in size_counts.most_common(10):
        print(f"{size[0]}x{size[1]}: {count} images")
    # Plot histograms
    widths, heights = zip(*sizes)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.hist(widths, bins=30, color='blue', alpha=0.7)
    plt.title('Image Widths')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Count')
    plt.subplot(1,2,2)
    plt.hist(heights, bins=30, color='orange', alpha=0.7)
    plt.title('Image Heights')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Count')
    plt.suptitle('Distribution of Image Sizes in Dataset (Sampled)')
    plt.show()

def explore_dataset():
    """
    Explore the dataset: count images per class in each split, plot class distributions, and show sample images.
    This helps understand class balance, data quality, and what the images look like before training.
    """
    splits = ["Train", "Validation", "Test"]
    class_names = []
    counts = {split: {} for split in splits}
    print("\n--- DATASET EXPLORATION ---\n")
    for split in splits:
        split_path = os.path.join(CONFIG["DATA_DIR"], split)
        if not os.path.exists(split_path):
            print(f"Split not found: {split_path}")
            continue
        classes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        if not class_names:
            class_names = classes
        for cls in classes:
            img_files = glob(os.path.join(split_path, cls, '*'))
            counts[split][cls] = len(img_files)
            print(f"{split} - {cls}: {len(img_files)} images")
    # Plot class distribution
    fig, ax = plt.subplots(1, len(splits), figsize=(15,4))
    for i, split in enumerate(splits):
        vals = [counts[split].get(cls, 0) for cls in class_names]
        ax[i].bar(class_names, vals, color=['green','red'])
        ax[i].set_title(f"{split} set")
        ax[i].set_ylabel("# images")
    plt.suptitle("Class distribution in each split")
    plt.show()
    # Show sample images
    print("\nSample images from each class (Train set):")
    fig, axs = plt.subplots(1, len(class_names), figsize=(8,4))
    for i, cls in enumerate(class_names):
        img_dir = os.path.join(CONFIG["DATA_DIR"], "Train", cls)
        img_files = glob(os.path.join(img_dir, '*'))
        if img_files:
            img = Image.open(img_files[0])
            axs[i].imshow(img)
            axs[i].set_title(cls)
            axs[i].axis('off')
    plt.show()
    print("\n--- END DATASET EXPLORATION ---\n")
    analyze_image_sizes()

def load_datasets():
    """Loads train, validation, and test datasets from directory."""
    train_ds = image_dataset_from_directory(
        os.path.join(CONFIG["DATA_DIR"], "Train"),
        labels="inferred",
        label_mode="binary",
        batch_size=CONFIG["BATCH_SIZE"],
        image_size=(CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]),
        shuffle=True,
        seed=CONFIG["SEED"]
    )
    val_ds = image_dataset_from_directory(
        os.path.join(CONFIG["DATA_DIR"], "Validation"),
        labels="inferred",
        label_mode="binary",
        batch_size=CONFIG["BATCH_SIZE"],
        image_size=(CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]),
        shuffle=False,
        seed=CONFIG["SEED"]
    )
    test_ds = image_dataset_from_directory(
        os.path.join(CONFIG["DATA_DIR"], "Test"),
        labels="inferred",
        label_mode="binary",
        batch_size=CONFIG["BATCH_SIZE"],
        image_size=(CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]),
        shuffle=False,
        seed=CONFIG["SEED"]
    )
    return train_ds, val_ds, test_ds

def build_cnn_model():
    """Builds a simple Sequential CNN."""
    inp = layers.Input(shape=(CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"], 3))
    x = layers.Rescaling(1./255)(inp)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG["LEARNING_RATE"]),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_xception_model():
    """Builds a transfer-learning model using Xception as base."""
    base = Xception(
        weights="imagenet",
        include_top=False,
        input_shape=(CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"], 3)
    )
    base.trainable = False
    inp = layers.Input(shape=(CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"], 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG["LEARNING_RATE"]),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_resnet_model():
    """Builds a transfer-learning model using ResNet50 as base."""
    base = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"], 3)
    )
    base.trainable = False
    inp = layers.Input(shape=(CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"], 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG["LEARNING_RATE"]),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def plot_history(hist):
    """Plots training/validation loss and accuracy."""
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(hist.history["loss"], label="train_loss")
    plt.plot(hist.history["val_loss"], label="val_loss")
    plt.legend(); plt.title("Loss")
    plt.subplot(1,2,2)
    plt.plot(hist.history["accuracy"], label="train_acc")
    plt.plot(hist.history["val_accuracy"], label="val_acc")
    plt.legend(); plt.title("Accuracy")
    plt.show()

def evaluate_model(model, test_ds):
    """Evaluates model on test set and plots confusion matrix & ROC AUC."""
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred_prob = model.predict(test_ds).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    print("ROC AUC:", roc_auc_score(y_true, y_pred_prob))
    ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"]).plot()
    plt.show()
    print("Accuracy:", accuracy_score(y_true, y_pred))

def main():
    explore_dataset()  # Data exploration before training
    train_ds, val_ds, test_ds = load_datasets()
    results = []

    # 1. Simple CNN
    print("\n--- Training Simple CNN ---")
    cnn_model = build_cnn_model()
    cb = [
        callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        callbacks.ModelCheckpoint(CONFIG["OUTPUT_MODEL_PATH"].replace('.h5', '_cnn.h5'), save_best_only=True)
    ]
    hist = cnn_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG["EPOCHS"],
        callbacks=cb,
        verbose=2
    )
    # ... plot history, evaluate, and append results as before ...

    # 2. Xception
    print("\n--- Training Xception (Transfer Learning) ---")
    xception_model = build_xception_model()
    cb = [
        callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        callbacks.ModelCheckpoint(CONFIG["OUTPUT_MODEL_PATH"].replace('.h5', '_xception.h5'), save_best_only=True)
    ]
    hist = xception_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG["EPOCHS"],
        callbacks=cb,
        verbose=2
    )
    # ... plot history, evaluate, and append results as before ...

    # 3. ResNet50
    print("\n--- Training ResNet50 (Transfer Learning) ---")
    resnet_model = build_resnet_model()
    cb = [
        callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        callbacks.ModelCheckpoint(CONFIG["OUTPUT_MODEL_PATH"].replace('.h5', '_resnet.h5'), save_best_only=True)
    ]
    hist = resnet_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG["EPOCHS"],
        callbacks=cb,
        verbose=2
    )
    # ... plot history, evaluate, and append results as before ...

    # Print summary table
    print("\n================ Model Performance Summary ================")
    print(f"{'Model':<25} {'Accuracy':<12} {'ROC AUC':<12}")
    for r in results:
        print(f"{r['Model']:<25} {r['Accuracy']:<12.4f} {r['ROC AUC']:<12.4f}")
    print("========================================================\n")

if __name__ == "__main__":
    main()
