import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 5
SEED = 42
CLASSES = ["Negative", "Positive"]


def parse_args():
    parser = argparse.ArgumentParser(description="Concrete crack detection with a CNN")
    parser.add_argument("--data-dir", default="archive")
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def save_figure(output_dir, name):
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, name), dpi=300, bbox_inches="tight")
    plt.close()


def load_dataframe(data_dir):
    rows = []
    for label in CLASSES:
        class_dir = os.path.join(data_dir, label)
        for name in sorted(os.listdir(class_dir)):
            rows.append([os.path.join(class_dir, name), label])
    return pd.DataFrame(rows, columns=["filepath", "label"])


def plot_class_distribution(df, output_dir):
    counts = df["label"].value_counts()
    plt.figure(figsize=(6, 5))
    plt.bar(["Positive", "Negative"], [counts["Positive"], counts["Negative"]])
    plt.title("Distribution of images by class")
    plt.xlabel("Class")
    plt.ylabel("Number of images")
    save_figure(output_dir, "class_distribution.png")


def plot_dataset_examples(df, output_dir):
    positive = df[df["label"] == "Positive"].sample(5, random_state=SEED)
    negative = df[df["label"] == "Negative"].sample(5, random_state=SEED)
    samples = pd.concat([positive, negative])
    plt.figure(figsize=(15, 6))
    for i, (path, label) in enumerate(samples[["filepath", "label"]].values):
        plt.subplot(2, 5, i + 1)
        plt.imshow(Image.open(path))
        plt.title(label, fontsize=10)
        plt.axis("off")
    plt.suptitle("Examples of images from the dataset", fontsize=16)
    save_figure(output_dir, "dataset_examples.png")


def split_dataframe(df):
    train_df, rest_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        rest_df, test_size=0.50, stratify=rest_df["label"], random_state=SEED
    )
    return train_df, val_df, test_df


def create_generators(train_df, val_df, test_df):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    common = dict(
        x_col="filepath",
        y_col="label",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=CLASSES,
    )
    train_generator = train_datagen.flow_from_dataframe(
        train_df, shuffle=True, seed=SEED, **common
    )
    val_generator = val_test_datagen.flow_from_dataframe(val_df, shuffle=False, **common)
    test_generator = val_test_datagen.flow_from_dataframe(test_df, shuffle=False, **common)
    return train_datagen, train_generator, val_generator, test_generator


def plot_augmentation(train_datagen, df, output_dir):
    path = df[df["label"] == "Positive"].sample(1, random_state=SEED)["filepath"].iloc[0]
    img = tf.keras.utils.load_img(path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = np.expand_dims(tf.keras.utils.img_to_array(img), axis=0)

    plt.figure(figsize=(10, 10))
    plt.subplot(3, 3, 1)
    plt.imshow(img)
    plt.title("Original image")
    plt.axis("off")

    flow = train_datagen.flow(img_array, batch_size=1, seed=SEED)
    for i in range(2, 10):
        batch = next(flow)
        plt.subplot(3, 3, i)
        plt.imshow(batch[0])
        plt.title(f"Augmentation {i - 1}")
        plt.axis("off")
    plt.suptitle("Examples of augmented images", fontsize=16)
    save_figure(output_dir, "augmentation_examples.png")


def build_model():
    model = Sequential([
        Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def plot_training_history(history, output_dir):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], marker="o", linewidth=2, label="Training accuracy")
    plt.plot(history.history["val_accuracy"], marker="o", linewidth=2, label="Validation accuracy")
    plt.title("Model accuracy during training", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], marker="o", linewidth=2, label="Training loss")
    plt.plot(history.history["val_loss"], marker="o", linewidth=2, label="Validation loss")
    plt.title("Model loss during training", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    save_figure(output_dir, "training_performance.png")


def plot_confusion_matrix(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Confusion matrix")
    save_figure(output_dir, "confusion_matrix.png")


def plot_predictions(filepaths, y_true, y_pred, indices, title, color, show_prediction, filename, output_dir):
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(indices[:10]):
        plt.subplot(2, 5, i + 1)
        plt.imshow(Image.open(filepaths[idx]))
        text = f"T:{CLASSES[y_true[idx]]}"
        if show_prediction:
            text += f"\nP:{CLASSES[y_pred[idx]]}"
        plt.title(text, color=color)
        plt.axis("off")
    plt.suptitle(title)
    save_figure(output_dir, filename)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    df = load_dataframe(args.data_dir)
    print("Positive images:", (df["label"] == "Positive").sum())
    print("Negative images:", (df["label"] == "Negative").sum())
    print("Total images:", len(df))
    plot_class_distribution(df, args.output_dir)

    train_df, val_df, test_df = split_dataframe(df)
    print("Train size:", len(train_df))
    print("Validation size:", len(val_df))
    print("Test size:", len(test_df))
    plot_dataset_examples(df, args.output_dir)

    train_datagen, train_generator, val_generator, test_generator = create_generators(
        train_df, val_df, test_df
    )
    print(train_generator.class_indices)
    plot_augmentation(train_datagen, train_df, args.output_dir)

    model = build_model()
    model.summary()

    history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)
    model.save(os.path.join(args.output_dir, "cnn_model.keras"))
    plot_training_history(history, args.output_dir)

    print("Final training accuracy:", history.history["accuracy"][-1])
    print("Final validation accuracy:", history.history["val_accuracy"][-1])
    print("Final training loss:", history.history["loss"][-1])
    print("Final validation loss:", history.history["val_loss"][-1])

    probabilities = model.predict(test_generator).flatten()
    y_pred = (probabilities > 0.5).astype(int)
    y_true = np.array(test_generator.classes)

    print("Test accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))
    plot_confusion_matrix(y_true, y_pred, args.output_dir)

    filepaths = test_generator.filepaths
    correct = np.where(y_pred == y_true)[0]
    incorrect = np.where(y_pred != y_true)[0]
    plot_predictions(filepaths, y_true, y_pred, incorrect, "Misclassified images", "red", True,
                     "incorrect_predictions.png", args.output_dir)
    plot_predictions(filepaths, y_true, y_pred, correct, "Correctly classified images", "green", False,
                     "correct_predictions.png", args.output_dir)


if __name__ == "__main__":
    main()
