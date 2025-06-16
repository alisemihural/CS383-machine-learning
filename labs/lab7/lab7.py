#!/usr/bin/env python3
# author: Ali Ural
# date: 02-06-2025
# description: LAB 7 - Naive Bayes

import numpy as np
import pandas as pd
import os
from PIL import Image
from collections import Counter
import math
import random

ALPHA = 0.25

# Helpers

def shuffle_data(X, y):
    combined = list(zip(X, y))
    random.shuffle(combined)
    X_shuffled, y_shuffled = zip(*combined)

    return np.array(X_shuffled), np.array(y_shuffled)

def calculate_confusion_matrix(y_true, y_pred, class_labels):
    label_to_idx = {label: i for i, label in enumerate(class_labels)}
    num_classes = len(class_labels)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = label_to_idx[true_label]
        pred_idx = label_to_idx[pred_label]
        conf_matrix[true_idx, pred_idx] += 1
    return conf_matrix

def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

## Part 1: CTG

def preprocess_ctg_features(X_train, X_val):
    means = np.mean(X_train, axis=0)
    X_train_bin = (X_train >= means).astype(int)
    X_val_bin = (X_val >= means).astype(int)

    return X_train_bin, X_val_bin, means

def train_naive_bayes_ctg(X_train_bin, y_train, unique_classes):
    n_samples, n_features = X_train_bin.shape
    class_priors = {cls: 0.0 for cls in unique_classes}

    # P(feature_j=1 | class) and P(feature_j=0 | class)
    # conditional_probs[class_val][feature_idx][value_0_or_1]
    conditional_probs = {cls: np.zeros((n_features, 2)) for cls in unique_classes}

    class_counts = Counter(y_train)

    for cls in unique_classes:
        class_priors[cls] = class_counts[cls] / n_samples

        # Get samples for the current class
        X_class = X_train_bin[y_train == cls]
        n_class_samples = X_class.shape[0]

        for feature_idx in range(n_features):
            feature_col = X_class[:, feature_idx]
            count_1 = np.sum(feature_col == 1)
            count_0 = n_class_samples - count_1

            # P(feature_j=1 | class) with Laplace smoothing
            conditional_probs[cls][feature_idx, 1] = (count_1 + ALPHA) / (n_class_samples + 2 * ALPHA)
            # P(feature_j=0 | class) with Laplace smoothing
            conditional_probs[cls][feature_idx, 0] = (count_0 + ALPHA) / (n_class_samples + 2 * ALPHA)

    return class_priors, conditional_probs

def predict_ctg_sample(sample_bin, class_priors, conditional_probs, unique_classes):
    log_posteriors = {}
    for cls in unique_classes:
        log_posterior = np.log(class_priors[cls])
        for feature_idx, feature_value in enumerate(sample_bin):
            # feature_value is 0 or 1
            prob_feature_given_class = conditional_probs[cls][feature_idx, feature_value]
            if prob_feature_given_class > 0: # Should always be true with smoothing
                 log_posterior += np.log(prob_feature_given_class)
            else: # Fallback just in case
                 log_posterior += -np.inf
        log_posteriors[cls] = log_posterior

    return max(log_posteriors, key=log_posteriors.get)

def part1_ctg(ctg_file_path="CTG.csv"):
    print("## Part 1: CTG")
    print("-" * 30)

    # 1. Read data
    try:
        data = pd.read_csv(ctg_file_path)
    except FileNotFoundError:
        print("Error, CTG.csv not found.") 
        return

    data.dropna(subset=['NSP'], inplace=True)
    data.fillna(data.mean(), inplace=True) 

    if 'CLASS' in data.columns:
        data = data.drop(columns=['CLASS'])

    y = data['NSP'].values.astype(int) 
    X = data.drop(columns=['NSP']).values

    X_numeric = np.zeros_like(X, dtype=float)
    for i in range(X.shape[1]):
        X_numeric[:, i] = pd.to_numeric(X[:, i], errors='coerce')

    col_means = np.nanmean(X_numeric, axis=0)
    inds = np.where(np.isnan(X_numeric))
    X_numeric[inds] = np.take(col_means, inds[1])
    X = X_numeric

    # 2. Shuffle observations
    X_shuffled, y_shuffled = shuffle_data(X, y)

    # 3. Select training (2/3 round up) and validation data
    n_total = X_shuffled.shape[0]
    train_size = math.ceil(n_total * 2/3)

    X_train = X_shuffled[:train_size]
    y_train = y_shuffled[:train_size]
    X_val = X_shuffled[train_size:]
    y_val = y_shuffled[train_size:]

    print(f"Total CTG samples: {n_total}")
    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    # 4. Pre-process data
    X_train_bin, X_val_bin, feature_means = preprocess_ctg_features(X_train, X_val)

    # 5. Train Naive Bayes
    unique_classes = sorted(np.unique(y_train))
    class_priors_ctg, conditional_probs_ctg = train_naive_bayes_ctg(X_train_bin, y_train, unique_classes)

    # 6. Classify validation samples
    y_pred_ctg = [predict_ctg_sample(sample, class_priors_ctg, conditional_probs_ctg, unique_classes) for sample in X_val_bin]

    # Output
    print("\nClass Priors (CTG):")
    for cls, prior in class_priors_ctg.items():
        print(f"  P(Class={cls}): {prior:.4f}")

    accuracy_ctg = calculate_accuracy(y_val, y_pred_ctg)
    print(f"\nValidation Accuracy (CTG): {accuracy_ctg:.4f}%")

    conf_matrix_ctg = calculate_confusion_matrix(y_val, y_pred_ctg, unique_classes)
    print("\nConfusion Matrix (CTG):")
    print(f"   Classes: {unique_classes}")
    print(conf_matrix_ctg)
    print("-" * 30)

# Part 2: Yale Faces 

def load_yale_faces_data(dataset_path, img_size=(40, 40)):
    images_data = []
    labels = []
    subject_ids = [] 

    if not os.path.isdir(dataset_path):
        print("Error, Yale Faces dataset not found.")
        return None, None, None

    for filename in sorted(os.listdir(dataset_path)):
        if filename.startswith("subject") and not filename.endswith(".txt"): 
            try:
                subject_id_str = filename.split('.')[0][7:] 
                if int(subject_id_str) < 2: 
                    continue

                img_path = os.path.join(dataset_path, filename)
                img = Image.open(img_path).convert('L')  

                img_resized = img.resize(img_size, Image.NEAREST)
                img_vector = np.array(img_resized).flatten() 

                images_data.append(img_vector)
                labels.append(subject_id_str) 
                subject_ids.append(subject_id_str)
            except Exception as e:
                print(f"Warning, could not process file {filename}: {e}")
                continue

    if not images_data:
        print(f"No images loaded from {dataset_path}. Check directory structure and filenames.")
        return None, None, None

    return np.array(images_data), np.array(labels), list(set(subject_ids))


def split_yale_data_by_subject(X, y, subjects):
    X_train, y_train = [], []
    X_val, y_val = [], []

    all_subject_ids = np.unique(y)

    for subject_id in all_subject_ids:
        subject_indices = np.where(y == subject_id)[0]
        np.random.shuffle(subject_indices)

        n_subject_images = len(subject_indices)
        n_train_subject = math.ceil(n_subject_images * 2/3)

        train_indices = subject_indices[:n_train_subject]
        val_indices = subject_indices[n_train_subject:]

        X_train.extend(X[train_indices])
        y_train.extend(y[train_indices])
        X_val.extend(X[val_indices])
        y_val.extend(y[val_indices])

    return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val)

def train_naive_bayes_yale(X_train, y_train, unique_classes, num_pixel_values=256):
    n_samples, n_features = X_train.shape
    class_priors = {cls: 0.0 for cls in unique_classes}

    conditional_probs = {cls: np.zeros((n_features, num_pixel_values)) for cls in unique_classes}

    class_counts = Counter(y_train)

    for cls in unique_classes:
        class_priors[cls] = class_counts[cls] / n_samples

        X_class = X_train[y_train == cls]
        n_class_samples = X_class.shape[0]

        if n_class_samples == 0: continue 

        for feature_idx in range(n_features):
            feature_col_class = X_class[:, feature_idx] 

            value_counts = Counter(feature_col_class)

            for pixel_value in range(num_pixel_values):
                count_val = value_counts.get(pixel_value, 0)
                conditional_probs[cls][feature_idx, pixel_value] = \
                    (count_val + ALPHA) / (n_class_samples + ALPHA * num_pixel_values)

    return class_priors, conditional_probs


def predict_yale_sample(sample, class_priors, conditional_probs, unique_classes, num_pixel_values=256):
    log_posteriors = {}
    for cls in unique_classes:
        log_posterior = np.log(class_priors[cls])

        for feature_idx, pixel_value in enumerate(sample):
            pixel_value = int(pixel_value) 
            if 0 <= pixel_value < num_pixel_values:
                prob_feature_val_given_class = conditional_probs[cls][feature_idx, pixel_value]
                if prob_feature_val_given_class > 0:
                    log_posterior += np.log(prob_feature_val_given_class)
                else:
                    log_posterior += -np.inf
            else:
                log_posterior += -np.inf 
        log_posteriors[cls] = log_posterior

    return max(log_posteriors, key=log_posteriors.get)


def part2_yale_faces(yale_dataset_path="yalefaces"):
    print("\n## Part 2: Yale Faces Dataset")
    print("-" * 30)

    # 1. Load images and extract labels
    # Image size 40x40, features = 1600
    # Nearest neighbor resize, features discrete [0, 255]
    img_size = (40, 40)
    X_yale, y_yale_labels, subject_ids_yale = load_yale_faces_data(yale_dataset_path, img_size)

    if X_yale is None or y_yale_labels is None:
        print("Error, loading issues.")
        return

    print(f"Loaded images from Yale Faces dataset.")
    print(f"Number of features per image: {X_yale.shape[1]}")
    print(f"Unique subject IDs (classes): {sorted(list(np.unique(y_yale_labels)))}")


    # 2. Shuffle is implicitly handled by split_yale_data_by_subject if it shuffles within subject
    # 3. Split training and validation (2/3 of each subject for training)
    X_train_yale, y_train_yale, X_val_yale, y_val_yale = split_yale_data_by_subject(X_yale, y_yale_labels, subject_ids_yale)
    
    if X_train_yale.shape[0] == 0 or X_val_yale.shape[0] == 0:
        print("Error, training or validation set is empty for Yale Faces. Check data splitting.")
        return

    print(f"Yale Training samples: {X_train_yale.shape[0]}, Validation samples: {X_val_yale.shape[0]}")

    # 4. Train Naive Bayes
    # Features are already discrete [0, 255]
    # We will use them as discrete values for P(xi=v|y)
    unique_classes_yale = sorted(list(np.unique(y_train_yale)))
    num_pixel_values = 256

    class_priors_yale, conditional_probs_yale = train_naive_bayes_yale(X_train_yale, y_train_yale, unique_classes_yale, num_pixel_values)

    # 5. Classify validation samples
    y_pred_yale = [predict_yale_sample(sample, class_priors_yale, conditional_probs_yale, unique_classes_yale, num_pixel_values) for sample in X_val_yale]

    # Output
    print("\nClass Priors (Yale Faces):")
    for cls, prior in class_priors_yale.items():
        print(f"  P(Subject={cls}): {prior:.4f}")

    accuracy_yale = calculate_accuracy(y_val_yale, y_pred_yale)
    print(f"\nValidation Accuracy (Yale Faces): {accuracy_yale:.4f}%")

    conf_matrix_yale = calculate_confusion_matrix(y_val_yale, y_pred_yale, unique_classes_yale)
    print("\nConfusion Matrix (Yale Faces):")
    print(f"   Classes (Subjects): {unique_classes_yale}")
    df_conf_matrix = pd.DataFrame(conf_matrix_yale, index=unique_classes_yale, columns=unique_classes_yale)
    print(df_conf_matrix)
    print("-" * 30)

# Main
if __name__ == "__main__":
    # Seed for shuffling
    random.seed(42)
    np.random.seed(42)

    part1_ctg(ctg_file_path="CTG.csv")
    print("\n" + "="*50 + "\n")
    part2_yale_faces(yale_dataset_path="yalefaces")


