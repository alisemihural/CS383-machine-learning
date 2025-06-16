# author: Ali Ural
# date: 14-10-2024
# description: LAB 2 - PRINCIPAL COMPONENT ANALYSIS

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

imgdir = "./yalefaces"

X = []
filenames = []

for filename in sorted(os.listdir(imgdir)):
    parts = filename.split('.')
    if parts[0].startswith("subject"): 
        im = Image.open(os.path.join(imgdir, filename))
        im = im.resize((40, 40))

        im = np.array(im, dtype=np.float64).flatten()  
        X.append(im)
        filenames.append(filename)

X = np.vstack(X)  
print("Data matrix:", X.shape)

# PART 1: PCA Dimensionality Reduction

mean_vector = np.mean(X, axis=0)
centered_matrix = X - mean_vector

# Compute PCA
U, S, Vt = np.linalg.svd(centered_matrix, full_matrices=False)
pca_2d_projection = centered_matrix @ Vt[:2].T

# Plot 2D PCA projection
plt.figure(figsize=(8, 6))
plt.scatter(pca_2d_projection[:, 0], pca_2d_projection[:, 1], alpha=0.7)
plt.title("PCA to 2D Projection")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()

# PART 2: Eigenfaces Reconstruction

eigenvalues = S**2
variance_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)
min_k = np.argmax(variance_ratio >= 0.95) + 1

target_img = "subject02.centerlight"
if target_img in filenames:
    target_index = filenames.index(target_img)
    target_centered = centered_matrix[target_index]

    # Project and reconstruct image
    projected = target_centered @ Vt[:min_k].T
    reconstructed = projected @ Vt[:min_k] + mean_vector

    original_img = X[target_index].reshape((40, 40))
    reconstructed_img = reconstructed.reshape((40, 40))

    # Plot the figures
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_img, cmap='gray')
    plt.title(f"Reconstructed (k={min_k})")
    plt.axis("off")
    plt.show()

    print(f"Minimum k for 95% variance: {min_k}")

else:
    print(f"{target_img} not found.")