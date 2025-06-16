# author: Ali Ural
# date: 04-06-2025
# description: PART I: YALEFACES PLOTS

import os, sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


imgdir = "./yalefaces"

X = []
first = True
for filename in os.listdir(imgdir):
	parts = filename.split('.')

	if parts[1] != 'txt':
		im = Image.open(imgdir + "/" + filename)
		im = im.resize((40,40))

		im = np.mat(im.getdata(),dtype=np.float64)

		if len(X) == 0:
			X = im
		else:
			X = np.append(X,im,axis=0)


# Setup
# Convert matrix to ndarray, and create two plots
X = np.array(X)
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# Plot 1: First Feature vs Second Feature
ax1.scatter(X[:, 0], X[:, 1], alpha=0.7)
ax1.set_title("Feature 1 vs Feature 2")
ax1.set_xlabel("Feature 1 (Pixel 0)")
ax1.set_ylabel("Feature 2 (Pixel 1)")
ax1.grid(True)

# Plot 2: Z-scored features
means = np.mean(X, axis=0)
stds = np.std(X, axis=0)
# Z-score formula
X_z = (X - means) / stds

# Plot Creation
ax2.scatter(X_z[:, 0], X_z[:, 1], alpha=0.7, color='green')

ax2.set_title("Z-scored Feature 1 vs Feature 2")
ax2.set_xlabel("Feature 1 (Z-Scored)")
ax2.set_ylabel("Feature 2 (Z-Scored)")
ax2.grid(True)

plt.tight_layout()
plt.show()
