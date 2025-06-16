import numpy as np
import matplotlib.pyplot as plt

X = np.array([[3,400], [2, -200], [3, 100], [5, 650]])

mean = np.mean(X, axis=0, keepdims=True)

cov = np.cov(X,rowvar=False, ddof=1)

m = np.mean(X,axis=0,keepdims=True) #1x2
s = np.std(X,axis=0,ddof=1,keepdims=True) #1x2
Xp = (X-m)/s #4x2 via broadcasting

plt.scatter(Xp[:,0],Xp[:,1])
plt.show()
print(Xp)
# print(X.shape)