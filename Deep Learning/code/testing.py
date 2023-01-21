import numpy as np
m = np.arange(-5, 5)
m = m + [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
n = np.array([m -5, m, m+5])
mean = np.mean(n, axis=(1,2), keepdims=True)
n_centered = n - mean
std = np.std(n, axis=(1,2), keepdims=True)
n_standardized = n_centered/std
print(n_standardized)
