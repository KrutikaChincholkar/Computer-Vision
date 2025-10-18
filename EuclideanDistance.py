import numpy as np

# Define two 3D points
p1 = np.array([3, 5, 7])
p2 = np.array([1, 1, 1])

# Euclidean Distance (L2 Norm)
ED = np.linalg.norm(p1 - p2)
print("Euclidean Distance:", ED)

# Manhattan Distance (L1 Norm)
MD = np.sum(np.abs(p1 - p2))
print("Manhattan Distance:", MD)

# Minkowski Distance (Generalized form)
p = 3   # Order parameter (p = 1 → Manhattan, p = 2 → Euclidean)
MN_D = np.power(np.sum(np.power(np.abs(p1 - p2), p)), 1/p)
print("Minkowski Distance (p=3):", MN_D)
