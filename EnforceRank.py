import numpy as np

# Function to enforce rank-2 constraint
def enforce_rank(F):
    # Singular Value Decomposition (SVD)
    # F = U * Σ * Vᵀ
    U, S, Vt = np.linalg.svd(F)
    
    # Set the smallest singular value to zero
    # This ensures the matrix has rank 2 instead of 3
    S[2] = 0
    
    # Reconstruct the new rank-2 matrix
    F_rank2 = U @ np.diag(S) @ Vt
    return F_rank2

# Generate a random 3x3 matrix (simulating estimated Fundamental matrix)
f_matrix = np.random.rand(3, 3)
print("Original Matrix F:")
print(f_matrix)

# Compute and print its rank
rank = np.linalg.matrix_rank(f_matrix)
print("\nRank of original matrix:", rank)

# Enforce rank-2 constraint
f_corrected = enforce_rank(f_matrix)
print("\nCorrected Matrix F (Rank-2 enforced):")
print(f_corrected)

# Compute and print the new rank
r2 = np.linalg.matrix_rank(f_corrected)
print("\nRank after enforcing constraint:", r2)
