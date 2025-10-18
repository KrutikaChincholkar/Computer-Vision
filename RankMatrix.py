import numpy as np

# -------------------- Case 1 --------------------
# All elements are zero → Rank = 0
F0 = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
])
r0 = np.linalg.matrix_rank(F0)
print("Rank is", r0, "→ All elements are zero")
print(F0)
print("__")

# -------------------- Case 2 --------------------
# All rows are linearly dependent (row3 = 3 * row1, row2 = row1)
# So there is only one unique row → Rank = 1
F1 = np.array([
    [3, 3, 3],
    [3, 3, 3],
    [9, 9, 9]
])
r1 = np.linalg.matrix_rank(F1)
print("Rank is", r1, "→ Rows are linearly dependent (row3 = 3 × row1)")
print(F1)
print("__")

# -------------------- Case 3 --------------------
# row3 = row1 + row2 → only 2 independent rows → Rank = 2
F2 = np.array([
    [3, 4, 3],
    [2, 2, 2],
    [5, 6, 5]   # 3rd row = row1 + row2
])
r2 = np.linalg.matrix_rank(F2)
print("Rank is", r2, "→ row3 = row1 + row2 (2 independent rows)")
print(F2)
print("__")

# -------------------- Case 4 --------------------
# All rows are independent → Rank = 3
F3 = np.array([
    [3, 2, 3],
    [2, 4, 9],
    [1, 3, 4]
])
r3 = np.linalg.matrix_rank(F3)
print("Rank is", r3, "→ All three rows are independent")
print(F3)
print("__")
