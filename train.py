import json

# Load public cases
with open('public_cases.json') as f:
    cases = json.load(f)

X = []
y = []
for case in cases:
    input = case['input']
    X.append([1, input['trip_duration_days'], input['miles_traveled'], input['total_receipts_amount']])
    y.append(case['expected_output'])

# compute linear regression coefficients using normal equation
# X is n x 4 matrix; we compute (X^T X)^(-1) X^T y

# We'll implement using simple python loops and matrix operations
n = len(X)
m = len(X[0])

# Compute X^T X
xtx = [[0]*m for _ in range(m)]
for row in X:
    for i in range(m):
        for j in range(m):
            xtx[i][j] += row[i]*row[j]

# Compute X^T y
xty = [0]*m
for row, target in zip(X, y):
    for i in range(m):
        xty[i] += row[i]*target

# Function to invert 4x4 matrix using Gauss-Jordan elimination

def invert(matrix):
    size = len(matrix)
    # Augment with identity
    aug = [row[:] + [1 if i==j else 0 for j in range(size)] for i, row in enumerate(matrix)]
    # Forward elimination
    for i in range(size):
        pivot = aug[i][i]
        if pivot == 0:
            # swap with a later row
            for k in range(i+1, size):
                if aug[k][i] != 0:
                    aug[i], aug[k] = aug[k], aug[i]
                    pivot = aug[i][i]
                    break
        # normalize row
        pivot = aug[i][i]
        for j in range(2*size):
            aug[i][j] /= pivot
        # eliminate others
        for k in range(size):
            if k != i:
                factor = aug[k][i]
                for j in range(2*size):
                    aug[k][j] -= factor * aug[i][j]
    # Extract inverse
    inv = [row[size:] for row in aug]
    return inv

inv_xtx = invert(xtx)

# Multiply inv_xtx with xty to get coefficients
coeffs = [0]*m
for i in range(m):
    for j in range(m):
        coeffs[i] += inv_xtx[i][j] * xty[j]

print(coeffs)
