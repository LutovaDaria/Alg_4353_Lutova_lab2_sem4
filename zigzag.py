import numpy as np

def zigzag_indices(n, m=None):
    if m is None: m = n
    indices = []
    for s in range(n + m - 1):
        if s % 2 == 0:
            x = min(s, n-1)
            y = s - x
            while x >= 0 and y < m:
                indices.append((x, y))
                x -= 1
                y += 1
        else:
            y = min(s, m-1)
            x = s - y
            while y >= 0 and x < n:
                indices.append((x, y))
                x += 1
                y -= 1
    return indices

def zigzag_flatten(matrix):
    indices = zigzag_indices(*matrix.shape)
    return [matrix[i, j] for i, j in indices]

def zigzag_unflatten(lst, n, m=None):
    if m is None:
        m = n
    matrix = np.zeros((n, m), dtype=np.int32)
    indices = zigzag_indices(n, m)
    for val, (i, j) in zip(lst, indices):
        matrix[i, j] = val
    return matrix