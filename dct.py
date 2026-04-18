import numpy as np
import math

def split_blocks(arr, block_h=8, block_w=8):
    h, w = arr.shape
    blocks = []
    for i in range(0, h, block_h):
        for j in range(0, w, block_w):
            block = arr[i:i + block_h, j:j + block_w]
            if block.shape == (block_h, block_w):
                blocks.append(block)
    return blocks


def split_blocks_with_padding(arr, block_h=8, block_w=8):
    """
    Split 2D array into blocks. If a block is incomplete (right/bottom edges),
    pad it with the mean of the existing pixels in that block (task requirement).

    Returns:
      blocks: list of (block_h, block_w) float64 arrays
      grid_h, grid_w: number of blocks by height/width
      padded_h, padded_w: padded image size (multiples of block size)
    """
    if arr.ndim != 2:
        raise ValueError("split_blocks_with_padding expects a 2D array")

    h, w = arr.shape
    grid_h = (h + block_h - 1) // block_h
    grid_w = (w + block_w - 1) // block_w
    padded_h = grid_h * block_h
    padded_w = grid_w * block_w

    blocks = []
    for bi in range(grid_h):
        for bj in range(grid_w):
            y0 = bi * block_h
            x0 = bj * block_w
            part = arr[y0 : min(y0 + block_h, h), x0 : min(x0 + block_w, w)]
            if part.shape == (block_h, block_w):
                blocks.append(part.astype(np.float64))
                continue

            fill = float(np.mean(part)) if part.size else 0.0
            block = np.full((block_h, block_w), fill, dtype=np.float64)
            block[: part.shape[0], : part.shape[1]] = part
            blocks.append(block)

    return blocks, grid_h, grid_w, padded_h, padded_w


def merge_blocks(blocks, grid_h, grid_w, block_h=8, block_w=8):
    """
    Merge blocks back into a 2D array of shape (grid_h*block_h, grid_w*block_w).
    """
    out = np.zeros((grid_h * block_h, grid_w * block_w), dtype=np.float64)
    k = 0
    for bi in range(grid_h):
        for bj in range(grid_w):
            out[
                bi * block_h : (bi + 1) * block_h,
                bj * block_w : (bj + 1) * block_w,
            ] = blocks[k]
            k += 1
    return out


def dct2_naive(block):
    block = block.astype(np.float64)
    n, m = block.shape
    out = np.zeros((n, m), dtype=np.float64)
    for u in range(n):
        for v in range(m):
            alpha_u = math.sqrt(1 / n) if u == 0 else math.sqrt(2 / n)
            alpha_v = math.sqrt(1 / m) if v == 0 else math.sqrt(2 / m)
            s = 0.0
            for x in range(n):
                for y in range(m):
                    s += (
                        block[x, y]
                        * math.cos(((2 * x + 1) * u * math.pi) / (2 * n))
                        * math.cos(((2 * y + 1) * v * math.pi) / (2 * m))
                    )
            out[u, v] = alpha_u * alpha_v * s
    return out


def idct2_naive(coeffs):
    coeffs = coeffs.astype(np.float64)
    n, m = coeffs.shape
    out = np.zeros((n, m), dtype=np.float64)
    for x in range(n):
        for y in range(m):
            s = 0.0
            for u in range(n):
                for v in range(m):
                    alpha_u = math.sqrt(1 / n) if u == 0 else math.sqrt(2 / n)
                    alpha_v = math.sqrt(1 / m) if v == 0 else math.sqrt(2 / m)
                    s += (
                        alpha_u
                        * alpha_v
                        * coeffs[u, v]
                        * math.cos(((2 * x + 1) * u * math.pi) / (2 * n))
                        * math.cos(((2 * y + 1) * v * math.pi) / (2 * m))
                    )
            out[x, y] = s
    return out


def dct_matrix(n):
    c = np.zeros((n, n), dtype=np.float64)
    for u in range(n):
        alpha = math.sqrt(1 / n) if u == 0 else math.sqrt(2 / n)
        for x in range(n):
            c[u, x] = alpha * math.cos(((2 * x + 1) * u * math.pi) / (2 * n))
    return c


def dct2_matrix(block):
    block = block.astype(np.float64)
    n, m = block.shape
    cn = dct_matrix(n)
    cm = dct_matrix(m)

    temp = np.dot(cn, block)
    result = np.dot(temp, cm.T)
    return result


def idct2_matrix(coeffs):
    coeffs = coeffs.astype(np.float64)
    n, m = coeffs.shape
    cn = dct_matrix(n)
    cm = dct_matrix(m)

    temp = np.dot(cn.T, coeffs)
    result = np.dot(temp, cm)
    return result


def quantize(coeffs, q_table):
    return np.round(coeffs / q_table).astype(np.int32)


def dequantize(coeffs_q, q_table):
    return coeffs_q.astype(np.float64) * q_table.astype(np.float64)