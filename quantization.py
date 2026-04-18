import numpy as np


# Standard JPEG quantization table for luminance (Y).
Q_LUMA_STD = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.int32,
)

# Standard JPEG quantization table for chrominance (Cb/Cr).
Q_CHROMA_STD = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=np.int32,
)

# Standard JPEG quantization table for chrominance (Cb/Cr).
Q_CHROMA_STD = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=np.int32,
)


def quality_scale_factor(quality: int) -> float:
    """
    Compute S from the task formula:
      S = 5000 / Quality, Quality in [1, 50)
      S = 200 - 2*Quality, Quality in [50, 100)

    We clamp quality into [1, 99] because the formula is defined for [1, 100)
    and quality=100 would lead to S=0 (degenerate quantization).
    """
    q = int(quality)
    if q < 1:
        q = 1
    if q > 99:
        q = 99

    if q < 50:
        return 5000.0 / q
    return 200.0 - 2.0 * q


def adapt_quant_table(base_table: np.ndarray, quality: int) -> np.ndarray:
    """
    Adapt quantization table to the chosen Quality by the task formula:
      q'_{y,x} = ceil( (q_{y,x} * S) / 100 )

    Then clamp each element to [1, 255] (common JPEG constraint).

    Input:
      base_table: NxM integer matrix (usually 8x8)
      quality: int
    Output:
      adapted table (np.int32)
    """
    base = np.array(base_table, dtype=np.float64)
    s = quality_scale_factor(quality)
    q_new = np.ceil((base * s) / 100.0).astype(np.int32)
    q_new = np.clip(q_new, 1, 255)
    return q_new



