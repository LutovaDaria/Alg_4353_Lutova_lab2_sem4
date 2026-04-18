"""
Simple Huffman helpers for JPEG-like coding.

Uses standard luminance tables from huffman_tables.py.
"""

from huffman_tables import AC_LUMA_BITS, AC_LUMA_VALS, DC_LUMA_BITS, DC_LUMA_VALS


def build_huffman_codebook(bits_counts, symbols):
    """
    Build canonical Huffman map: symbol -> bitstring.
    bits_counts: list of 16 numbers (for lengths 1..16)
    symbols: list of symbols in canonical order
    """
    codebook = {}
    code = 0
    pos = 0
    for bit_len, count in enumerate(bits_counts, start=1):
        for _ in range(count):
            sym = symbols[pos]
            codebook[sym] = format(code, f"0{bit_len}b")
            code += 1
            pos += 1
        code <<= 1
    return codebook


def invert_codebook(codebook):
    """Build reverse map: bitstring -> symbol."""
    return {v: k for k, v in codebook.items()}


# Build standard tables once.
DC_HUFF = build_huffman_codebook(DC_LUMA_BITS, DC_LUMA_VALS)
AC_HUFF = build_huffman_codebook(AC_LUMA_BITS, AC_LUMA_VALS)
DC_HUFF_INV = invert_codebook(DC_HUFF)
AC_HUFF_INV = invert_codebook(AC_HUFF)


def encode_dc_symbol(category):
    """
    Encode DC category (size) to Huffman bits.
    category in [0..11] for baseline luminance table.
    """
    return DC_HUFF[int(category)]


def encode_ac_symbol(run, size):
    """
    Encode AC (run,size) pair to Huffman bits.
    Symbol format: (run << 4) | size.
    """
    symbol = (int(run) << 4) | int(size)
    return AC_HUFF[symbol]


def read_huffman_symbol(bits, pos, inverse_table):
    """
    Read one Huffman symbol from bitstring.
    Returns (symbol, new_pos).
    """
    code = ""
    while pos < len(bits):
        code += bits[pos]
        pos += 1
        if code in inverse_table:
            return inverse_table[code], pos
    raise ValueError("Invalid Huffman stream")
