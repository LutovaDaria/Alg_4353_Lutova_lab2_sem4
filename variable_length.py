def amplitude_category(value):
    if value == 0:
        return 0
    return abs(int(value)).bit_length()


def amplitude_bits(value, size):
    if size == 0:
        return ""
    if value > 0:
        return format(int(value), f"0{size}b")
    val = (1 << size) - 1 + int(value)
    return format(val, f"0{size}b")


def decode_amplitude(bits, size):
    if size == 0:
        return 0
    v = int(bits, 2)
    if v >= (1 << (size - 1)):
        return v
    return v - ((1 << size) - 1)