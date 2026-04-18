def dc_differential_encode(dc_values):
    if not dc_values:
        return []
    diffs = [dc_values[0]]
    for prev, curr in zip(dc_values, dc_values[1:]):
        diffs.append(curr - prev)
    return diffs


def dc_differential_decode(dc_diffs):
    if not dc_diffs:
        return []
    dc_values = [dc_diffs[0]]
    for d in dc_diffs[1:]:
        dc_values.append(dc_values[-1] + d)
    return dc_values


def rle_ac_encode(ac63):
    # ITU-T81: ZRL=(15,0), EOB=(0,0)
    result = []
    zero_count = 0
    for coef in ac63:
        if coef == 0:
            zero_count += 1
            if zero_count == 16:
                result.append((15, 0))
                zero_count = 0
        else:
            result.append((zero_count, int(coef)))
            zero_count = 0
    result.append((0, 0))
    return result


def rle_ac_decode(rle_pairs):
    ac = []
    for run, value in rle_pairs:
        if (run, value) == (0, 0):
            break
        if (run, value) == (15, 0):
            ac.extend([0] * 16)
            continue
        ac.extend([0] * run)
        ac.append(value)
    while len(ac) < 63:
        ac.append(0)
    return ac[:63]