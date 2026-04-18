import json
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import dct
import huffman_codec
import quantization
import zigzag


# -----------------------------
# Small helpers (bits + padding)
# -----------------------------

def bits_to_bytes(bitstring: str) -> tuple[bytes, int]:
    """Pack '0101...' into bytes. Returns (bytes, bit_count_without_padding)."""
    bit_count = len(bitstring)
    pad = (8 - (bit_count % 8)) % 8
    bitstring += "0" * pad
    out = bytearray()
    for i in range(0, len(bitstring), 8):
        out.append(int(bitstring[i:i + 8], 2))
    return bytes(out), bit_count


def bytes_to_bits(data: bytes, bit_count: int) -> str:
    bits = "".join(format(b, "08b") for b in data)
    return bits[:bit_count]


def pad_to_8(channel: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Pad to multiples of 8 by edge replication."""
    h, w = channel.shape
    hp = (h + 7) // 8 * 8
    wp = (w + 7) // 8 * 8
    out = np.zeros((hp, wp), dtype=np.float64)
    out[:h, :w] = channel.astype(np.float64)
    if hp > h:
        out[h:hp, :w] = out[h - 1:h, :w]
    if wp > w:
        out[:, w:wp] = out[:, w - 1:w]
    return out, h, w


def split_blocks_8(channel: np.ndarray) -> list[np.ndarray]:
    blocks = []
    h, w = channel.shape
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            blocks.append(channel[i:i + 8, j:j + 8])
    return blocks


def merge_blocks_8(blocks: list[np.ndarray], h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=np.float64)
    idx = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            out[i:i + 8, j:j + 8] = blocks[idx]
            idx += 1
    return out


# -----------------------------
# 2.4 VLC amplitude coding (task)
# -----------------------------

def amplitude_category(value: int) -> int:
    if value == 0:
        return 0
    return abs(int(value)).bit_length()


def amplitude_bits(value: int, size: int) -> str:
    if size == 0:
        return ""
    value = int(value)
    if value > 0:
        return format(value, f"0{size}b")
    mapped = (1 << size) - 1 + value
    return format(mapped, f"0{size}b")


def decode_amplitude(bits: str, size: int) -> int:
    if size == 0:
        return 0
    v = int(bits, 2)
    if v >= (1 << (size - 1)):
        return v
    return v - ((1 << size) - 1)


# -----------------------------
# 2.3 DC differential + AC RLE (task)
# -----------------------------

def dc_differential_encode(dc_values: list[int]) -> list[int]:
    if not dc_values:
        return []
    diffs = [int(dc_values[0])]
    for prev, curr in zip(dc_values, dc_values[1:]):
        diffs.append(int(curr) - int(prev))
    return diffs


def dc_differential_decode(dc_diffs: list[int]) -> list[int]:
    if not dc_diffs:
        return []
    vals = [int(dc_diffs[0])]
    for d in dc_diffs[1:]:
        vals.append(vals[-1] + int(d))
    return vals


def rle_ac_encode(ac63: list[int]) -> list[tuple[int, int]]:
    # ZRL=(15,0), EOB=(0,0)
    result = []
    zero_count = 0
    for coef in ac63:
        coef = int(coef)
        if coef == 0:
            zero_count += 1
            if zero_count == 16:
                result.append((15, 0))
                zero_count = 0
        else:
            result.append((zero_count, coef))
            zero_count = 0
    result.append((0, 0))
    return result


def rle_ac_decode(pairs: list[tuple[int, int]]) -> list[int]:
    ac = []
    for run, value in pairs:
        if (run, value) == (0, 0):
            break
        if (run, value) == (15, 0):
            ac.extend([0] * 16)
            continue
        ac.extend([0] * run)
        ac.append(int(value))
    while len(ac) < 63:
        ac.append(0)
    return ac[:63]


# -----------------------------
# Color conversion (simple)
# -----------------------------

def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128
    return np.clip(np.stack([y, cb, cr], axis=-1), 0, 255).astype(np.uint8)


def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    ycbcr = ycbcr.astype(np.float32)
    y, cb, cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)


# -----------------------------
# Channel encode/decode (2.2-2.5)
# -----------------------------

def encode_channel(channel_u8: np.ndarray, qtable: np.ndarray) -> tuple[bytes, dict]:
    padded, orig_h, orig_w = pad_to_8(channel_u8)
    hp, wp = padded.shape
    blocks = split_blocks_8(padded)

    dc_vals: list[int] = []
    ac_vals: list[list[int]] = []

    for b in blocks:
        coeff = dct.dct2_matrix(b - 128.0)
        q = dct.quantize(coeff, qtable)
        zz = zigzag.zigzag_flatten(q)
        dc_vals.append(int(zz[0]))
        ac_vals.append([int(v) for v in zz[1:]])

    dc_diffs = dc_differential_encode(dc_vals)

    bits = ""
    # DC: Huffman(size) + amplitude bits
    for diff in dc_diffs:
        size = amplitude_category(diff)
        bits += huffman_codec.encode_dc_symbol(size)
        bits += amplitude_bits(diff, size)

    # AC: Huffman(run/size) + amplitude bits
    for ac in ac_vals:
        pairs = rle_ac_encode(ac)
        for run, value in pairs:
            if (run, value) == (0, 0):
                bits += huffman_codec.AC_HUFF[0x00]  # EOB
                continue
            if (run, value) == (15, 0):
                bits += huffman_codec.AC_HUFF[0xF0]  # ZRL
                continue
            size = amplitude_category(value)
            bits += huffman_codec.encode_ac_symbol(run, size)
            bits += amplitude_bits(value, size)

    payload, bit_count = bits_to_bytes(bits)
    meta = {
        "orig_h": int(orig_h),
        "orig_w": int(orig_w),
        "pad_h": int(hp),
        "pad_w": int(wp),
        "blocks": int(len(blocks)),
        "bit_count": int(bit_count),
        "byte_count": int(len(payload)),
    }
    return payload, meta


def decode_channel(payload: bytes, meta: dict, qtable: np.ndarray) -> np.ndarray:
    bits = bytes_to_bits(payload, int(meta["bit_count"]))
    pos = 0
    blocks_count = int(meta["blocks"])

    # DC
    dc_diffs = []
    for _ in range(blocks_count):
        size, pos = huffman_codec.read_huffman_symbol(bits, pos, huffman_codec.DC_HUFF_INV)
        size = int(size)
        amp = bits[pos:pos + size]
        pos += size
        dc_diffs.append(decode_amplitude(amp, size))
    dc_vals = dc_differential_decode(dc_diffs)

    # AC per block
    restored_blocks = []
    for bi in range(blocks_count):
        ac = []
        # Important: consume symbols until EOB, even if we already reached 63 values.
        # Otherwise we can desync the stream for the next block.
        while True:
            sym, pos = huffman_codec.read_huffman_symbol(bits, pos, huffman_codec.AC_HUFF_INV)
            sym = int(sym)

            if sym == 0x00:  # EOB
                break

            if sym == 0xF0:  # ZRL = 16 zeros
                need = max(0, 63 - len(ac))
                ac.extend([0] * min(16, need))
                continue

            run = (sym >> 4) & 0x0F
            size = sym & 0x0F

            need_zeros = max(0, 63 - len(ac))
            ac.extend([0] * min(run, need_zeros))

            amp = bits[pos:pos + size]
            pos += size
            val = decode_amplitude(amp, size)

            if len(ac) < 63:
                ac.append(val)

            # If we already have 63, keep reading until EOB but ignore extra values.

        while len(ac) < 63:
            ac.append(0)

        zz = [int(dc_vals[bi])] + ac[:63]
        q_block = zigzag.zigzag_unflatten(zz, 8, 8).astype(np.float64)
        coeff = dct.dequantize(q_block, qtable)
        block = dct.idct2_matrix(coeff) + 128.0
        restored_blocks.append(np.clip(block, 0, 255))

    pad_h = int(meta["pad_h"])
    pad_w = int(meta["pad_w"])
    channel = merge_blocks_8(restored_blocks, pad_h, pad_w)
    channel = channel[: int(meta["orig_h"]), : int(meta["orig_w"])]
    return channel.astype(np.uint8)


# -----------------------------
# 2.7 File format: JSON header + raw payload
# -----------------------------

def write_container(path: Path, header: dict, payload: bytes) -> None:
    header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")
    with open(path, "wb") as f:
        f.write(b"SJPG\n")
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)
        f.write(payload)


def read_container(path: Path) -> tuple[dict, bytes]:
    with open(path, "rb") as f:
        magic = f.read(5)
        if magic != b"SJPG\n":
            raise ValueError("Bad container format")
        header_len = struct.unpack("<I", f.read(4))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        payload = f.read()
    return header, payload


# -----------------------------
# High-level API
# -----------------------------

def compress_image(input_path: str, output_path: str, quality: int = 70, stage_log: list[str] | None = None) -> None:
    img = Image.open(input_path).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)
    ycbcr = rgb_to_ycbcr(rgb)
    if stage_log is not None:
        stage_log.append(f"[СЖАТИЕ] Исходное изображение: {input_path}")
        stage_log.append(f"[СЖАТИЕ] Размер: {rgb.shape[1]}x{rgb.shape[0]}, каналов: {rgb.shape[2]}")
        stage_log.append("[СЖАТИЕ] Перевод RGB -> YCbCr выполнен")

    # JPEG-like: use different tables for luma (Y) and chroma (Cb/Cr).
    q_y = quantization.adapt_quant_table(quantization.Q_LUMA_STD, quality).astype(np.float64)
    q_c = quantization.adapt_quant_table(quantization.Q_CHROMA_STD, quality).astype(np.float64)
    if stage_log is not None:
        stage_log.append(f"[СЖАТИЕ] Quality = {quality}")
        stage_log.append(
            f"[СЖАТИЕ] Q(Y): min={int(q_y.min())}, max={int(q_y.max())}; "
            f"Q(C): min={int(q_c.min())}, max={int(q_c.max())}"
        )

    payload_parts = []
    channels_meta = []
    for ch in range(3):
        qtab = q_y if ch == 0 else q_c
        part, meta = encode_channel(ycbcr[..., ch], qtab)
        meta["q"] = "Y" if ch == 0 else "C"
        payload_parts.append(part)
        channels_meta.append(meta)
        if stage_log is not None:
            name = "Y" if ch == 0 else ("Cb" if ch == 1 else "Cr")
            stage_log.append(
                f"[СЖАТИЕ] Канал {name}: блоков={meta['blocks']}, бит={meta['bit_count']}, байт={meta['byte_count']}"
            )

    header = {
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "color_space": "YCbCr",
        "quality": int(quality),
        "quant_tables": {
            "Y": q_y.astype(int).tolist(),
            "C": q_c.astype(int).tolist(),
        },
        "dc_table": "STD_LUMA",
        "ac_table": "STD_LUMA",
        "channels_meta": channels_meta,
    }
    payload = b"".join(payload_parts)
    write_container(Path(output_path), header, payload)
    if stage_log is not None:
        stage_log.append(f"[СЖАТИЕ] Контейнер записан: {output_path}")
        stage_log.append(f"[СЖАТИЕ] Размер контейнера (payload): {len(payload)} байт")


def decompress_image(input_path: str, output_image_path: str, stage_log: list[str] | None = None) -> None:
    header, payload = read_container(Path(input_path))
    q_y = np.array(header["quant_tables"]["Y"], dtype=np.float64)
    q_c = np.array(header["quant_tables"]["C"], dtype=np.float64)
    if stage_log is not None:
        stage_log.append(f"[ДЕКОМПРЕССИЯ] Чтение контейнера: {input_path}")
        stage_log.append(
            f"[ДЕКОМПРЕССИЯ] Метаданные: {header['width']}x{header['height']}, quality={header['quality']}, payload={len(payload)} байт"
        )

    channels = []
    offset = 0
    for meta in header["channels_meta"]:
        byte_count = int(meta["byte_count"])
        part = payload[offset: offset + byte_count]
        offset += byte_count
        qtab = q_y if meta.get("q") == "Y" else q_c
        channels.append(decode_channel(part, meta, qtab))
        if stage_log is not None:
            qname = meta.get("q", "?")
            stage_log.append(
                f"[ДЕКОМПРЕССИЯ] Канал {qname}: считано {byte_count} байт, блоков={meta['blocks']}, pad={meta['pad_w']}x{meta['pad_h']}"
            )

    ycbcr = np.stack(channels, axis=-1).astype(np.uint8)
    rgb = ycbcr_to_rgb(ycbcr)
    Image.fromarray(rgb, mode="RGB").save(output_image_path)
    if stage_log is not None:
        stage_log.append(f"[ДЕКОМПРЕССИЯ] Восстановленное изображение сохранено: {output_image_path}")


def compute_metrics(original_path: Path, restored_path: Path) -> dict:
    orig = np.array(Image.open(original_path).convert("RGB"), dtype=np.float64)
    rec = np.array(Image.open(restored_path).convert("RGB"), dtype=np.float64)
    if orig.shape != rec.shape:
        return {"error": f"shape mismatch: {orig.shape} vs {rec.shape}"}

    diff = orig - rec
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    max_abs = int(np.max(np.abs(diff)))
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = float(10.0 * np.log10((255.0 ** 2) / mse))
    return {"mse": mse, "mae": mae, "max_abs": max_abs, "psnr": psnr}


def build_test_images(base_dir: Path, out_dir: Path) -> dict[str, Path]:
    """
    Build test image set:
    - original RGB
    - grayscale
    - BW by rounding round(c/255)*255
    - BW with dithering
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    source_candidates = [
        ("lena", base_dir / "lenna.png"),
        ("color", base_dir / "color.png"),
    ]
    found = {}
    for name, p in source_candidates:
        if p.exists():
            found[name] = p

    if not found:
        raise FileNotFoundError("Put at least one source image: lenna.png or color.png")

    test_images: dict[str, Path] = {}
    for base_name, src in found.items():
        rgb = Image.open(src).convert("RGB")
        rgb_path = out_dir / f"{base_name}_rgb.png"
        rgb.save(rgb_path)
        test_images[f"{base_name}_rgb"] = rgb_path

        gray = rgb.convert("L")
        gray_path = out_dir / f"{base_name}_gray.png"
        gray.save(gray_path)
        test_images[f"{base_name}_gray"] = gray_path

        bw_round = gray.point(lambda c: round(c / 255) * 255)
        bw_round_path = out_dir / f"{base_name}_bw_round.png"
        bw_round.save(bw_round_path)
        test_images[f"{base_name}_bw_round"] = bw_round_path

        bw_dither = gray.convert(mode="1")
        bw_dither_path = out_dir / f"{base_name}_bw_dither.png"
        bw_dither.save(bw_dither_path)
        test_images[f"{base_name}_bw_dither"] = bw_dither_path

    return test_images


def run_practical_part(base: Path, qualities: list[int], stage_log: list[str]) -> None:
    practical_dir = base / "practical_results"
    intermediate_dir = practical_dir / "intermediate_images"
    compressed_dir = practical_dir / "compressed"
    restored_dir = practical_dir / "restored"
    plots_dir = practical_dir / "plots"
    for d in [intermediate_dir, compressed_dir, restored_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    test_images = build_test_images(base, intermediate_dir)
    stage_log.append(f"[ПРАКТИКА] Набор тестовых изображений: {len(test_images)} файлов")

    for image_name, image_path in test_images.items():
        stage_log.append(f"[ПРАКТИКА] Изображение: {image_name} ({image_path.name})")
        input_size = image_path.stat().st_size
        q_points = []
        size_points = []

        for q in qualities:
            comp_path = compressed_dir / f"{image_name}_q{q}.sjpg"
            rest_path = restored_dir / f"{image_name}_q{q}.png"

            compress_image(str(image_path), str(comp_path), quality=q)
            decompress_image(str(comp_path), str(rest_path))

            comp_size = comp_path.stat().st_size
            ratio = input_size / comp_size if comp_size else 0.0
            metrics = compute_metrics(image_path, rest_path)

            stage_log.append(
                f"[ПРАКТИКА] q={q}: in={input_size} B, out={comp_size} B, "
                f"ratio={ratio:.3f}, PSNR={metrics['psnr']:.3f}"
            )

            q_points.append(q)
            size_points.append(comp_size)

        plt.figure(figsize=(7, 4))
        plt.plot(q_points, size_points, marker="o")
        plt.title(f"Compressed size vs Quality: {image_name}")
        plt.xlabel("Quality")
        plt.ylabel("Compressed size (bytes)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = plots_dir / f"{image_name}_size_vs_quality.png"
        plt.savefig(plot_path, dpi=140)
        plt.close()
        stage_log.append(f"[ПРАКТИКА] График сохранен: {plot_path.name}")

    stage_log.append("[ПРАКТИКА] Готово. Папка результатов: practical_results/")


def main():
    base = Path(__file__).resolve().parent
    out_dir = base / "practical_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "run_log_ru.txt"
    stage_log: list[str] = []
    qualities = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    run_practical_part(base, qualities=qualities, stage_log=stage_log)

    run_log.write_text("\n".join(stage_log), encoding="utf-8")

    print("Done. Results folder:", out_dir)
    print("Log:", run_log)


if __name__ == "__main__":
    main()
