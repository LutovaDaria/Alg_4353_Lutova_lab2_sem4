import os
import struct

import numpy as np
from PIL import Image

MAGIC = b"RAW1"
TYPE_BW = 0
TYPE_GRAY = 1
TYPE_RGB = 2
CS_RGB = 0
CS_YCBCR = 1


def save_raw(image: Image.Image, filename: str, img_type: str, color_space: str = "RGB"):
    img_type_map = {"bw": TYPE_BW, "grayscale": TYPE_GRAY, "color": TYPE_RGB, "rgb": TYPE_RGB}
    cs_map = {"RGB": CS_RGB, "YCbCr": CS_YCBCR}
    if img_type not in img_type_map:
        raise ValueError("img_type must be one of: bw, grayscale, color/rgb")
    if color_space not in cs_map:
        raise ValueError("color_space must be RGB or YCbCr")

    width, height = image.size
    t = img_type_map[img_type]
    cs = cs_map[color_space]

    if t == TYPE_RGB:
        payload = image.convert("RGB").tobytes()
    elif t == TYPE_GRAY:
        payload = image.convert("L").tobytes()
    else:
        # 1 byte per pixel: 0 or 255 (task wording: one byte defines pixel value)
        arr = np.array(image.convert("1", dither=Image.Dither.NONE), dtype=np.uint8)
        payload = (arr * 255).astype(np.uint8).tobytes()

    header = MAGIC + struct.pack(">BBII", t, cs, width, height)
    with open(filename, "wb") as f:
        f.write(header)
        f.write(payload)


def load_raw(filename: str):
    with open(filename, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError("Not a RAW1 file")
        t, cs, width, height = struct.unpack(">BBII", f.read(10))
        payload = f.read()

    if t == TYPE_RGB:
        img = Image.frombytes("RGB", (width, height), payload)
        mode = "RGB"
    elif t == TYPE_GRAY:
        img = Image.frombytes("L", (width, height), payload)
        mode = "L"
    elif t == TYPE_BW:
        img = Image.frombytes("L", (width, height), payload).convert("1", dither=Image.Dither.NONE)
        mode = "1"
    else:
        raise ValueError("Unknown image type code")

    cs_name = "RGB" if cs == CS_RGB else "YCbCr" if cs == CS_YCBCR else "unknown"
    meta = {"type": t, "mode": mode, "width": width, "height": height, "color_space": cs_name}
    return img, meta


def calculate_compression_ratio(raw_file, compressed_file):
    raw_size = os.path.getsize(raw_file)
    compressed_size = os.path.getsize(compressed_file)
    return raw_size, compressed_size, (raw_size / compressed_size if compressed_size else float("inf"))


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    os.makedirs(os.path.join("outputs", "raw"), exist_ok=True)

    img = Image.open("color.png")
    gray_img = img.convert("L")
    gray_img.save(os.path.join("outputs", "raw", "gray_image.png"))

    bw_img = gray_img.point(lambda c: round(c / 255) * 255)
    bw_img.save(os.path.join("outputs", "raw", "bw_image.png"))

    bw_dither = gray_img.convert(mode="1")
    bw_dither.save(os.path.join("outputs", "raw", "bw_image_dither.png"))

    save_raw(img, os.path.join("outputs", "raw", "color.raw"), "color", "RGB")
    lenna_img = Image.open("lenna.png")
    save_raw(lenna_img, os.path.join("outputs", "raw", "lenna.raw"), "color", "RGB")
    save_raw(gray_img, os.path.join("outputs", "raw", "gray.raw"), "grayscale", "RGB")
    save_raw(bw_img, os.path.join("outputs", "raw", "bw.raw"), "bw", "RGB")
    save_raw(bw_dither, os.path.join("outputs", "raw", "bw_dither.raw"), "bw", "RGB")

    load_raw(os.path.join("outputs", "raw", "color.raw"))[0].save(os.path.join("outputs", "raw", "color_loaded.png"))
    load_raw(os.path.join("outputs", "raw", "gray.raw"))[0].save(os.path.join("outputs", "raw", "gray_loaded.png"))
    load_raw(os.path.join("outputs", "raw", "bw.raw"))[0].save(os.path.join("outputs", "raw", "bw_loaded.png"))
    load_raw(os.path.join("outputs", "raw", "bw_dither.raw"))[0].save(os.path.join("outputs", "raw", "bw_dither_loaded.png"))
    load_raw(os.path.join("outputs", "raw", "lenna.raw"))[0].save(os.path.join("outputs", "raw", "lenna_loaded.png"))

    files = [
        (os.path.join("outputs", "raw", "color.raw"), "color.png"),
        (os.path.join("outputs", "raw", "gray.raw"), os.path.join("outputs", "raw", "gray_image.png")),
        (os.path.join("outputs", "raw", "lenna.raw"), "lenna.png"),
    ]
    for raw_file, compressed_file in files:
        raw_size, compressed_size, ratio = calculate_compression_ratio(raw_file, compressed_file)
        print(f"Файл {raw_file}: {raw_size} байт")
        print(f"Файл {compressed_file}: {compressed_size} байт")
        print(f"Коэффициент сжатия: {ratio:.2f}")
        print()
