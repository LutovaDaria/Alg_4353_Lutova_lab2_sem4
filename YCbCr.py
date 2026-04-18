import numpy as np

def rgb_to_ycbcr_array(rgb):
    # rgb: numpy array shape (H, W, 3)
    rgb = rgb.astype(np.float32)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    y  = 0.299*r + 0.587*g + 0.114*b
    cb = -0.168736*r - 0.331264*g + 0.5*b + 128
    cr = 0.5*r - 0.418688*g - 0.081312*b + 128
    ycbcr = np.stack([y, cb, cr], axis=-1)
    return np.clip(ycbcr, 0, 255).astype(np.uint8)

def ycbcr_to_rgb_array(ycbcr):
    # ycbcr: numpy array shape (H, W, 3)
    ycbcr = ycbcr.astype(np.float32)
    y  = ycbcr[..., 0]
    cb = ycbcr[..., 1]
    cr = ycbcr[..., 2]
    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    from PIL import Image

    img = Image.open("color.png").convert("RGB")
    arr = np.array(img)
    ycbcr_arr = rgb_to_ycbcr_array(arr)
    restored_arr = ycbcr_to_rgb_array(ycbcr_arr)
    Image.fromarray(restored_arr, "RGB").save("restored.png")
