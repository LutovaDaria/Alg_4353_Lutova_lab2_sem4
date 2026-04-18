import numpy as np
from PIL import Image


def downsample_2x(image_np):
    # image_np: numpy array (H, W, C) or (H, W)
    return image_np[::2, ::2]

def upsample_2x(image_np):
    return image_np.repeat(2, axis=0).repeat(2, axis=1)

def linear_interpolate(x1, y1, x2, y2, x):
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

def linear_spline(x_nodes, y_nodes, x):
    # x_nodes, y_nodes: lists; x: float
    for i in range(len(x_nodes)-1):
        if x_nodes[i] <= x <= x_nodes[i+1]:
            return linear_interpolate(x_nodes[i], y_nodes[i], x_nodes[i+1], y_nodes[i+1], x)
    raise ValueError("x out of bounds")

def bilinear_interpolate(x1, x2, y1, y2, z11, z12, z21, z22, x, y):
    return (
        z11 * (x2-x) * (y2-y) +
        z21 * (x-x1) * (y2-y) +
        z12 * (x2-x) * (y-y1) +
        z22 * (x-x1) * (y-y1)
    ) / ((x2-x1) * (y2-y1))

#2

def resize_bilinear(image, new_h, new_w):
    h, w = image.shape[:2]
    if len(image.shape) == 2:
        channels = 1
        image = image[..., None]
    else:
        channels = image.shape[2]

    out = np.zeros((new_h, new_w, channels), dtype=image.dtype)
    x_ratio = w / new_w
    y_ratio = h / new_h

    for i in range(new_h):
        for j in range(new_w):
            x = (j + 0.5) * x_ratio - 0.5
            y = (i + 0.5) * y_ratio - 0.5
            x0 = int(np.floor(x))
            x1 = min(x0 + 1, w - 1)
            y0 = int(np.floor(y))
            y1 = min(y0 + 1, h - 1)
            dx = x - x0
            dy = y - y0

            for c in range(channels):
                z11 = image[y0, x0, c]
                z21 = image[y0, x1, c]
                z12 = image[y1, x0, c]
                z22 = image[y1, x1, c]
                value = (
                    z11 * (1-dx) * (1-dy) +
                    z21 * dx * (1-dy) +
                    z12 * (1-dx) * dy +
                    z22 * dx * dy
                )
                out[i, j, c] = np.clip(value, 0, 255)
    if channels == 1:
        return out[:, :, 0]
    return out

if __name__ == "__main__":
    img = np.array(Image.open("color.png"))
    resized = resize_bilinear(img, 300, 300)
    Image.fromarray(resized).save("resized.png")