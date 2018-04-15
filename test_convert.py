import cv2
import numpy as np
from PIL import Image

image_path = '/mnt/data/imagenet/ILSVRC2012_img_val/ILSVRC2012_val_00000003.JPEG'

im = Image.open(image_path)
im2 = cv2.imread(image_path, cv2.IMREAD_COLOR)
# im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

im_np = np.asarray(im, dtype=np.uint8)
print('PIL: (w, h)', im.size)
print('PIL NP shape: (h, w, c)', im_np.shape)
print('cv2 shape: (h, w, c)', im2.shape)
# print('h'np.array_equal(im2, im_np))


def resize_pil(img, size):
    w, h = img.size
    if w < h:
        ow = size
        oh = int(size * h / w)
        return img.resize((ow, oh))

    else:
        oh = size
        ow = int(size * w / h)
        return img.resize((ow, oh))


def resize_cv2(img, size):
    h, w = img.shape[:2]
    scale = size * 1.0 / min(h, w)
    if h < w:
        newh, neww = size, int(scale * w + 0.5)
    else:
        newh, neww = int(scale * h + 0.5), size
    # return ResizeTransform(h, w, newh, neww, self.interp)
    return cv2.resize(img, (neww, newh))


im_rz = resize_pil(im, 256)
im_rz_np = np.asarray(im_rz, dtype=np.uint8)
print('PIL: (w, h)', im_rz.size)
print('PIL NP Shape: ', im_rz_np.shape)

im2_rz = resize_cv2(im2, 256)
im2_rz = cv2.cvtColor(im2_rz, cv2.COLOR_BGR2RGB)
print('CV2: (h, w, c)', im2_rz.shape)
print(np.array_equal(im2_rz, im_rz_np))
print(im2_rz == im_rz_np)
