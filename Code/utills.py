from PIL import Image
import torch
import numpy as np


def letterbox_image_padded(image, size=(640, 640)):
    """ Resize image with unchanged aspect ratio using padding """
    image_copy = image.copy()
    iw, ih = image_copy.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image_copy = image_copy.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0, 0, 0))
    new_image.paste(image_copy, ((w - nw) // 2, (h - nh) // 2))
    return new_image

def image_to_numpy(img):
    new_image = np.asarray(img)[np.newaxis, :, :, :] / 255.
    return new_image

def preprocess(im):
        """ Prepares input image before inference """
        im = image_to_numpy(im)
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).float()
        return im
