# loads imagenet and writes it into one massive binary file
# this file generates a lmdb with floating point values

# import cv2
from PIL import Image
import os
import numpy as np
from tensorpack.dataflow import *
from tensorpack.dataflow.image import *
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

def resize_image(x, size):
    w, h = img.size
    if w < h:
        ow = size
        oh = int(size * h / w)
        return img.resize((ow, oh))
    else:
        oh = size
        ow = int(size * w / h)
        return img.resize((ow, oh))

class ResizeShortestEdgePillow(imgaug.ResizeShortestEdge):
    def _get_augment_params(self, img):
        h, w = img.shape[:2]
        scale = self.size * 1.0 / min(h, w)
        if h < w:
            newh, neww = self.size, int(scale * w + 0.5)
        else:
            newh, neww = int(scale * h + 0.5), self.size
        # a = np.asarray(im.resize((neww, newh)))
        return ResizeTransformPillow(h, w, newh, neww, Image.BILINEAR)

class ResizeTransformPillow(imgaug.transform.ResizeTransform):
    def apply_image(self, img):
        # assert img.shape[:2] == (self.h, self.w)
        im = Image.fromarray(img)
        ret = np.asarray(im.resize((self.neww, self.newh), self.interp))
        return ret

if __name__ == '__main__':
    class BinaryILSVRC12(dataset.ILSVRC12Files):
        def get_data(self):
            for fname, label in super(BinaryILSVRC12, self).get_data():
                im = Image.open(fname)
                if im.mode is not 'RGB':
                    im = im.convert('RGB')
                yield [np.asarray(im), label]

    # imagenet_path = os.environ['IMAGENET']
    output_path = '/mnt/data/imagenet/ilsvrc12_{}_lmdb_224_pytorch'
    imagenet_path = '/mnt/data/imagenet/original'
    aug = imgaug.AugmentorList([ResizeShortestEdgePillow(256), imgaug.CenterCrop(224)])

    for name in ['train']:
        output_path = output_path.format(name)
        ds0 = BinaryILSVRC12(imagenet_path, name)
        # ds0 = dataset.ILSVRC12(imagenet_path, name)
        # ds0 = MapDataComponent(ds0, lambda x: print(x.shape), 0) 
        ds1 = AugmentImageComponent(ds0, aug)
        # ds = MapDataComponent(ds1, lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB), 0)
        ds = PrefetchDataZMQ(ds1, nr_proc=1)
        dftools.dump_dataflow_to_lmdb(ds, output_path)
