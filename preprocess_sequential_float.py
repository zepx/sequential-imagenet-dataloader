# loads imagenet and writes it into one massive binary file
# this file generates a lmdb with floating point values

import cv2
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

if __name__ == '__main__':
    class BinaryILSVRC12(dataset.ILSVRC12Files):
        def __init__(self, dir, name):
            super(BinaryILSVRC12, self).__init__(
                    dir, name)

            self.original = '/mnt/data/imagenet'
            self.valdir = os.path.join(self.original, 'ILSVRC2012_img_val_sorted')
            self.traindir = os.path.join(self.original, 'ILSVRC2012_img_train')
            
            self.val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(self.valdir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ])),
                batch_size=1, shuffle=False,
                num_workers=8, pin_memory=True)

        def get_data(self):
            for i, (input, target) in enumerate(self.val_loader):
                input = input.permute(0,2,3,1)
                # print(input.size())
                yield [input.numpy()[0], target.numpy()[0]]
    # imagenet_path = os.environ['IMAGENET']
    output_path = '/mnt/data/imagenet/ilsvrc12_{}_lmdb_224_pytorch'
    imagenet_path = '/mnt/data/imagenet/original'

    for name in ['train']:
        output_path = output_path.format(name)
        ds0 = BinaryILSVRC12(imagenet_path, name)
        ds = PrefetchDataZMQ(ds0, nr_proc=1)
        dftools.dump_dataflow_to_lmdb(ds, output_path)
