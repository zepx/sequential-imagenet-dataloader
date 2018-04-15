# loads imagenet and writes it into one massive binary file

import cv2
import os
import numpy as np
from tensorpack.dataflow import *
from tensorpack.dataflow.image import *

if __name__ == '__main__':
    # images returned by cv2 are h, w,  c
    output_path = '/mnt/data/imagenet/ilsvrc12_{}_lmdb_224_pytorch'
    imagenet_path = '/mnt/data/imagenet/original'
    aug = imgaug.AugmentorList([imgaug.ResizeShortestEdge(256), imgaug.CenterCrop((224, 224))])

    for name in ['val']:
        output_path = output_path.format(name)
        ds0 = dataset.ILSVRC12(imagenet_path, name, shuffle=False)
        ds1 = AugmentImageComponent(ds0, aug)
        ds = MapDataComponent(ds1, lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB), 0)
        ds = PrefetchDataZMQ(ds, nr_proc=1)
        dftools.dump_dataflow_to_lmdb(ds, output_path)
