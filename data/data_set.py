# coding=UTF-8

from chainercv.datasets.voc.voc_bbox_dataset import VOCBboxDataset
from chainercv.datasets.voc import voc_utils
from chainercv.transforms import resize_bbox
from chainercv.transforms import random_flip
from chainercv.transforms import flip_bbox
from ..config import cfg
from torch.utils.data import Dataset  
from skimage.transform import resize
import numpy as np

__name_list=voc_utils.voc_bbox_label_names

def caffe_normalize(img):
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img=img*255.0
    img = (img - mean).astype(np.float32, copy=True)
    return img

def preprocess(img,min_size=cfg.img_shorter_len,
        max_size=cfg.img_longer_len,easy_mode=False,easy_h_w=416):

    c,h,w=img.shape
    if not easy_mode:
        scale1=1.0*min_size/min(h,w)
        scale2=1.0*max_size/max(h,w)
        scale=min(scale1,scale2)
        # scale=1.0*self._shorter_len/min(h,w)
        scale=(scale,scale)

    else:
        scale=(1.0*easy_h_w/w,1.0*easy_h_w/h)

    # NOTE check the value range
    img=img/255.0
    img=resize(img,(c,h*scale[1],w*scale[0]),mode='reflect')

    img=caffe_normalize(img)
    return img

class Transform(object):
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = random_flip(
            img, x_random=True, return_param=True)
        bbox = flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, [scale,scale]

class TrainDataset(Dataset):
    classes=__name_list
    def __init__(self):
        self.cfg=cfg
        self.sdb=VOCBboxDataset(cfg.voc_dir,'trainval')
        self.trans=Transform(min_size=cfg.img_shorter_len,max_size=cfg.img_longer_len )
    
    def __getitem__(self,idx):
        # NOTE: sdb returns the `yxyx`...
        ori_img, bbox, label, difficult = self.sdb.get_example(idx)

        img, bbox, label, scale = self.trans((ori_img, bbox, label))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), bbox.copy(), label.copy(), scale


    def __len__(self):
        return len(self.sdb)


class TestDataset(Dataset):
    classes=__name_list
    def __init__(self, opt, split='test', use_difficult=True):
        self.cfg = cfg
        self.sdb = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.sdb.get_example(idx)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.sdb)

