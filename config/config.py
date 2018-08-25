# coding=UTF-8

class CFG():
    voc_dir='/root/workspace/data/VOC2007_2012/VOCdevkit/VOC2007'

    feat_dir='./features/vgg16'
    device_id=1
    use_cuda=True

    caffe_model="vgg16_caffe_pretrain.pth"
    use_caffe=True
    
    loc_mean=[.0,.0,.0,.0]
    # loc_std=[.1,.1,.2,.2]
    loc_std=[1.,1.,1.,1.]

    img_shorter_len=600
    img_longer_len=1000

cfg=CFG()
