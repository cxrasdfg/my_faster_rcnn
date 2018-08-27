# coding=UTF-8

class CFG():
    voc_dir='/root/workspace/data/VOC2007_2012/VOCdevkit/VOC2007'

    train_feat_dir='./features/vgg16/train/'
    train_use_offline_feat=True

    test_feat_dir='./features/vgg16/test/'
    test_use_offline_feat=True

    device_id=0
    use_cuda=True

    weights_dir='./weights/'

    caffe_model="./models/vgg16_caffe_pretrain.pth"
    use_caffe=True
    use_caffe_classifier=True
    
    loc_mean=[.0,.0,.0,.0]
    # loc_std=[.1,.1,.2,.2]
    loc_std=[5.,5.,10.,10.]

    img_shorter_len=600
    img_longer_len=1000

    rand_seed=0
    epochs=20
    lr=1e-4
    weight_decay=0.0005
    use_adam=True
    rpn_sigma=3.5
    rcnn_sigma=1.

    out_thruth_thresh=.5

    # rpn path
    rpn_thresh_pos=.7
    rpn_thresh_neg=.3

    rcnn_pos_thresh=.5
    rcnn_neg_thresh_lo=.1
    rcnn_neg_thresh_hi=.5
    
    eval_number=10000
   
cfg=CFG()
