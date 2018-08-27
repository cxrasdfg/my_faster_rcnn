# coding=UTF-8
# author:theppsh

import numpy as np 
import torch

np.random.seed(1234567)
torch.manual_seed(1234567)
torch.cuda.manual_seed(1234567)

from tqdm import tqdm
import time
import gc
import os
from show_bbox import show_img,tick_show,draw_bbox as draw_box
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from chainercv.evaluations import eval_detection_voc as voc_eval
from config import cfg
from data import TrainDataset,TestDataset,TrainSetExt,TestSetExt,preprocess
from net import FasterRCNN as MyNet
import cv2
import re

def train():
    print("my name is van")
    # let the random counld be the same
    
    if cfg.train_use_offline_feat:
        data_set=TrainSetExt()
    else:
        data_set=TrainDataset()
    data_loader=DataLoader(data_set,batch_size=1,shuffle=True,drop_last=False)

    net=MyNet(data_set.classes)
    net.print_net_info()

    epoch,iteration,w_path=get_check_point()
    if w_path:
        model=torch.load(w_path)
        if cfg.train_use_offline_feat:
            net.load_state_dict(model)
        else:
            net.load_state_dict(model)
        print("Using the model from the last check point:%s"%(w_path),end=" ")
        epoch+=1

    net.train()
    is_cuda=cfg.use_cuda
    did=cfg.device_id
    if is_cuda:
        net.cuda(did)
    
    epoches=int(1e6)
   
    while epoch<epoches:
        
        # train the rpn
        print('******epoch %d*********' % (epoch))

        for i,(imgs,boxes,labels,scale,img_sizes) in tqdm(enumerate(data_loader)):
            if is_cuda:
                imgs=imgs.cuda(did)
                labels=labels.cuda(did)
                boxes=boxes.cuda(did)
                scale=scale.cuda(did).float()
                img_sizes=img_sizes.cuda(did).float()
            loss=net.train_once(imgs,boxes,labels,scale,img_sizes)
            tqdm.write('Epoch:%d, iter:%d, loss:%.5f'%(epoch,iteration,loss))

            iteration+=1

        if cfg.train_use_offline_feat:
            torch.save(net.state_dict(),'%sweights_%d_%d'%(cfg.weights_dir,epoch,iteration) )
        else:
            torch.save(net.state_dict(),'%sweights_%d_%d'%(cfg.weights_dir,epoch,iteration) )

        epoch+=1

def test_net():
    data_set=TestDataset()
    data_loader=DataLoader(data_set,batch_size=1,shuffle=True,drop_last=False)

    classes=data_set.classes
    net=MyNet(classes)
    _,_,last_time_model=get_check_point()
    # assign directly
    # last_time_model='./weights/weights_21_110242'

    if os.path.exists(last_time_model):
        model=torch.load(last_time_model)
        if cfg.test_use_offline_feat:
            net.load_state_dict(model)
        else:
            net.load_state_dict(model)
        print("Using the model from the last check point:`%s`"%(last_time_model))
    else:
        raise ValueError("no model existed...")
    net.eval()
    is_cuda=cfg.use_cuda
    did=cfg.device_id
    # img_src=cv2.imread("/root/workspace/data/VOC2007_2012/VOCdevkit/VOC2007/JPEGImages/000012.jpg")
    # img_src=cv2.imread('./example.jpg')
    img_src=cv2.imread('./dog.jpg') # BGR
    img=img_src[:,:,::-1] # RGB
    h,w,_=img.shape
    img=img.transpose(2,0,1) # [c,h,w]

    img=preprocess(img)
    img=img[None]
    img=torch.tensor(img)
    if is_cuda:
        net.cuda(did)
        img=img.cuda(did)
    boxes,labels,probs=net(img,torch.tensor([[w,h]]).type_as(img))[0]

    prob_mask=probs>cfg.out_thruth_thresh
    boxes=boxes[prob_mask ] 
    labels=labels[prob_mask ].long()
    probs=probs[prob_mask]
    draw_box(img_src,boxes,color='pred',
        text_list=[ 
            classes[_]+'[%.3f]'%(__)  for _,__ in zip(labels,probs)
            ]
        )
    show_img(img_src,-1)
    
def eval_net(net=None,num=cfg.eval_number):
    if cfg.test_use_offline_feat:
        data_set=TestSetExt()
    else:
        data_set=TestDataset()
    data_loader=DataLoader(data_set,batch_size=1,shuffle=False,drop_last=False)
    
    is_cuda=cfg.use_cuda
    did=cfg.device_id

    if net is None:
        classes=data_set.classes
        net=MyNet(classes)
        _,_,last_time_model=get_check_point()
        # assign directly
        # last_time_model='./weights/weights_21_110242'

        if os.path.exists(last_time_model):
            model=torch.load(last_time_model)
            if cfg.test_use_offline_feat:
                net.load_state_dict(model)
            else:
                net.load_state_dict(model)
            print("Using the model from the last check point:`%s`"%(last_time_model))
            
            if is_cuda:
                net.cuda(did)
        else:
            raise ValueError("no model existed...")

    net.eval()
   
    upper_bound=num

    gt_bboxes=[]
    gt_labels=[]
    gt_difficults=[]
    pred_bboxes=[]
    pred_classes=[]
    pred_scores=[]

    for i,(img,sr_im_size,cur_im_size,gt_box,label,diff) in tqdm(enumerate(data_loader)):
        if i> upper_bound:
            break

        sr_im_size=sr_im_size.float()
        cur_im_size=cur_im_size.float()
        if is_cuda:
            img=img.cuda(did)
            im_size=sr_im_size.cuda(did)
            cur_im_size=cur_im_size.cuda(did)
            
        pred_box,pred_class,pred_prob=net(img,im_size,
            offline_feat=True,cur_image_size=cur_im_size)[0]
        prob_mask=pred_prob>cfg.out_thruth_thresh
        pbox=pred_box[prob_mask ] 
        plabel=pred_class[prob_mask ].long()
        pprob=pred_prob[prob_mask]

        gt_bboxes += list(gt_box.numpy())
        gt_labels += list(label.numpy())
        gt_difficults += list(diff.numpy().astype('bool'))

        pred_bboxes+=[pbox.cpu().detach().numpy()]
        pred_classes+=[plabel.cpu().numpy()]
        pred_scores+=[pprob.cpu().detach().numpy()]

        # pred_bboxes+=[np.empty(0) ]
        # pred_classes+=[np.empty(0) ]
        # pred_scores+=[np.empty(0) ]

    res=voc_eval(pred_bboxes,pred_classes,pred_scores,
        gt_bboxes,gt_labels,gt_difficults,use_07_metric=True)
    print(res)

    # avoid potential error
    net.train()


def get_check_point():
    pat=re.compile("""weights_([\d]+)_([\d]+)""")
    base_dir=cfg.weights_dir
    w_files=os.listdir(base_dir)
    if len(w_files)==0:
        return 0,0,None
    w_files=sorted(w_files,key=lambda elm:int(pat.match(elm)[1]),reverse=True)

    w=w_files[0]
    res=pat.match(w)
    epoch=int(res[1])
    iteration=int(res[2])

    return epoch,iteration,base_dir+w

if __name__ == '__main__':
    # ttt()
    train()
    # test_net()
    # eval_net()