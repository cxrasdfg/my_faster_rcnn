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
from config import cfg
from data import TrainDataset,TestDataset,TrainSetExt,preprocess
from net import FasterRCNN as MyNet
import cv2
import re

def train():
    print("my name is van")
    # let the random counld be the same
    
    if cfg.use_offline_feat:
        data_set=TrainSetExt()
    else:
        data_set=TrainDataset()
    data_loader=DataLoader(data_set,batch_size=1,shuffle=True,drop_last=False)

    net=MyNet(data_set.classes)
    net.print_net_info()

    epoch,iteration,w_path=get_check_point()
    if w_path:
        model=torch.load(w_path)
        if cfg.use_offline_feat:
            net.load_state_dict(model)
        else:
            net.load_state_dict(model)
        print("Using the model from the last check point:%s"%(w_path),end=" ")

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
        if cfg.use_offline_feat:
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
    if os.path.exists(last_time_model):
        model=torch.load(last_time_model)
        if cfg.use_offline_feat:
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
    img_src=img_src[:,:,::-1] # RGB
    h,w,_=img_src.shape
    img_src=img_src.transpose(2,0,1) # [c,h,w]

    img=preprocess(img_src)
    img=img[None]
    img=torch.tensor(img)
    if is_cuda:
        net.cuda(did)
        img=img.cuda(did)
    boxes,labels,probs=net(img,torch.tensor([w,h]).type_as(img))[0]

    prob_mask=probs>.5
    boxes=boxes[prob_mask ] 
    labels=labels[prob_mask ].long()
    probs=probs[prob_mask]
    draw_box(img_src,boxes,color='pred',
        text_list=[ 
            classes[_]+'[%.3f]'%(__)  for _,__ in zip(labels,probs)
            ]
        )
    show_img(img_src,-1)
    

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