# coding=UTF-8
# author:theppsh

import torch
from torch.autograd import Variable
from torch.nn import Conv2d,BatchNorm2d,Softmax,ReLU,LeakyReLU,Linear
import numpy  as np 
from torchvision.transforms import transforms
from torch.utils.data import Dataset,DataLoader
import cv2
import torch.nn.functional as F
from torchvision.models import vgg16,densenet121 as Backbone

import torchvision
from collections import OrderedDict
import matplotlib.pyplot as plt
from roi_pool_cupy import RoIPooling2D
from nms.pth_nms import pth_nms as ext_nms 
from tqdm import tqdm
import time
import gc
import os
from show_bbox import show_img,tick_show,draw_bbox as draw_box
from torchvision import transforms
from config import cfg
from data import TrainDataset,TestDataset 

import re

DEBUG=False
SHOW_RPN_RES=False
CUR_IMG=None
np.random.seed(1234567)
torch.manual_seed(1234567)
torch.cuda.manual_seed(1234567)


def decom_vgg16():
    # the 30th layer of features is relu of conv5_3
    if cfg.use_caffe:
        model = vgg16(pretrained=False)
        tt=torch.load(cfg.caffe_model)
        model.load_state_dict(tt)
    else:
        model=vgg16(True)
    features = list(model.features)[:30]
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6]
    del classifier[5]
    del classifier[2]

    classifier = torch.nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return torch.nn.Sequential(*features), classifier

class ResNet(torchvision.models.ResNet):
    def __init__(self,block, layers):
        super(ResNet,self).__init__(block,layers)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(torchvision.models.resnet.model_zoo.load_url(torchvision.models.resnet.model_urls['resnet18']))
    return model


normalize =transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
])

def img_normalization(img):
    r"""Normalize the image
    Args:
        img (np.ndarray): [h,w,3], obeys the `bgr` format, pixel value is in [0,255]
    Return:
        img (np.ndarray): [3,h,w], obeys the `rgb` format and is normalized to [0,1]
    """
    img=img[:,:,::-1] # convert bgr to rgb
    img=img.astype('float32')
    img=img/255.0  # normalization...
    img=img.transpose(2,0,1)
    img=torch.tensor(img).float()
    img=normalize(img)
    return img

class VOCDataset(Dataset):  
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]    

    def __init__(self,voc_root,list_path,easy_mode=True):
        super(VOCDataset,self).__init__()
        self._root=voc_root
        self._list_path=list_path
        self._img_list=[]
        self._gt=[]
        self._shorter_len=600
        self._max_len=1000 # constraint it or the gpu memory will boom... 
        self._easy_mode=easy_mode
        self._easy_h_w=224
        # load the list
        with open(self._root+'/'+self._list_path,encoding='utf-8') as list_file:
            buffer=list_file.read()
            buffer=buffer.split('\n')
            for i in buffer:
                temp=i.split(' ')
                assert (len(temp)-1)% 5 ==0
                if temp[0] == '':
                    continue
                self._img_list.append(temp[0])
                del temp[0]
                temp=np.array([int(_) if str.isdigit(_) else float(_)   for _ in temp],dtype='float32')
                self._gt.append(temp.reshape([-1,5]))

        assert len(self._gt) == len(self._img_list)            
        # print(buffer)

    def __getitem__(self,idx):
        img_path=self._root+'/'+ self._img_list[idx] # abosolute path
        boxes=self._gt[idx]  # [N,5].(cls_id,xmin,ymin,xmax,ymax), float type.
        boxes=np.array(boxes) # copy to avoid the potential changes.. 
        img=cv2.imread(img_path)
        h,w,c=img.shape
        # boxes[:,1:]=boxes[:,1:]/np.array([w,h,w,h]) # normalization   
        # boxes[:,1:]/=np.array([w,h,w,h])

        if not self._easy_mode:
            scale1=1.0*self._shorter_len/min(h,w)
            scale2=1.0*self._max_len/max(h,w)
            scale=min(scale1,scale2)
            # scale=1.0*self._shorter_len/min(h,w)
            scale=(scale,scale)

        else:
            scale=(1.0*self._easy_h_w/w,1.0*self._easy_h_w/h)
        img=cv2.resize(img,(int(w*scale[0]),int(h*scale[1]) ),interpolation=cv2.INTER_LINEAR)
        boxes[:,1:]=resize_boxes(boxes[:,1:],scale)
        
        tqdm.write("Read image: id=%d" %(idx),end=",\t ")
        if SHOW_RPN_RES or DEBUG :
            global CUR_IMG
            CUR_IMG=img.copy()
            draw_box(CUR_IMG,boxes[:,1:],color='gt')

        # TODO: add some data augmentation: flip, disort, paste and random crop 

        # img=img[:,:,::-1] # convert bgr to rgb
        # img=img.astype('float32')
        # img-=127. # zero mean
        # img=img/128.0  # normalization...
        # img=img.transpose(2,0,1)
        img=img_normalization(img)
        
        assert self._easy_mode or scale[0]==scale[1] 
        return img,boxes[:,1:],(boxes[:,0]).astype('int'),np.array(scale)

        raise NotImplementedError('__getitem__() not completed...')

    def __len__(self):
        return len(self._img_list)

class DetectorBlock(torch.nn.Module):
    def __init__(self,input_dim=7*7*1024,classes=TrainDataset.classes,front_fc=None):
        super(DetectorBlock,self).__init__()
        if front_fc is not None:
            self.fc_3_4=front_fc
        else:
            self.fc_3_4=torch.nn.Sequential(OrderedDict([
                ('fc3',Linear(input_dim,4096,bias=True)),
                ('relu3',ReLU(inplace=True)),
                ('fc4',Linear(4096,4096,bias=True)),
                ('relu4',ReLU(inplace=True))
            ]))
            for param in self.fc_3_4.parameters():
                param.data.normal_(0,0.01)
        self.classfier=torch.nn.Sequential(OrderedDict([
            ('fc5_1',Linear(self.fc_3_4[-2].out_features,len(classes)+1)),# plus 1 for the background
            # ('relu5_1',ReLU(inplace=True)),
            ('softmax5_1',Softmax())
        ]))
        self.box_reg=torch.nn.Sequential(OrderedDict([
            ('sfc5_2',Linear(self.fc_3_4[-2].out_features,4*len(classes))) # class-wise
        ]))

        for name,param in self.classfier.named_parameters():
            torch.nn.init.normal_(param.data,std=0.01)
        
        for name,param in self.box_reg.named_parameters():
            torch.nn.init.normal_(param.data,std=0.001)

    def forward(self,x):
        b,c,h,w=x.shape
        x=x.view(b,-1)
        x=self.fc_3_4(x)
        x1=self.classfier(x)
        x2=self.box_reg(x)
        
        return x1,x2

class MyNet(torch.nn.Module):
    def __init__(self,classes):
        super(MyNet,self).__init__()
        self.roi_size=[7,7]

        net='vgg16'

        # self.extractor=Backbone(pretrained=True).features 
        if net=='resnet18':
            self.extractor=resnet18(pretrained=True) 
            for param in self.extractor.parameters():
                param.requires_grad=False
            rpn_input_features=256
        elif net=='resnet101':
            self.extractor=resnet101(pretrained=True)
            rpn_input_features=1024
        elif net=='vgg16':
            self.extractor,classifier= decom_vgg16()
            rpn_input_features=512
        else:
            raise ValueError()
        

        self.stride=16
        # self.roi_pooling=ROIPooling(self.roi_size,self.stride)
        # try this roi pooling...
        self.roi_pooling=RoIPooling2D(self.roi_size[0],self.roi_size[1],1.0/self.stride)
        # self.roi_pooling=RoIPool(self.roi_size[0],self.roi_size[1],1.0/self.stride)
        
        self.loc_anchors=get_locally_anchors()
        self.anchor_num=len(self.loc_anchors)
        self.detector=DetectorBlock(self.roi_size[0]*self.roi_size[1]*rpn_input_features,

        classes=classes,front_fc=classifier if net=='vgg16' else None)
        self.rpn=torch.nn.Sequential(OrderedDict([
            ('rpn_conv1',Conv2d(rpn_input_features,512,(3,3),1,padding=1)),
            # ('rpn_bn_1',BatchNorm2d(512)),
            ('rpn_relu1',ReLU(inplace=True)),
            ('rpn_conv2',Conv2d(512,(4+2)*self.anchor_num,(1,1),1,padding=0)), # 4+2 means 4 coordinates, 1 objectness and 1 not objectness, 9 is the number of anchors
        ]))
        
        self.rpn_loss=RPNMultiLoss()
        self.fast_rcnn_loss=FastRCnnLoss()
        
        # mean and std for parameterized boxes
        self.mean=torch.tensor(cfg.loc_mean)
        self.std=torch.tensor(cfg.loc_std)

        for name,param in self.rpn.named_parameters():
            if 'conv' in name:
                torch.nn.init.normal_(param.data,mean=0,std=0.01) # as the paper said, gausian distribution of mean=0, std=0.01

        self.get_optimizer()
    
    def get_optimizer(self,lr=1e-3,use_adam=False):
        """
        return optimizer, It could be overwriten if you want to specify 
        special optimizer
        """
        params=[]
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                # if 'detector.fc' in key:
                #     if 'bias' in key:
                #         params += [{'params': [value], 'lr': lr * 20, 'weight_decay': 0}]
                #     else:
                #         params += [{'params': [value], 'lr': lr *10, 'weight_decay': 0.9}]
                # else:
                #     if 'bias' in key:
                #         params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                #     else:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': 0.0005}]
        if use_adam:
            print("Using Adam optimizer")
            self.optimizer = torch.optim.Adam(params)
        else:
            print("Using SGD optimizer")
            self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def only_train_cls(self,imgs,box,label):
        label=label[0]
        box=box[0]
        x=self.extractor(imgs)
        _col=torch.zeros(len(box),1).type_as(box)
        rois=torch.cat([_col,box],dim=1)
        x=self.roi_pooling(x,rois)
        pos_cls,out_reg_box=self.detector(x)  

        label= label+1
        label=label.long()
        loss=-pos_cls[torch.arange(len(pos_cls)).long(),(label).long()].log().sum()
        # loss=torch.nn.CrossEntropyLoss()(pos_cls,label)

        # pos_cls=pos_cls.softmax(dim=1)
        ttt=pos_cls[torch.arange(len(pos_cls)).long(),(label).long()]
        _,acc=pos_cls[:,1:].max(dim=1)
        acc=acc+1
        acc=(acc==label).sum().float()/len(acc) 
        
        tqdm.write("Only train for classification: max prob=%.5f, acc=%.5f" \
            %(ttt.max(),acc) ,end=",\t ")

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

        
    def train_once(self,imgs,box,label,scale):
        r"""Train the rpn and fast r-cnn together
        Args:
            imgs (tensor): [1,3,h,w]
            box (tensor): [1,N,4]
            label (tensor): [1,N]
        """
        assert imgs.shape[0] == 1
        assert box.shape[0]==1

        # return self.only_train_cls(imgs,box,label)

        t1=time.time()
        img_size,img_feat,anchors,out_rois,\
            out_cls,sampled_rois \
            = self.first_stage(imgs,12000,2000,scale)
        
        # tqdm.write("max of img_feat:%.5f, sum of img_feat:%.5f"%(img_feat.max(),img_feat.sum()), end=",\t ")
        if SHOW_RPN_RES:
            draw_box(CUR_IMG,sampled_rois[:,1:][:6],'roi')
            show_img(CUR_IMG,1)
        # if DEBUG:
            # tick_show(CUR_IMG,sampled_rois[:,1:],'roi')
        
        # print('Time of first stage:%.3f' %(time.time()-t1))
        # since the batch_size=1
        anchors=anchors[0] # [N,4]
        box=box[0] # [N,4]
        label=label[0] # [N] 
        out_cls=out_cls[0]
        out_rois=out_rois[0]

        #################### RPN #####################
        # match the anchor to gt box
        t1=time.time()
        gt_loc,assign=self.anchor_target(anchors,box,img_size)
        # print('Time of anchor matching:%.3f' %(time.time()-t1)) 

        # rpn loss
        pos_mask=(assign==1)
        neg_mask=(assign==0)
        pos_cls=out_cls[pos_mask]
        neg_cls=out_cls[neg_mask]
        pos_out_box=out_rois[pos_mask]
        gt_loc=gt_loc[pos_mask]

        # prepare the loss parameters
        # for first 128 positive samples (for classification)
        pos_rand_idx=torch.randperm(len(pos_cls))
        pos_rand_idx=pos_rand_idx[:128] 
        # for the rest negative samples (for clssification)
        neg_rand_idx=torch.randperm(len(neg_cls))
        neg_rand_idx=neg_rand_idx[:256-len(pos_rand_idx)]
        
        tqdm.write("RPN matching: %d pos, %d neg"%(len(pos_rand_idx),len(neg_rand_idx) ),end=",\t ")
        # prepare parameters for loss function
        # localtion parameters
        out_pred_box=pos_out_box[pos_rand_idx]
        gt_box=gt_loc[pos_rand_idx]

        # class parameters
        out_pos_cls= pos_cls[pos_rand_idx]
        out_neg_cls= neg_cls[neg_rand_idx]
        
        loss=0
        loss=self.rpn_loss(out_pos_cls,
            out_neg_cls,
            out_pred_box,
            gt_box,
            len(anchors)//len(self.loc_anchors))
        

        ################# fast r-cnn #####################
        # fast r-cnn         
        t1=time.time()
        rois,gt_loc,assign= self.roi_target(sampled_rois[:,1:],box,label)

        # print("Time of roi matching:%.3f" %(time.time()-t1))
        # since batch_size=1, img_id is always 0
        _col=torch.zeros(len(rois),1).type_as(rois)
        rois=torch.cat([_col,rois],dim=1)

        pos_mask=(assign>0)
        neg_mask=(assign==0)
        pos_roi=rois[pos_mask]
        pos_label=assign[pos_mask]
        neg_roi=rois[neg_mask]
        neg_label=assign[neg_mask]
        pos_gt_loc=gt_loc[pos_mask]
        
        ###############
        # chaneg the incices, select 32 positive and 96 negative
        pos_rand_idx=torch.randperm(len(pos_roi))[:32]
        neg_rand_idx=torch.randperm(len(neg_roi))[:128-len(pos_rand_idx)]

        # prepare the box 
        pos_rois=pos_roi[pos_rand_idx]
        target_box=pos_gt_loc[pos_rand_idx]
        pos_rois_corresbonding_gt_label=pos_label[pos_rand_idx]
        neg_rois=neg_roi[neg_rand_idx]

        # number
        num_pos_roi=len(pos_rois)
        num_neg_roi=len(neg_rois)

        tqdm.write("fast r-cnn matching: %d pos, %d neg"%(num_pos_roi,num_neg_roi) ,end=",\t ")
        # get the roi pooling featres
        t1=time.time()
        
        # NOTE find the memory leak...
        x=self.roi_pooling(img_feat,torch.cat([pos_rois,neg_rois],dim=0) ) # [num_pos_roi+num_neg_roi,c,7,7]
        # print("Time of roi pooling:%.3f"%(time.time()-t1))
        # tqdm.write("max of roi pooling:%.5f, sum of roi pooling:%.5f"%(
        #     x.max(),x.sum()), end=",\t "
        #     )

        if DEBUG:
            tick_show(CUR_IMG,pos_rois[:,1:],'roi')

        # [num_pos_rois+num_neg_roi,1+obj_cls_num], [num_pos_roi+num_neg_roi,4]
        out_cls,out_reg_box=self.detector(x)  

        # the img_id is useless, so remove it
        pos_rois=pos_rois[:,1:]
       
        out_reg_box=out_reg_box[:num_pos_roi] # [n',4*cls_num]
        
        # NOTE: out box is cls-wise box, we need to select the box
        out_reg_box=out_reg_box.view(len(out_reg_box),-1,4) # [n',cls_num,4]

        # [n',4]
        out_reg_box=out_reg_box[torch.arange(num_pos_roi).long(),
            pos_rois_corresbonding_gt_label-1] # -1 since the box reg branch obly has cls_num classes

        out_pos_cls=out_cls[:num_pos_roi]
        out_neg_cls=out_cls[num_pos_roi:]
        rcnn_loss=self.fast_rcnn_loss(out_pos_cls,
            pos_rois_corresbonding_gt_label,
            out_neg_cls,out_reg_box,target_box)

        loss+=rcnn_loss
        tqdm.write("fast r-cnn loss:%.5f" %(rcnn_loss.item()) ,end=',\t ')
        ##############
        
        # gradient descent
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # delete
        # del img_feat, rois, out_cls,out_rois,out_reg_box

        return loss.item()
        

    def anchor_target(self,anchors,gt_box,img_size,thresh_pos=.7,thresh_neg=.3):
        r"""Assign the anchors to gt_box
        Args:
            anchors (tensor): [a,4]
            gt_box (tensor): [b,4]
        Return:
            gt_loc (a,4): offset and scale of anchors related to gt_box
            assign (tensor): [a,4] ::math:
                assign_i = \begin{cases}
                    1 & \text{if positive} \\
                    0 & \text{if negative} \\
                    -1 & \text{if non-pos and non-neg} \\ 
                \end{cases}
        """
        # remove the boxes cross the boundary
        a=len(anchors)
        inp_mask=(anchors[:,0]>=0) *\
            (anchors[:,1]>=0)*\
            (anchors[:,2]<=img_size[0]-1)*\
            (anchors[:,3]<=img_size[1]-1) # [a]
        
        anchors=anchors[inp_mask]

        # 1 denotes pos,0 denotes neg, -1 denotes `do not care`
        assign=torch.full([len(anchors)],-1).long() # [a']
        ious=t_box_iou(anchors,gt_box) # [a',b]
        max_ious,idx=ious.max(dim=1) # [a']
        
        # parameterized...
        gt_loc=encode_box(gt_box[idx],anchors) # [a',4]

        # assign neg
        assign[max_ious<thresh_neg]=0 

        # assign highest iou, it might cover the neg label
        _,idx=ious.max(dim=0) # [b]
        assign[idx] = 1

        # assign pos > threshold
        assign[max_ious>thresh_pos] = 1
        
        # unmap to the size of a 
        # unmap assign
        new_assign=torch.full([a],-1).long()
        new_assign[inp_mask]=assign
        assign = new_assign
        
        # unmap gt_loc
        new_gt_loc=torch.full([a,4],0).type_as(gt_box)
        new_gt_loc[inp_mask]=gt_loc
        gt_loc=new_gt_loc

        if gt_box.is_cuda:
            gt_loc=gt_loc.cuda()

        return gt_loc,assign

    def roi_target(self,rois,gt_box,label,pos_thresh=.5,
                    neg_thresh_lo=.0,neg_thresh_hi=.5):
        r"""Assign roi to gt_box
        Args:
            rois (tensor): [a,4]
            gt_box (tensor): [b,4] 
        Return:
            rois (tensor): [a+b,4]
            gt_loc (tensor):[a+b,4]
            assign (tensor): [a+b]
        """
        assert rois.shape[1]==4, "please remove the img_id"
        rois=torch.cat([rois,gt_box],dim=0) # [a+b,4]
        
        ious=t_box_iou(rois,gt_box) # [a+b,b]
        max_ious,idx=ious.max(dim=1)
        
        # parameterizd box
        gt_loc=encode_box(gt_box[idx],rois)

        # assign the neg:
        assign=torch.full([len(rois)],-1).long().type_as(label)

        neg_mask=(max_ious>neg_thresh_lo)*(max_ious<neg_thresh_hi)
        # if neg_mask.sum() == 0:
            # tqdm.write("Warning: neg_roi for fast r-cnn is zero",end=" ")
            # neg_mask=(max_ious<neg_thresh_hi)
            # raise ValueError("there is no negative roi for fast r-cnn")
        assign[neg_mask]=0
        
        # assign the pos:
        pos_mask=max_ious>pos_thresh

        # plus one since 0 denotes the neg, we must begin from the 1
        assign[pos_mask]=label[idx][pos_mask].long()+1 

        # normalize?
        mean=self.mean # [4]
        std=self.std # [4]

        mean=mean[None].expand_as(gt_loc).type_as(gt_loc)
        std=std[None].expand_as(gt_loc).type_as(gt_loc)

        gt_loc-=mean
        gt_loc=gt_loc/std

        return rois,gt_loc,assign


    def first_stage(self,x,num_prop_before,num_prop_after,scale,min_size=16):
        r"""The first part of the network, including the feature extraction,
        and rpn forwarding
        Args:
            x (tensor): [b,3,H,W], batch of images
            num_prop_before (int): remained rois for each image before sorting
            num_prop_after (int): remainted rois for each image after sorting
            scale (tensor[float]): [b,2] stores the scale for width and height...  
            min_size(int): threshold for discarding the boxes...
        Return:
            img_size (shape): image width and height
            img_features (tensor): [b,c,h,w]
            anchors (tensor): [b,N,4] 
            out_rois (tensor): [b,N,4], the output of rpn, parameterized boxes
            out_cls (tensor): [b,N,2], the output of rpn, softmax class
            sampled_rois (tensor): [n',5], sampled_rois for fast-rcnn, fmt is 
        (img_id,left,top,right,bottom)self.fc_3_4[-2].out_features
        """
        img_size=x.shape[2:][::-1]
        t1=time.time()
        img_features=self.extractor(x) # [b,c,h,w]
        # print("Time of feature extraction:%.3f" %(time.time()-t1))

        t1=time.time()
        rois=self.rpn(img_features) # [b,anchor_num*(4+2),w,h]
        # print("Time of rpn:%.3f" %(time.time()-t1))

        b,c,h,w=img_features.shape
        
        t1=time.time()
        out_rois,out_cls=self.convert_rpn_feature(rois) # [b,N,4],[b,N,2]
        # print("Time of convert rpn features:%.3f" %(time.time()-t1))

        # anchors in the feature map 
        # anchors=self.get_anchors(img_features,img_size) # [N,4], ps: do not the train the rpn, en? 
        anchors=get_anchors(self.loc_anchors,h,w,self.stride,is_cuda=x.is_cuda)
        anchors=anchors[None].expand(b,-1,-1).contiguous() # [b,N,4]
        # anchors=torch.tensor(anchors).float() # to torch.tensor

        # rois [b,N,4], `xyxy`
        rois= decode_box(out_rois,anchors) # decode to the normalized box

        # clip the boundary
        # rois[:,:,:2]=np.maximum(rois[:,:,:2],.0) # `left`, `top` should be greater than zero`
        # rois[:,:,2:]=np.minimum(rois[:,:,2:],1.0) # `right`, `bottom` should be samller than 1 
        # rois[:,:,:2][rois[:,:,:2]<0]=0
        # rois[:,:,2:][rois[:,:,2:]>1.0]=1.0
        # rois[:,:,:2]=torch.clamp(rois[:,:,:2].clone(),min=.0,max=1.)
        # rois[:,:,2:]=torch.clamp(rois[:,:,2:].clone(),min=.0,max=1.)
        rois[:,:,[0,2]]=rois[:,:,[0,2]].clone().clamp(min=.0,max=img_size[0]-1)
        rois[:,:,[1,3]]=rois[:,:,[1,3]].clone().clamp(min=.0,max=img_size[1]-1)
        
        # remove the rois whose size < threshold...
        h=rois[:,:,2]-rois[:,:,0] # [b,N]
        w=rois[:,:,3]-rois[:,:,1] # [b,N]
        min_size=scale*min_size # [b,2]
        
        # if len(list(rois_masked.shape))==2:
        #     rois_masked=rois_masked [None]
        #     out_cls_masked=out_cls_masked[None]

        # nms for each image
        rois_with_id=torch.empty(0).type_as(rois)
        
        t1=time.time()
        for i in range(b):
            scale_mask=(h[i]>min_size[i,1])*(w[i]>min_size[i,0]) # [N] 
            rois_masked=rois[i][scale_mask] # [N',4]
            out_cls_masked=out_cls[i][scale_mask] # [N',2]

            # sort by the pos
            rois_i=rois_masked
            tt=out_cls_masked[:,0] # [N',1] idx 0 is the pos
            _,idx=tt.sort(descending=True)    
            idx=idx[:num_prop_before]
            tt=_[:num_prop_before][:,None]
            rois_i=rois_i[idx]
            
            temp=self.nms(torch.cat([rois_i,tt],dim=1),img_id=i) # [M,1+4+1]
            
            # select top-N
            # here 5 is the pos label
            _,idx=temp[:,5].sort(descending=True) 
            idx=idx[:num_prop_after]
            temp=temp[idx]
            rois_with_id=torch.cat([rois_with_id,temp], dim=0) # [M'+M,1+4+2]

        # print("Time of nms in first stage:%.3f" %(time.time()-t1))

        sampled_rois=rois_with_id[:,:1+4] # [n',1+4]
        sampled_cls=rois_with_id[:,1+4:] # [n',cls_num]

        # select top-N
        # _,idx=labels[:,0].sort(descending=True)
        # idx=idx[:num_prop]
        # rois=rois[idx]
        tqdm.write("Number of sampled rois:%d" %(len(sampled_rois)),end=",\t ")

        return img_size,img_features,anchors,out_rois,out_cls,sampled_rois  

    def convert_fast_rcnn(self,classes,boxes,rois,ratios):
        r"""Convert the output of the fast r-cnn to the real box with nms
        Args:
            classes (tensor): [N,obj_cls_num+1]
            boxes (tensor): [N,4*cls_num], parameters of bounding box regression 
            rois (tensor): [N,1+4], the coordinate obeys format `xyxy`
            ratios (tensor[float]): [b,2], stores the scale ratio to the original image
        Return:
           res (list[object]): [img_num] 
        """
        res=[]
        for i in range(len(rois)):
            # for the i=th image
            # find the i-th im
            mask=(rois[:,0]==i)
            
            i_rois=rois[mask] # [M,1+4]
            i_param_boxes=boxes[mask] # [M,4*cls_num]
            i_param_cls=classes[mask] # [M,1+obj_cls_num]
            if len(i_rois)==0:
                # no other images
                break 
            ratio=ratios[i]
            # the image_id is useless, so remove it
            i_rois=i_rois[:,1:] # [M,4]

            i_param_boxes=i_param_boxes.view(len(i_param_boxes),-1,4) # [M,cls_num,4] 

            mean=self.mean # [4]
            std=self.std # [4]

            mean=mean[None].expand_as(i_param_boxes).type_as(i_param_boxes)
            std=std[None].expand_as(i_param_boxes).type_as(i_param_boxes)

            i_param_boxes=(i_param_boxes*std+mean)

            i_rois=i_rois[:,None].expand_as(i_param_boxes) # [M,cls_num,4]
            r_boxes= decode_box(i_param_boxes,i_rois) # [M,cls_num,4]
            _,cls_num,_=r_boxes.shape
            
            # remove the neg_cls_score and apply nms
            res_box,res_label,res_prob=self._suppres(r_boxes,i_param_cls[:,1:],cls_num)
            res_box*=ratio
            res.append((res_box,res_label,res_prob))

        return res

        raise NotImplementedError()

    def _suppres(self,boxes,prob,cls_num):
        r"""Suppresion for validation of the fast r-cnn
        Args:
            boxes (tensor): [M,cls_num,4]
            prob (tensor): [M,cls_num]
            cls_num: number of classes
        Return:
            res_boxes (tensor): [M',4]
            res_labels (tensor): [M']
            res_prob (tensor): [M']  
        """
        assert cls_num == prob.shape[1]
        
        res_boxes=torch.empty(0).type_as(boxes)
        res_labels=res_boxes.clone()
        res_prob=res_boxes.clone()
        for cls in range(cls_num):
            if cls == 11:
                print(cls)
            box_cls=boxes[:,cls,:] # [M,4]
            prob_cls=prob[:,cls] # [M]
            box_cls=self.nms(torch.cat([box_cls,prob_cls[:,None]],dim=1),.5) # [m',5]
            box_cls,prob_cls=box_cls[:,:4],box_cls[:,4] # [m',4], [m']
            res_boxes=torch.cat([res_boxes,box_cls],dim=0)
            res_labels= torch.cat([res_labels,
                torch.full([len(box_cls)],cls).type_as(boxes)],
                dim=0
                )
            res_prob=torch.cat([res_prob,prob_cls],
                dim=0
                )

        return res_boxes,res_labels,res_prob

    def forward(self,x,src_size):
        current_size=x.shape[2:][::-1]
        current_size=torch.tensor(current_size)[None].expand(x.shape[0],-1) # [b,2]
        current_size=current_size.type_as(x).float()
        ratios=src_size/current_size.float() # [b,2]
        ratios=ratios[:,None].expand(-1,2,-1).contiguous().view(-1,4) # [b,4]

        img_size,img_features,\
            anchors,out_rois,\
            out_cls,sampled_rois=self.first_stage(x,6000,300,scale=1./ratios)        

        # roi pooling...
        x=self.roi_pooling(img_features,sampled_rois)
        classes,boxes=self.detector(x)
       
        # convert and nms
        res=self.convert_fast_rcnn(classes,boxes,sampled_rois,ratios)

        return res

    def nms(self,rois,thresh=.7,img_id=None,_use_ext=True):
        """
        nms to remove the duplication
        input:
            rois (tensor): [N,4+cls_num], attention that the location format is `xyxy`
            thresh (float): the threshold for nms
            img_id (int): the image id will add to the rois
            _use_ext (bool): use the cuda extension of nms
        return:
            rois (tensor): [M,1+4+cls_num] if img_id is not None else [M,4+cls_num] 
        """
        cls_num=rois.shape[1]-4
        
        for i in range(cls_num):
            if _use_ext:
                # it dose not work...., since the state is null pointer
                dets=rois[:,[0,1,2,3,i+4]]
                order=ext_nms(dets,thresh=thresh)
                mask=torch.full([len(dets)],1,dtype=torch.uint8)
                mask[order]=0
                del order
                rois[:,i+4][mask]=0
            else:
                _,indices=rois[:,i+4].sort(descending=True)
                _use_deprecated=False
                # deprecated...
                if _use_deprecated:
                    for k in range(len(indices)):
                        idx_k=indices[k]
                        box_k=rois[idx_k:idx_k+1,:4]
                        if rois[idx_k,i+4] <1e-6:
                            continue
                        for j in range(k+1,len(indices)):
                            idx_j=indices[j]
                            box_j=rois[idx_j:idx_j+1,:4]
                            box_iou=MyNet.intersection_overlap_union(box_j,box_k)
                            if box_iou[0]>=thresh:
                                rois[idx_j,i+4]=0
                #####################################################
                # new... it might be faster...
                else:
                    while len(indices)>1:
                        idx_k=indices[0]
                        idx_rest=indices[1:]
                        box_k=rois[[idx_k],:4]
                        box_rest=rois[idx_rest,:4]
                        iou=t_box_iou(box_k,box_rest)
                        iou=iou[0] 
                        keep_mask=iou<thresh
                        # rois[indices,i+4][1:][1-keep_mask][:]=0
                        rois[indices[1:][1-keep_mask],i+4]=0
                        indices=indices[1:][keep_mask]
        
        sorted_rois,_=rois[:,4:].max(dim=1)
        rois=rois[sorted_rois>1e-6] # [M,4+cls_num] 

        # append the id to the rois if valid
        if img_id is not None: 
           extra_dim=torch.full([len(rois),1],img_id).type_as(rois)
           rois=torch.cat([extra_dim,rois],dim=1) # [M,1+4+cls_num] 
        return rois 

        raise NotImplementedError()

    def convert_rpn_feature(self,x):
        """
        convert thei rpn output feature to the corresponding parameterized rois and labels
        input:
            x (tensor): [b,anchor_num*(4+2),h,w]
        output:
            param_rois (tensor): [b,anchor_num*h*w,4]
            labels (tensor): [b,anchor_num*h*w,2]
        """
        b,fc,fh,fw=x.shape
        x=x.view(b,self.anchor_num,(4+2),fh,fw) # [b,anchor_num,(4+2),fh,fw]
        x=x.permute(0,1,3,4,2).contiguous() # [b,anchor_num,fh,fw,(4+2)]
        x=x.view(b,-1,6)
        param_rois=x[:,:,:4] # [b,N,4]
        labels=x[:,:,4:] # [b,N,2]
        labels=F.softmax(labels,dim=2) # softmax...
        return param_rois, labels
        raise NotImplementedError('you must implement the feature to roi')
    

def _smooth_l1_loss(x,gt,sigma):
    sigma2 = sigma ** 2
    diff = (x - gt)
    abs_diff = diff.abs()
    flag = (abs_diff < (1. / sigma2))
    y=torch.where(flag, (sigma2 / 2.) * (diff ** 2), (abs_diff-.5/sigma2))
    return y.sum()
    
class RPNMultiLoss(torch.nn.Module):
    def __init__(self):
        super(RPNMultiLoss,self).__init__()
    
    def forward(self,pos_cls,neg_cls,out_box,gt_box,n_reg,_lambda=10):
        assert len(pos_cls) == len(out_box) # must be the same
        n_cls=len(pos_cls)+len(neg_cls)
        
        # class loss
        cls_loss=-pos_cls[:,0].log().sum()-neg_cls[:,1].log().sum()

        # smooth l1...
        # loss=cls_loss/n_cls+reg_loss*_lambda/n_reg
        reg_loss=_smooth_l1_loss(out_box,gt_box,3.)
        loss=cls_loss/n_cls+reg_loss/n_cls

        tqdm.write("rpn loss=%.5f: reg=%.5f, cls=%.5f" %(loss.item(),reg_loss.item(),cls_loss.item()),end=",\t ")
        return loss
        raise NotImplementedError()


class FastRCnnLoss(torch.nn.Module):
    def __init__(self):
        super(FastRCnnLoss,self).__init__()
    
    def forward(self,pos_cls,pos_label,neg_cls,out_box,gt_box,_lambda=1):
        # let slot 0 denote the negative class, then you should plus one on the label:
        # cls_loss=-pos_cls[[_ for _ in range(len(pos_cls))],(pos_label+1).long()].log().sum()-neg_cls[:,0].log().sum()

        # pos_label is already plus by one 
        assert pos_label.min() >0
        num_pos=len(pos_cls)
        num_neg=len(neg_cls)
        n_cls=num_pos+num_neg

        cls_loss=-pos_cls[torch.arange(num_pos).long(),(pos_label).long()].log().sum()\
            -(neg_cls[:,0].log().sum() if len(neg_cls)!=0 else 0)
        
        ttt=pos_cls[torch.arange(num_pos ).long(),(pos_label).long()].max()
        _,acc=pos_cls[:,:].max(dim=1)
        acc=acc
        acc=acc.long()
        acc=(acc==pos_label).sum().float()/num_pos
        tqdm.write("fast r-cnn: max prob=%.5f, acc=%.5f" \
            %(ttt,acc),end=",\t ")

        # smooth l1...
        reg_loss=_smooth_l1_loss(out_box,gt_box,1.)

        loss=cls_loss/n_cls+reg_loss/n_cls
        # loss=cls_loss

        return loss
        raise NotImplementedError()

def main():
    print("my name is van")
    # let the random counld be the same
    
    data_set=TrainDataset()
    #data_set=VOCDataset('/root/workspace/data/VOC2007_2012','train.txt',easy_mode=False)
    data_loader=DataLoader(data_set,batch_size=1,shuffle=True,drop_last=False)
   
    # data_loader2=DataLoader(VOCDataset('/root/workspace/data/VOC2007_2012','train.txt',easy_mode=True),batch_size=1,shuffle=True,drop_last=False)

    net=MyNet(data_set)
    # last_time_model='./model.pkl'
    epoch,iteration,w_path=get_check_point()

    if w_path:
        model=torch.load(w_path)
        net.load_state_dict(model)
        print("Using the model from the last check point:%s"%(w_path),end=" ")

    net.train()
    is_cuda=True
    if is_cuda:
        net.cuda()
    
    epoches=int(1e6)
    # rpn_loss=RPNMultiLoss()
    # params=[]
    # for param in net.parameters():
        # if param.requires_grad:
            # params.append(param)
    # rpn_opt=torch.optim.SGD(iter(params) ,lr=0.001,momentum=0.9,weight_decay=0.0005)
    # fast_rcnn_loss=FastRCnnLoss()
    while epoch<epoches:
        
        # train the rpn
        print('******epoch %d*********' % (epoch))
        # for i,(imgs,boxes,labels) in tqdm(enumerate(data_loader)):
        #     if next(net.parameters()).is_cuda:
        #         imgs=imgs.cuda()
        #         boxes=boxes.cuda()
        #     loss=net.train_rpn(imgs,boxes,rpn_loss,rpn_opt)
        #     print('rpn loss:%f'%(loss.data) )

        #     loss=net.train_fast_rcnn(imgs,boxes,labels,fast_rcnn_loss,rpn_opt)
        #     print('fast r-cnn loss:%f' %(loss.data))
        #     if epoch>6e4:
        #         for g in rpn_opt.param_groups:
        #             g['lr'] = 0.0001
        # for i,(imgs,boxes,labels) in tqdm(enumerate(data_loader2)):
        #     if next(net.parameters()).is_cuda:
        #         imgs=imgs.cuda()
        #         boxes=boxes.cuda()
        #     loss=net.train_fast_rcnn(imgs,boxes,labels,fast_rcnn_loss,rpn_opt)
        #     print('fast r-cnn loss:%f' %(loss.data))

        for i,(imgs,boxes,labels,scale) in tqdm(enumerate(data_loader)):
            if is_cuda:
                imgs=imgs.cuda()
                labels=labels.cuda()
                boxes=boxes.cuda()
                scale=scale.cuda().float()
            loss=net.train_once(imgs,boxes,labels,scale)
            tqdm.write('Epoch:%d, iter:%d, loss:%.5f'%(epoch,iteration,loss))

            iteration+=1

        torch.save(net.state_dict(),'./models/weights_%d_%d'%(epoch,iteration) )
        epoch+=1

def test_net():
    data_set=VOCDataset('/root/workspace/data/VOC2007_2012','train.txt',easy_mode=False)
    data_loader=DataLoader(data_set,batch_size=1,shuffle=True,drop_last=False)
   

    net=MyNet(data_set)
    _,_,last_time_model=get_check_point()
    if os.path.exists(last_time_model):
        model=torch.load(last_time_model)
        net.load_state_dict(model)
        print("Using the model from the last check point:`%s`"%(last_time_model))
    else:
        raise ValueError("no model existed...")
    net.eval()
    is_cuda=True
    # img_src=cv2.imread("/root/workspace/data/VOC2007_2012/VOCdevkit/VOC2007/JPEGImages/000012.jpg")
    # img_src=cv2.imread('./example.jpg')
    img_src=cv2.imread('./dog.jpg')
    h,w,_=img_src.shape
    scale1=1.0*600/min(h,w)
    scale2=1.0*1000/max(h,w)
    scale=min(scale1,scale2)
    scale=(scale,scale)
    img=cv2.resize(img_src,(int(w*scale[0]),int(h*scale[1]) ),interpolation=cv2.INTER_LINEAR)
    img=img_normalization(img)
    img=img[None]
    if is_cuda:
        net.cuda()
        img=img.cuda()
    boxes,labels,probs=net(img,torch.tensor([w,h]).type_as(img))[0]

    classes=data_set.classes
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
    

def encode_box(real_boxes,anchor_boxes):
    """Encode the real_box to the corresponding parameterized coordinates
    Args:
        real_boxes (tensor):[N,4], whose format is `xyxy`
        anchor_boxes (tensor):[N,4], i-th anchor is responsible for the i-th real box,
    and it's format is `xyxy`
    Return:
        parameterized boxes (tensor): [N,4]
    """
    assert real_boxes.shape==anchor_boxes.shape,'`real_boxes.shape` must be the same sa the `anchor_boxes`'
    if real_boxes.is_cuda and not anchor_boxes.is_cuda:
        anchor_boxes=anchor_boxes.cuda()
    assert anchor_boxes.is_cuda == anchor_boxes.is_cuda
    
    # change the boxes to `ccwh`
    real_boxes=xyxy2ccwh(real_boxes,inplace=False)
    anchor_boxes=xyxy2ccwh(anchor_boxes,inplace=False)

    encoded_xy=(real_boxes[:,:2]-anchor_boxes[:,:2])/anchor_boxes[:,2:]  # [N,2]
    encoded_wh=torch.log(real_boxes[:,2:]/anchor_boxes[:,2:])  # [N,2]

    return torch.cat([encoded_xy,encoded_wh],dim=1) # [N,4]

def decode_box(param_boxes,anchor_boxes):
    """Translate the parameterized box to the real boxes, real boxes
    are not the ground truths, just refer to boxes with format `xyxy` 
    Args:
        param_boxes (tensor) : [b,N,4], contain parameterized coordinates
        anchor_boxes (tensor) : [b,N,4], fmt is `xyxy`
    Return:
        boxes (tensor) : [b,N,4], whose format is `xyxy`
    """
    assert param_boxes.shape == anchor_boxes.shape 
    if param_boxes.is_cuda and not anchor_boxes.is_cuda:
        anchor_boxes=anchor_boxes.cuda()
    b,n,_=param_boxes.shape
    # change anchors to `ccwh`
    anchor_boxes=xyxy2ccwh(anchor_boxes.contiguous().view(-1,4),inplace=False).view(b,n,4)

    decoded_xy=param_boxes[:,:,:2]*anchor_boxes[:,:,2:]+anchor_boxes[:,:,:2] # [b,N,2]
    decoded_wh=torch.exp(param_boxes[:,:,2:])*anchor_boxes[:,:,2:] # [b,N,2]

    decode_box=torch.cat([decoded_xy,decoded_wh],dim=2)
    # change to `xyxy`
    decode_box=ccwh2xyxy(decode_box.view(-1,4),inplace=True).view(b,n,4)

    return  decode_box # [b,N,4]

def ccwh2xyxy(boxes,inplace=False):
    r"""Change the format of boxes from `ccwh` to `xyxy`
    Args:
        boxes (tensor): [n,4]
        inplace (bool): will return a new object if not enabled
    Return:
        after_boxes (tensor): [n,4], the transformed boxes
    """
    if inplace:
        after_boxes=boxes
    else:
        after_boxes=boxes.clone()
    
    if isinstance(after_boxes,Variable) and inplace:
        after_boxes[:,:2]=after_boxes[:,:2].clone()-after_boxes[:,2:]/2
        after_boxes[:,2:]=after_boxes[:,2:].clone()+after_boxes[:,:2]
    else:
        after_boxes[:,:2]-=after_boxes[:,2:]/2
        after_boxes[:,2:]+=after_boxes[:,:2]
    
    return after_boxes

def xyxy2ccwh(boxes,inplace=False):
    r"""Change the format of boxes from `xyxy` to `ccwh`
    Args:
        boxes (tensor): [n,4]
        inplace (bool): will return a new object if not enabled
    Return:
        after_boxes (tensor): [n,4], the transformed boxes
    """
    if inplace:
        after_boxes=boxes
    else:
        after_boxes=boxes.clone()
    if isinstance(after_boxes,Variable) and inplace:
        # use clone, or it will raise the inplace error
        after_boxes[:,2:]=after_boxes[:,2:].clone()-after_boxes[:,:2]
        after_boxes[:,:2]=after_boxes[:,:2].clone()+after_boxes[:,2:]/2
    else:
        after_boxes[:,2:]-=after_boxes[:,:2]
        after_boxes[:,:2]+=after_boxes[:,2:]/2
    return after_boxes

def t_meshgrid_2d(x_axis,y_axis):
    r"""Return 2d coordinates matrices of the two arrays
    Args:
        x_axis (tensor): [a]
        y_axis (tensor): [b]
    Return:
        x_axist (tensor): [b,a]
        y_axist (tensor): [b,a]
    """
    a,b=len(x_axis),len(y_axis)
    x_axis=x_axis[None].expand(b,a).clone()
    y_axis=y_axis[:,None].expand(b,a).clone()

    return x_axis,y_axis
    
def get_anchors(loc_anchors,h,w,stride=16,is_cuda=False):
    r"""Get the anchors with the size of the feature map
    Args:
        loc_anchors (tensor): [n,4]
        h (int): height
        w (int): width
    Return:
        anchors (tensor): [n*h*w,4]
    """
    n=len(loc_anchors)
    x_axis=torch.linspace(0,w-1,w)*stride
    y_axis=torch.linspace(0,h-1,h)*stride

    x_axis,y_axis=t_meshgrid_2d(x_axis,y_axis) # [h,w]

    x_axis=x_axis[None,None].expand(n,2,h,w).contiguous() # [n,2,h,w]
    y_axis=y_axis[None,None].expand(n,2,h,w).contiguous() # [n,2,h,w]

    # NOTE: contiguous is necessary since there are inplace operations below
    anchors=loc_anchors[:,:,None,None].expand(-1,-1,h,w).contiguous() # [n,4,h,w]
    
    # local coordinate to world coordinate
    # NOTE: inplace operations
    anchors[:,[0,2],:,:]+=x_axis
    anchors[:,[1,3],:,:]+=y_axis

    # transpose
    # NOTE: contiguous is necessary
    anchors=anchors.permute(0,2,3,1).contiguous() # [n,h,w,4]
    
    # reshape
    anchors=anchors.view(-1,4) # [n*h*w,4]
    if is_cuda:
        anchors=anchors.cuda()

    return anchors
    

def get_locally_anchors(stride=16,scales=[8,16,32],ars=[.5,1,2]):
    r"""Get the anchors in a locally window's coordinate
    Args:
        stride (int): 
        scales (list):[a] stores the anchor's scale relative to the feature map
        ars (list):[b] stores the aspect ratio of the anchor
    Return:
        locally_anchors (tensor):[a*b,4], coordinates obey the format `xyxy`
    """
    stride=torch.tensor(stride).float()
    scales=torch.tensor(scales).float()
    ars=torch.tensor(ars).float()

    n_scale,n_ar=len(scales),len(ars)
    ars=ars.sqrt()[:,None] # [n_ar, 1]

    base_anchors=scales[:,None,None].expand(-1,n_ar,2)/\
        torch.cat([ars,1/ars],dim=1) # [n_scale,n_ar,2]
    base_anchors*=stride

    stride/=2
    base_anchors=torch.cat(
        [stride.expand(n_scale,n_ar,2),
        base_anchors],
        dim=2
        ) # [n_scale,n_ar,4],fmt is `ccwh`
    base_anchors=base_anchors.view(-1,4) # [n_scale*n_ar,4]

    # change to `xyxy`
    base_anchors=ccwh2xyxy(base_anchors,inplace=True)

    return base_anchors


def t_box_iou(A,B):
    r"""Calculate iou between two boxes :attr:`A`
    and :attr:`B` obeys the format `xyxy`

    Args:
        A (tensor): [a,4]
        B (tensor): [b,4]
    Return:
        ious (tensor): [a,b] ::math:`ious_{ij}`
    denotes iou of `A_i` and `B_j`
    """
    a=A.shape[0]
    b=B.shape[0]
    AreaA=A[:,2:]-A[:,:2]
    AreaA=AreaA[:,0]*AreaA[:,1] # [a]
    AreaB=B[:,2:]-B[:,:2]
    AreaB=AreaB[:,0]*AreaB[:,1] # [b]
    
    AreaA=AreaA[:,None].expand(a,b) 
    AreaB=AreaB[None].expand(a,b)
    A=A[:,None].expand(a,b,4)
    B=B[None].expand(a,b,4)
    

    max_l=torch.where(A[:,:,0]>B[:,:,0],A[:,:,0],B[:,:,0])
    min_r=torch.where(A[:,:,2]<B[:,:,2],A[:,:,2],B[:,:,2])
    max_t=torch.where(A[:,:,1]>B[:,:,1],A[:,:,1],B[:,:,1])
    min_b=torch.where(A[:,:,3]<B[:,:,3],A[:,:,3],B[:,:,3])

    width=min_r-max_l
    height=min_b-max_t
    width=width.clamp(min=0)
    height=height.clamp(min=0)

    union=width*height # [a,b]
    ious=union/(AreaA+AreaB-union)

    return ious

def resize_boxes(boxes,scales):
    r"""Resize the boxes with scales
    Args:
        boxes (np.ndarray): [n,4]
        scales (tupple): scales[0] is the w direction, scales[1] is the h direction
    Return:
        boxes (np.ndarray): [n,4]
    """
    boxes=boxes.copy()
    boxes[:,[0,2]]*=scales[0]
    boxes[:,[1,3]]*=scales[1]
    
    return boxes

def test_torch():
    print('test torch...')
    b1=torch.tensor([[.0,.0,1.0,1.0],[.0,.0,1.0,2.0]])
    b2=torch.tensor([[.0,.0,.5,.5],[.5,.5,1.5,1.5]])
    print(t_box_iou(b1,b2))
    
    a1=[1,2,3,4]
    a2=[5,6,7]
    print(np.meshgrid(a1,a2))
    print(t_meshgrid_2d(torch.tensor(a1),torch.tensor(a2)))

    loc_anchors=get_locally_anchors()
    print(loc_anchors)

    anchors=get_anchors(loc_anchors,16,32)
    print(anchors)

    encoded= encode_box(b1,b2)
    decode=decode_box(encoded[None],b2[None])[0]
    print(b1,decode)

def get_check_point():
    pat=re.compile("""weights_([\d]+)_([\d]+)""")
    base_dir='./models/'
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
    main()
    # test_net()
    # test()
    # test_torch()
