# coding=UTF-8

import torch
from torch.autograd import Variable
from torch.nn import Conv2d,BatchNorm2d,Softmax,ReLU,LeakyReLU,Linear
import numpy  as np 
from torchvision.transforms import transforms

import torch.nn.functional as F

import torchvision
from collections import OrderedDict
import matplotlib.pyplot as plt
from roi_pool_cupy import RoIPooling2D
from nms.pth_nms import pth_nms as ext_nms 
import time

from tqdm import tqdm

from .vgg16_caffe import decom_vgg16 as vgg16

from .loss import RPNMultiLoss,FastRCnnLoss

from .net_tool import *

class DetectorBlock(torch.nn.Module):
    def __init__(self,input_dim,classes,front_fc=None):
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

class FasterRCNN(torch.nn.Module):
    def __init__(self,classes):
        super(FasterRCNN,self).__init__()
        self.roi_size=[7,7]

        self.extractor,classifier= vgg16()
        rpn_input_features=512
        self.stride=16
       
        self.roi_pooling=RoIPooling2D(self.roi_size[0],self.roi_size[1],1.0/self.stride)
        
        self.loc_anchors=get_locally_anchors()
        self.anchor_num=len(self.loc_anchors)
        self.detector=DetectorBlock(self.roi_size[0]*self.roi_size[1]*rpn_input_features,

        classes=classes,front_fc=classifier if cfg.use_caffe_classifier else None)

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

        self.get_optimizer(lr=cfg.lr,use_adam=cfg.use_adam,weight_decay=cfg.weight_decay)
    
    def get_optimizer(self,lr=1e-3,use_adam=False,weight_decay=0.0005):
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
                    params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]
        if use_adam:
            print("Using Adam optimizer")
            self.optimizer = torch.optim.Adam(params)
        else:
            print("Using SGD optimizer")
            self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer

        
    def train_once(self,imgs,box,label,scale,img_sizes):
        r"""Train the rpn and fast r-cnn together
        Args:
            imgs (tensor): [1,3,h,w]
            box (tensor): [1,N,4]
            label (tensor): [1,N]
            scale (tensor): [1,2] scale for width and height 
            img_sizes (tensor): [1,2] images sizes
        """
        assert imgs.shape[0] == 1
        assert box.shape[0]==1

        # return self.only_train_cls(imgs,box,label)

        t1=time.time()
        img_size,img_feat,anchors,out_rois,\
            out_cls,sampled_rois \
            = self.first_stage(imgs,img_sizes,12000,2000,scale,force_extract=False)
        
        # tqdm.write("max of img_feat:%.5f, sum of img_feat:%.5f"%(img_feat.max(),img_feat.sum()), end=",\t ")
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
        

    def anchor_target(self,anchors,gt_box,img_size,
            thresh_pos=cfg.rpn_thresh_pos,
            thresh_neg=cfg.rpn_thresh_neg):
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
            gt_loc=gt_loc.cuda(gt_box.device.index)

        return gt_loc,assign

    def roi_target(self,rois,gt_box,label,pos_thresh=cfg.rcnn_pos_thresh,
                    neg_thresh_lo=cfg.rcnn_neg_thresh_lo,
                    neg_thresh_hi=cfg.rcnn_neg_thresh_hi):
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


    def first_stage(self,x,img_size,num_prop_before,
        num_prop_after,scale,min_size=16,force_extract=False):
        r"""The first part of the network, including the feature extraction,
        and rpn forwarding
        Args:
            x (tensor): [b,3,H,W], batch of images
            img_size (tensor): [b,2]  image size for each batch 
            num_prop_before (int): remained rois for each image before sorting
            num_prop_after (int): remainted rois for each image after sorting
            scale (tensor[float]): [b,2] stores the scale for width and height...  
            min_size (int): threshold for discarding the boxes...
            force_extract (bool): will forward the extractor network for x if enabled
        Return:
            img_size (shape): image width and height
            img_features (tensor): [b,c,h,w]
            anchors (tensor): [b,N,4] 
            out_rois (tensor): [b,N,4], the output of rpn, parameterized boxes
            out_cls (tensor): [b,N,2], the output of rpn, softmax class
            sampled_rois (tensor): [n',5], sampled_rois for fast-rcnn, fmt is 
        (img_id,left,top,right,bottom)self.fc_3_4[-2].out_features
        """
        # img_size=x.shape[2:][::-1]
        # all the images should be the same size
        assert (img_size[0][None].expand_as(img_size) != img_size).sum() == 0
        img_size=img_size[0]

        t1=time.time()
        if force_extract :
            img_features=self.extractor(x) # [b,c,h,w]
        else:
            img_features=x
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

            rois_i=rois_masked
            tt=out_cls_masked[:,0] # [N',1] idx 0 is the pos

            _,idx=tt.sort(descending=True)    
            idx=idx[:num_prop_before]
            tt=_[:num_prop_before][:,None]
            rois_i=rois_i[idx]

            use_ext=True
            # sort by the pos
            if not use_ext:
                temp=self.nms(torch.cat([rois_i,tt],dim=1),img_id=i) # [M,1+4+1]
                # select top-N
                # here 5 is the pos label
                _,idx=temp[:,5].sort(descending=True) 
                idx=idx[:num_prop_after]
                temp=temp[idx]
            else:
                temp=torch.cat([rois_i,tt],dim=1) # [M,4+1]
                keep_idx=ext_nms(temp, .7)
                temp=temp[keep_idx] # keep_idx is already sorted

                temp=temp[:num_prop_after] # [num_p_a,4+1]
                extra_dim=torch.full([len(temp),1],i).type_as(temp)
                temp=torch.cat([extra_dim,temp],dim=1) # [M',1+4+1]
            
            rois_with_id=torch.cat([rois_with_id,temp], dim=0) # [M'+M,1+4+2]

        # print("Time of nms in first stage:%.3f" %(time.time()-t1))

        sampled_rois=rois_with_id[:,:1+4] # [n',1+4]
        # sampled_cls=rois_with_id[:,1+4:] # [n',cls_num]

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

    def _suppres(self,boxes,prob,cls_num,thresh=.5):
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
            # if cls == 11:
                # print(cls)
            box_cls=boxes[:,cls,:] # [M,4]
            prob_cls=prob[:,cls] # [M]
            box_cls=self.nms(torch.cat([box_cls,prob_cls[:,None]],dim=1),thresh) # [m',5]
            if len(box_cls)==0:
                continue 
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

    def forward(self,x,src_size,offline_feat=False,cur_image_size=None):
        r""" Net Eval
        Args:
            x (tensor[float]): [b,c,h,w]
            src_size (tensor[int]) : [b,2]
            offline_feat (bool): indicats the attr::`x` is from the offline...
            cur_img_size (tensor) [b,2]: it will be ignored when attr::`offline_feat` is `False`
        Return:
            res (list): [b], the result boxes
        """
        if offline_feat:
            current_size=cur_image_size
            assert current_size is not None
        else:
            current_size=x.shape[2:][::-1]
            current_size=torch.tensor(current_size)[None].expand(x.shape[0],-1) # [b,2]
            current_size=current_size.type_as(x).float()
        ratios=src_size/current_size.float() # [b,2]
        ratios=ratios[:,None].expand(-1,2,-1).contiguous().view(-1,4) # [b,4]

        img_size,img_features,\
            anchors,out_rois,\
            out_cls,sampled_rois=self.first_stage(x,current_size,6000,300,
                scale=1./ratios,force_extract= (not offline_feat) )        

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
            dets=rois[:,[0,1,2,3,i+4]]
            order=ext_nms(dets,thresh=thresh)
            mask=torch.full([len(dets)],1,dtype=torch.uint8)
            mask[order]=0
            del order
            rois[:,i+4][mask]=0
        
        sorted_rois,_=rois[:,4:].max(dim=1)
        rois=rois[sorted_rois>1e-6] # [M,4+cls_num] 

        # append the id to the rois if valid
        if img_id is not None: 
           extra_dim=torch.full([len(rois),1],img_id).type_as(rois)
           rois=torch.cat([extra_dim,rois],dim=1) # [M,1+4+cls_num] 
        return rois 


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
    
    def print_net_info(self):
        print("****************\t NET STRUCTURE \t ****************")
        print("********\t Extractor \t ********")
        print(self.extractor)
        print("********\t RPN \t ********")
        print(self.rpn)
        print("********\t ROI Detector\t ********")
        print("\t FC-3-4 \t")
        print(self.detector.fc_3_4)
        print("\t Classifier \t ")
        print(self.detector.classfier)
        print("\t Box Regressor \t")
        print(self.detector.box_reg)
        print("****************\t NET INFO END  \t****************")

