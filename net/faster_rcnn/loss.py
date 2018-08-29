# coding=UTF-8

import torch
from tqdm import tqdm
from config import cfg

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
        reg_loss=_smooth_l1_loss(out_box,gt_box,cfg.rpn_sigma)
        loss=cls_loss/n_cls+reg_loss/n_reg*_lambda

        tqdm.write("rpn loss=%.5f: reg=%.5f, cls=%.5f" %(loss.item(),reg_loss.item(),cls_loss.item()),end=",\t ")
        return loss

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
        reg_loss=_smooth_l1_loss(out_box,gt_box,cfg.rcnn_sigma)

        # loss=cls_loss/n_cls+reg_loss/num_pos*_lambda
        loss=cls_loss/n_cls+reg_loss*_lambda/(num_pos**2)
        # loss=cls_loss

        return loss
