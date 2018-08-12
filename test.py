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
from torchvision.models import densenet121 as Backbone
import torchvision
from collections import OrderedDict
import matplotlib.pyplot as plt
from roi_pooling import RoIPool
from nms.pth_nms import NMSLayer
from tqdm import tqdm
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
        x = self.layer4(x)
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

class VOCDataset(Dataset):  
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]    

    def __init__(self,voc_root,list_path,easy_mode=True):
        super(VOCDataset,self).__init__()
        self._root=voc_root
        self._list_path=list_path
        self._img_list=[]
        self._gt=[]
        self._shorter_len=600
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
        boxes[:,1:]/=np.array([w,h,w,h])

        if not self._easy_mode:
            if h>w:
                img=cv2.resize(img,(self._shorter_len,int(float(self._shorter_len)*h/w)),interpolation=cv2.INTER_LINEAR)
            else:
                img=cv2.resize(img,(int(float(self._shorter_len)*w/h),self._shorter_len),interpolation=cv2.INTER_LINEAR)
        else:
            img=cv2.resize(img,(self._easy_h_w,self._easy_h_w),interpolation=cv2.INTER_LINEAR)

        img=img[:,:,::-1] # convert bgr to rgb
        img=img.astype('float32')
        img-=127. # zero mean
        img=img/128.0  # normalization...
        img=img.transpose(2,0,1)

        return img,boxes[:,1:],(boxes[:,0]).astype('int')

        raise NotImplementedError('__getitem__() not completed...')

    def __len__(self):
        return len(self._img_list)



class ROIPooling(torch.nn.Module):
    
    def __init__(self,_size):
        """
        input:
            _size (tuple): output size of the roi
        """
        super(ROIPooling,self).__init__()
        # self._stride=stride  # the scaling ratio of the bounding box 
        self._size=_size    # (h,w) the size of the feature map after roi pooing
        

    def forward(self,x,rois,fmt):
        """
        perform roi pooling on the features
        input:
            x (tensor): [b,c,h,w], feature before the roi pooling
            rois (tensor): [N,5], set of roi, [img_id,xmin,ymin,xmax,ymax], the coordinate is on the oringinal image...
        output:
            roi_features (tensor): [N,c,self._size[0],self._size[1]] 
        """
        assert fmt in ['xyxy','ccwh']
        assert rois.shape[1]==5

        img_ids=rois[:,0] # [N,]
        rois=rois[:,1:]

        if fmt=='ccwh':
            # convert to `xyxy`
            rois[:,:2]-=rois[:,2:]/2
            rois[:,2:]+=rois[:,:2]

        b,fc,fh,fw=x.shape
        # since the box belongs to [0,1], we should scale it to [0,h] and [0,w]
        rois[:,0]*=fw
        rois[:,1]*=fh
        rois[:,2]*=fw
        rois[:,3]*=fh


        raise NotImplementedError("roi pooling, silly b...")

class DetectorBlock(torch.nn.Module):
    def __init__(self,input_dim=7*7*1024,data_set=VOCDataset):
        super(DetectorBlock,self).__init__()
        self.fc_3_4=torch.nn.Sequential(OrderedDict([
            ('fc3',Linear(input_dim,1024)),
            ('relu3',ReLU(inplace=True)),
            ('fc4',Linear(1024,256)),
            ('relu4',ReLU(inplace=True))
        ]))
        self.classfier=torch.nn.Sequential(OrderedDict([
            ('fc5_1',Linear(256,len(data_set.classes)+1)),# plus 1 for the background
            ('relu5_1',ReLU(inplace=True)),
            ('softmax5_1',Softmax())
        ]))
        self.box_reg=torch.nn.Sequential(OrderedDict([
            ('sfc5_2',Linear(256,4))
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
    def __init__(self,data_set=VOCDataset):
        super(MyNet,self).__init__()
        # if input_size[0] != input[1]:
        #     raise ValueError('input size must be equal...')
        # if input_size[0] % 32 != 0 and input_size[1] % 32 !=0:
        #     raise ValueError('input size must be modded by 32')
        # self.stride=32
        anchor_aspect,anchor_scale=np.meshgrid([0.5,1,2],[128,256,512]) # (128,256,512) (1:2,1:1,2:1)
        boxes_w=anchor_scale*np.sqrt(anchor_aspect) # [3,3]
        boxes_h=anchor_scale/np.sqrt(anchor_aspect) # [3,3]
        boxes_wh=np.concatenate([boxes_w.reshape(-1,1),boxes_h.reshape(-1,1)],axis=1) # [9,2]

        self.anchors_wh=boxes_wh # [anchor_num,2], width and height for the anchors 
        self.anchor_num=len(boxes_wh)

        self.roi_size=[7,7]

        # self.extractor=Backbone(pretrained=True).features 
        self.extractor=resnet18(pretrained=True) 
        # self.roi_pooling=ROIPooling(self.roi_size)
        self.roi_pooling=RoIPool(self.roi_size[0],self.roi_size[1],1.0)
        
        self.detector=DetectorBlock(self.roi_size[0]*self.roi_size[1]*512,data_set=data_set)
        self.rpn=torch.nn.Sequential(OrderedDict([
            ('rpn_conv1',Conv2d(512,512,(3,3),1,padding=1)),
            ('rpn_bn_1',BatchNorm2d(512)),
            ('rpn_relu1',ReLU(inplace=True)),
            ('rpn_conv2',Conv2d(512,(4+2)*self.anchor_num,(1,1),1,padding=0)), # 4+2 means 4 coordinates, 1 objectness and 1 not objectness, 9 is the number of anchors
        ]))
        
        self.nms_layer=NMSLayer()
        # self.anchor_boxes=[] # default anchor boxes in the conv feature map.
        # fh=input_size[0]/self.stride
        # fw=input_size[1]/self.stride
        # grid_y,grid_x=np.meshgrid(np.linespace(0,fh,fh),np.linspace(0,fw,fw))
        
        # grid_y=grid_y+.5 # let it center
        # grid_x=grid_x+.5 # let it center
        # default_boxes=np.concatenate([[grid_x],[grid_y]],axis=0) # [2,w,h]
        # default_boxes=np.tile(np.expand_dims(default_boxes,3),[1,1,1,9]) # [2,w,h,9]
        # default_boxes=default_boxes.transpose(1,2,3,0)    # [w,h,anchor_num,2], 2 for the x and y
        for name,param in self.rpn.named_parameters():
            if 'conv' in name:
                torch.nn.init.normal_(param.data,mean=0,std=0.01) # as the paper said, gausian distribution of mean=0, std=0.01

    
    def train_rpn(self,imgs,box,loss_func,opt):
        """
        trian the rpn, batch=1?
        input:
            imgs (tensor): [1,3,h,w] input images:[1,3,416,416],type: Variable
            box (tensor): [N,4], coordinates `ccwh` 
            opt (object): torch optimization object
        """
        assert imgs.shape[0] == 1  # every time train the rpn for a single image...
        assert box.shape[0]==1
        box=box.reshape(-1,4)
        x=self.extractor(imgs)
        x=self.rpn(x) # [1,anchor_num*(4+2),h,w]
        b,c,h,w=x.shape 
        
        # anchors (ndarray[np.float64]), drop_mask (ndarray[np.bool])
        anchors,drop_mask= self.get_anchors(x,imgs.shape[2:][::-1],drop=True) # drop the anchor crossing the boundaries
        anchors=anchors[drop_mask] # drop the boundaries for training...
        
        # out_box (tensor[float32]), out_cls (tensor(float32))
        out_box,out_cls=self.convert_rpn_feature(x) # convert the feature, output [b,N,4], [b,N,2]
        out_box=out_box.view(-1,4) # [N,4]
        out_cls=out_cls.view(-1,2) # [N,2]

        # ps:in pytorch mask should be type of `uint8`, but in numpy, mask must be the `bool` type
        out_box=out_box[torch.tensor(drop_mask.astype('uint8'))] # drop the boundary 
        out_cls=out_cls[torch.tensor(drop_mask.astype('uint8'))] # drop the boundary
   
        idx_pos_anchor,idx_pos_box,idx_neg = MyNet.match_anchor(anchors,box)
        pos_anchors=anchors[idx_pos_anchor]
        target_param_box=MyNet.encode_box(box[idx_pos_box],pos_anchors)

        # prepare the 256 mini-batch
        pos_cls=out_cls[idx_pos_anchor]
        neg_cls=out_cls[idx_neg]
        pos_out_box=out_box[idx_pos_anchor]

        # for first 128 positive samples (for classification)
        pos_rand_idx=np.random.permutation(len(pos_cls))
        pos_rand_idx=pos_rand_idx[:128] 
        # for the rest negative samples (for clssification)
        neg_rand_idx=np.random.permutation(len(neg_cls))
        neg_rand_idx=neg_rand_idx[:256-len(pos_rand_idx)]
        
        # prepare parameters for loss function
        # localtion parameters
        out_pred_box=pos_out_box[pos_rand_idx]
        gt_box=target_param_box[pos_rand_idx]

        # class parameters
        out_pos_cls= pos_cls[pos_rand_idx]
        out_neg_cls= neg_cls[neg_rand_idx]

        # compute the loss
        # loss_func=RPNMultiLoss()
        
        gt_box=Variable(gt_box)
        if next(self.parameters()).is_cuda:
            gt_box=gt_box.cuda()

        loss=loss_func(out_pos_cls,out_neg_cls,out_pred_box,gt_box,h*w)

        # gradient descent
        loss.backward()
        opt.step()
        opt.zero_grad()

        return loss

    def train_fast_rcnn(self,imgs,boxes,labels,loss_func,opt):
        """
        trian the fast r-cnn with the rpn
        input:
            imgs (tensor): [b,c,h,w] b==1
            boxes (tensor): [b,N,4], format is `xyxy` 
            labels (tensor): [b,N,]
            loss_func (nn.Module): the loss function
            opt (nn.opt.Optimizer)
        """
        assert imgs.shape[0]==1 
        img_features,rois=self.first_stage(imgs,2000) # [b,c,w,h],[N,1+4]        
        b,c,fh,fw=img_features.shape
        
        # only support batch size of one
        boxes=boxes[0]
        labels=labels[0]

        # match the rois and the gt boxes
        idx_pos_roi,idx_pos_box,idx_neg=MyNet.match_anchor(rois[:,1:],boxes,thresh_pos=.5,thresh_neg=(.1,.5),Afmt='xyxy')
        
        # chaneg the incices, select 16 positive and 48 negative
        pos_rand_idx=np.random.permutation(len(idx_pos_roi))[:16]
        neg_rand_idx=np.random.permutation(len(idx_neg))[:48]
        

        idx_pos_roi=idx_pos_roi[pos_rand_idx]
        idx_pos_box=idx_pos_box[pos_rand_idx]
        idx_neg=idx_neg[neg_rand_idx]

        # use the pytorch's mask format
        # idx_pos_roi=torch.tensor(idx_pos_roi.astype('uint8'))
        # idx_pos_box=torch.tensor(idx_pos_box.astype('uint8'))
        # idx_neg=torch.tensor(idx_neg.astype('uint8'))

        # prepare the box 
        pos_rois=rois[idx_pos_roi]
        pos_rois_corresbonding_gt=boxes[idx_pos_box]
        pos_rois_corresbonding_gt_label=labels[idx_pos_box]
        neg_rois=rois[idx_neg]

        # number
        num_pos_roi=len(pos_rois)
        num_neg_roi=len(neg_rois)

        # rescale for the roi pooling...
        for i in range(4):
            pos_rois[:,1+i]*= fw if i % 2 ==0 else fh
            neg_rois[:,1+i]*=fw if i % 2 ==0 else fh

        # get the roi pooling featres
        x=self.roi_pooling(img_features,torch.cat([pos_rois,neg_rois],dim=0)) # [num_pos_roi+num_neg_roi,c,7,7]

        # re-normalize
        for i in range(4):
            pos_rois[:,1+i]/=fw if i % 2 == 0 else fh
            neg_rois[:,1+i]/=fw if i % 2 == 0 else fh

        # [num_pos_rois+num_neg_roi,1+obj_cls_num], [num_pos_roi+num_neg_roi,4]
        out_cls,out_reg_box=self.detector(x)  

        # the img_id is useless, so remove it
        pos_rois=pos_rois[:,1:]
        # change the box to `ccwh`
        pos_rois[:,2:]-=(pos_rois[:,:2]).clone()
        pos_rois[:,:2]+=(pos_rois[:,2:]/2).clone()
        pos_rois_corresbonding_gt[:,2:]-=(pos_rois_corresbonding_gt[:,:2]).clone()
        pos_rois_corresbonding_gt[:,:2]+=(pos_rois_corresbonding_gt[:,2:]/2).clone()

        target_box=MyNet.encode_box(pos_rois_corresbonding_gt,pos_rois) # [num_pos_roi+num_neg_roi,4]
        out_reg_box=out_reg_box[:num_pos_roi]

        out_pos_cls=out_cls[:num_pos_roi]
        out_neg_cls=out_cls[num_pos_roi:]
        loss=loss_func(out_pos_cls,pos_rois_corresbonding_gt_label,out_neg_cls,out_reg_box,target_box)

        # gradient descent
        loss.backward()
        opt.step()
        opt.zero_grad()

        return loss


    def first_stage(self,x,num_prop):
        img_size=x.shape[2:][::-1]
        img_features=self.extractor(x) # [b,c,h,w]
        rois=self.rpn(img_features) # [b,anchor_num*(4+2),w,h]
        b,c,h,w=img_features.shape
        rois,labels=self.convert_rpn_feature(rois) # [b,N,4],[b,N,2]

        # anchors in the feature map 
        anchors=self.get_anchors(img_features,img_size) # [N,4], ps: do not the train the rpn, en? 
        anchors=np.tile(anchors[None],[b,1,1]) # [b,N,4]
        anchors=torch.tensor(anchors).float() # to torch.tensor

        # rois [b,N,4]: (center_x,center_y,w,h)
        rois= MyNet.decode_box(rois,anchors) # decode to the normalized box

        # clip the boundary
        # convert to `xyxy` 
        rois[:,:,:2]-=(rois[:,:,2:]/2).clone()
        rois[:,:,2:]+=(rois[:,:,:2]).clone()

        # clip the boundary
        # rois[:,:,:2]=np.maximum(rois[:,:,:2],.0) # `left`, `top` should be greater than zero`
        # rois[:,:,2:]=np.minimum(rois[:,:,2:],1.0) # `right`, `bottom` should be samller than 1 
        # rois[:,:,:2][rois[:,:,:2]<0]=0
        # rois[:,:,2:][rois[:,:,2:]>1.0]=1.0
        rois[:,:,:2]=torch.clamp(rois[:,:,:2].clone(),min=.0,max=1.)
        rois[:,:,2:]=torch.clamp(rois[:,:,2:].clone(),min=.0,max=1.)

        # nms for each image
        rois_with_id=torch.empty(0)
        if rois.is_cuda:
            rois_with_id=rois_with_id.cuda()
        for i in range(b):
            temp=self.nms(torch.cat([rois[i],labels[i]],dim=1),img_id=i) # [M,1+4+2]
            rois_with_id=torch.cat([rois_with_id,temp], dim=0) # [M'+M,1+4+2]
        
        rois=rois_with_id[:,:1+4] # [N',1+4]
        labels=rois_with_id[:,1+4:] # [N',cls_num]

        # select top-N
        _,idx=labels[:,0].sort(descending=True)
        idx=idx[:num_prop]
        rois=rois[idx]

        return img_features,rois  

    def convert_fast_rcnn(self,classes,boxes,rois):
        """
        convert the output of the fast r-cnn to the real box with nms
        input:
            classes (tensor): [N,obj_cls_num+1]
            boxes (tensor): [N,4], parameters of bounding box regression 
            rois (tensor): [N,1+4], the coordinate obeys format `xyxy` (normalized at [0,1])
        output:
           res (list[object]): [img_num] 
        """
        res=[]
        for i in range(len(rois)):
            # for the i=th image
            
            # find the i-th im
            mask=rois[:,0]==i
            i_rois=rois[mask] # [M,1+4]
            i_param_boxes=boxes[mask] # [M,4]
            i_param_cls=classes[mask] # [M,1+obj_cls_num]
            if len(i_rois)==0:
                # no other images
                break 
            # the image_id is useless, so remove it
            i_rois=i_rois[:,1:] # [M,4]
            
            # change to `ccwh`
            i_rois[:,2:]-=(i_rois[:,:2]).clone()
            i_rois[:,:2]+=(i_rois[:,2:]/2).clone()

            r_boxes= MyNet.decode_box(i_param_boxes[None],i_rois[None]) # [M,4], fmt is `ccwh`
            r_boxes=r_boxes[0]
            
            # change to `xyxy`
            r_boxes[:,:2]-=(r_boxes[:,2:]/2).clone()
            r_boxes[:,2:]+=(r_boxes[:,:2]).clone()

            # remove the neg_cls_score and nms
            rois=self.nms(torch.cat([r_boxes,i_param_cls[:,1:]],dim=1)) # [M,4+obj_cls_num)]
            res.append(rois)
        return res

        raise NotImplementedError()

    def forward(self,x):
        img_features,rois=self.first_stage(x,300)        
        b,c,fh,fw=img_features.shape

        # since the box belongs to [0,1], we should scale it to [0,h] and [0,w]
        # attention: rois[:,0] is the images id...
        rois[:,1]*=fw
        rois[:,2]*=fh
        rois[:,3]*=fw
        rois[:,4]*=fh

        # roi pooling...
        x=self.roi_pooling(img_features,rois)
        classes,boxes=self.detector(x)

        # re-normalize 
        rois[:,1]/=fw
        rois[:,2]/=fh
        rois[:,3]/=fw
        rois[:,4]/=fh

        # convert and nms
        res=self.convert_fast_rcnn(classes,boxes,rois)

        return res

    def nms(self,rois,thresh=.7,img_id=None,_use_ext=False):
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
                order=self.nms_layer(dets,thresh=thresh)
                mask=torch.full([len(dets)],1,dtype=torch.uint8)
                mask[order]=0
                rois[:,i+4][mask]=0
            else:
                _,indices=rois[:,i+4].sort(descending=True)
                # deprecated...
                # for k in range(len(indices)):
                #     idx_k=indices[k]
                #     box_k=rois[idx_k:idx_k+1,:4]
                #     if rois[idx_k,i+4] <1e-6:
                #         continue
                #     for j in range(k+1,len(indices)):
                #         idx_j=indices[j]
                #         box_j=rois[idx_j:idx_j+1,:4]
                #         box_iou=MyNet.intersection_overlap_union(box_j,box_k)
                #         if box_iou[0]>=thresh:
                #             rois[idx_j,i+4]=0
                #####################################################
                # new... it might be faster...
                while len(indices)>1:
                    idx_k=indices[0]
                    idx_rest=indices[1:]
                    box_k=rois[[idx_k],:4]
                    box_rest=rois[idx_rest,:4]
                    iou=MyNet.intersection_overlap_union(box_k.expand([len(box_rest),4]),box_rest)
                    
                    keep_mask=iou<thresh
                    rois[indices,i+4][1:][1-keep_mask]=0
                    indices=indices[1:][keep_mask]
        
        sorted_rois,_=rois[:,4:].max(dim=1)
        rois=rois[sorted_rois>1e-6] # [M,4+cls_num] 

        # append the id to the rois if valid
        if img_id is not None: 
           extra_dim=torch.full([len(rois),1],img_id)
           if rois.is_cuda:
               extra_dim=extra_dim.cuda()
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
    
    
    @staticmethod
    def xyxy2ccwh(boxes):
        """
        convert the box from format of 'xyxy' to 'ccwh'
        """
        boxes=np.array(boxes) # do not effect the original boxes
        boxes[:,2:]-=boxes[:,:2]
        boxes[:,:2]+=boxes[:,2:]/2
        return boxes
    
    @staticmethod
    def ccwh2xyxy(boxes):
        """
        convert the box from format of 'ccwh' to 'xyxy'
        """
        boxes=np.array(boxes) # do not effect the original boxes
        boxes[:,:2]-=boxes[:,2:]/2
        boxes[:,2:]+=boxes[:,:2]
        return boxes
    
    @staticmethod
    def iou_torch(A,B):
        """
        input:
            A (tensor): [N,4], format is `xyxy`
            B (tensor): [N,4], format is `xyxy`
        return:
            iou: [N], iou[i] is the iou between A[i] and B[i]
        """
        maxl,_=torch.max(torch.cat([A[:,0:1],B[:,0:1]],dim=1),dim=1)
        minr,_=torch.max(torch.cat([A[:,2:3],B[:,2:3]],dim=1),dim=1)
        maxt,_=torch.max(torch.cat([A[:,1:2],B[:,1:2]],dim=1),dim=1)
        minb,_=torch.min(torch.cat([A[:,3:4],B[:,3:4]],dim=1),dim=1)
        
        dw=minr-maxl
        dh=minb-maxt
        dw[dw<0]=.0
        dh[dh<0]=.0

        union=dw*dh
        AreaA=A[:,2:]-A[:,:2]
        AreaA=AreaA[:,0]*AreaA[:,1]
        AreaB=B[:,2:]-B[:,:2]
        AreaB=AreaB[:,0]*AreaB[:,1]
        iou=union/(AreaA+AreaB-union)
        return iou

    @staticmethod
    def intersection_overlap_union(A,B):
        """
        compute the iou
        input:
            A: [N,4], format is 'xyxy'
            B: [N,4], format is 'xyxy'
        return:
            iou: [N], iou[i] is the iou between A[i] and B[i]
        """
        if isinstance(A,torch.Tensor) or isinstance(B,torch.Tensor):
            return MyNet.iou_torch(A,B)
            
        maxl=np.max(np.concatenate([A[:,0:1],B[:,0:1]],axis=1),axis=1)
        minr=np.min(np.concatenate([A[:,2:3],B[:,2:3]],axis=1),axis=1)
        maxt=np.max(np.concatenate([A[:,1:2],B[:,1:2]],axis=1),axis=1)
        minb=np.min(np.concatenate([A[:,3:4],B[:,3:4]],axis=1),axis=1)
        
        dw=minr-maxl
        dh=minb-maxt
        dw=np.maximum(dw,0)
        dh=np.maximum(dh,0)

        union=dw*dh
        AreaA=A[:,2:]-A[:,:2]
        AreaA=AreaA[:,0]*AreaA[:,1]
        AreaB=B[:,2:]-B[:,:2]
        AreaB=AreaB[:,0]*AreaB[:,1]
        iou=union/(AreaA+AreaB-union)
        return iou
        

    @staticmethod
    def iou_cartesian(A,B,Afmt='ccwh'):
        """
        compute the intersectoin overlap union for each of the pair boxes in A and B
        input:
            A: [N,4], N boxes
            B: [M,4], M boxesv
            fmt: format of the boxes:
                'ccwh': (center_x,center_y,w,h),default format
                'xyxy': (left,top,right,bottom)
        output:
            An N x M matrix C, where C[i,j] is the iou between A[i] and B[j]
        """
        if Afmt not in ['ccwh','xyxy']:
            raise ValueError('fmt should be `ccwh` or `xyxy`')

        # do not effect the oringinal boxes
        if Afmt=='ccwh':
            # convert (center_x,center_y,w,h) to (left,top,right,bottom)
            A=MyNet.ccwh2xyxy(A)
            # B=MyNet.ccwh2xyxy(B)
        else:
            if isinstance(A,Variable):
                A=np.array(A.detach())
            A=np.array(A) # copy
        B=np.array(B) # copy

        N,_=A.shape
        M,_=B.shape
        CarA=np.tile(A.reshape(N,1,4),[1,M,1])
        CarB=np.tile(B.reshape(1,M,4),[N,1,1])
        CarA=CarA.reshape(-1,4) # [N*M,4]
        CarB=CarB.reshape(-1,4) # [N*M.4]

        iou=MyNet.intersection_overlap_union(CarA,CarB)
        C=iou.reshape(N,M)
        return C

    @staticmethod
    def match_anchor(anchors,boxes,thresh_pos=.7,thresh_neg=.3,Afmt='ccwh'):
        """
        match the anchors and gt boxes in a single image

        input:
            anchors (ndarray): [N,4]
            boxes (ndarray): [M,4]
            thresh_pos_anchor (float): threshold for the positive matching 
            thresh_neg (float or (tupple)): threshold for the negative matching
            Afmt (str): format of the anchors' coordinates
        output:
            idx_pos_anchor (list[int]): [C,], stores the index of the positive anchor
            idx_pos_box (list[int]): [C,], stores the index of the box with the correspondoh anchor
            idx_neg (list[int]): [D,], idx_neg stores the indices for the negative boxes 
        a ground truth can be matched to many anchors, but a particular
        anchor can be matched only once, there are two strategies for matching:
            1. match the anchor and box with the highest iou (from all pairs of the box and anchor)
            2. match the anchor and box if their iou is larger than a threshold
        """
        ious=MyNet.iou_cartesian(anchors, boxes,Afmt)

        # find the positive matches
        # firstly, find the highest iou for the all of the gt boxes
        ious_mask=np.array(ious)
        idx=np.full(len(anchors),-1) # create a buffer
        while True:
            i,j=np.unravel_index(ious_mask.argmax(),ious_mask.shape) # get i,j for the highest iou, verbose operation in numpy...
            if ious_mask[i,j]< 1e-6: # means case 1 has been completed
                break
            idx[i]=j
            ious_mask[i,:]=0  # the anchor can`t be matched for the other boxes 
            ious_mask[:,j]=0  # the box can`t be matched for he other anchors`
        
        # secondly, find the matching for the rest anchors
        # mask=(idx<0) *(ious.argmax(axis=1)+1) # if the i-th anchor has already been matched, then mask[i]=0
        # mask-=1  # mask[i]=-1 means i-th anchor is matched
        mask2=(ious.max(axis=1)>=thresh_pos) *(idx<0) # only need > threshold and not matched yet
        idx[mask2]=ious.argmax(axis=1)[mask2]
        idx_pos=idx

        idx_pos_anchor=np.where(idx>=0)[0] # get the index of the positive anchors
        idx_pos_box=idx[idx_pos_anchor]  # get the positive anchors' corresponding box

        # for the negative index
        mask=idx<0  # negative matching should be in the rest of the anchors
        if isinstance(thresh_neg,float):
            mask=mask*(np.max(ious,axis=1)<thresh_neg)
        elif isinstance(thresh_neg,tuple):
            temp=np.max(ious,axis=1)
            mask=mask*(temp>=thresh_neg[0])*(temp<thresh_neg[1])
        else:
            raise ValueError('`thresh_neg must be `float` or `tuple(float,float)`')
        idx_neg=np.where(mask)[0]

        return idx_pos_anchor,idx_pos_box, idx_neg
        raise NotImplementedError()

    @staticmethod
    def encode_box(real_boxes,anchor_boxes):
        """
        encode the real_box to the corresponding parameterized coordinates
        input:
            real_boxes:[N,4], whose format of 4 is (center_x,center_y,w,h)
            anchor_boxes:[N,4], i-th anchor is responsible for the i-th real box
        output:
            parameterized boxes
        """
        assert len(real_boxes) == len(anchor_boxes)
        if not isinstance(anchor_boxes,torch.Tensor):
            anchor_boxes=torch.from_numpy(anchor_boxes.astype('float32'))
        if real_boxes.is_cuda:
            anchor_boxes=anchor_boxes.cuda()
        elif anchor_boxes.is_cuda:
            real_boxes=real_boxes.cuda()
        encoded_xy=(real_boxes[:,:2]-anchor_boxes[:,:2])/anchor_boxes[:,2:]  # [N,2]
        encoded_wh=torch.log(real_boxes[:,2:]/anchor_boxes[:,2:])  # [N,2]

        return torch.cat([encoded_xy,encoded_wh],dim=1)
        raise NotImplementedError()

    @staticmethod
    def decode_box(param_boxes,anchor_boxes):
        """
        translate the parameterized box to the real boxes, real boxes
        are not the ground truths, just refer to boxes with format (center_x,center_y,w,h)
        input:
            param_boxes (tensor) : [b,N,4], contain parameterized coordinates
            anchor_boxes (tensor) : [b,N,4], the rpn anchors with the real boxes
        output:
            boxes (tensor) : [b,N,4], whose format is (center_x,center_y,w,h)
        """
        assert param_boxes.shape == anchor_boxes.shape 
        if param_boxes.is_cuda:
            anchor_boxes=anchor_boxes.cuda()
        elif anchor_boxes.is_cuda:
            param_boxes=param_boxes.cuda()
        decoded_xy=param_boxes[:,:,:2]*anchor_boxes[:,:,2:]+anchor_boxes[:,:,:2] # [b,N,2]
        decoded_wh=torch.exp(param_boxes[:,:,2:])*anchor_boxes[:,:,2:] # [b,N,2]
        return torch.cat([decoded_xy,decoded_wh],dim=2) # [b,N,4]
        raise NotImplementedError()

    def get_anchors(self,x,img_size,drop=False):
        """
        get the default anchor box on the feature map
        input: 
            x (tensor):convolutional feature map [c,h,w], single image?...
            img_size (tupple):(w,h), attention, w is the first
            drop (bool): a mask to the anchors which cross the boundaries
            clip (bool): to clip the boxes cross the boundaries
        output:
            default_boxes:[anchor_num*h*w,4], 4 is for [center_x,center_y,w,h]
        """
        # prepare the anchor boxes...
        b,fc,fh,fw=x.shape
        assert b==1
        grid_x,grid_y=np.meshgrid(np.linspace(0,fw-1,fw),np.linspace(0,fh-1,fh))
        grid_y=grid_y+.5 # let it center
        grid_x=grid_x+.5 # let it center
        grid_x=grid_x/fw # normalization
        grid_y=grid_y/fh # normalization
        default_boxes=np.concatenate([[grid_x],[grid_y]],axis=0) # [2,h,w]
        default_boxes=np.tile(default_boxes[None],[self.anchor_num,1,1,1]) # [anchor_num,2,h,w]

        anchors_wh=self.anchors_wh/np.array([img_size[0],img_size[1]]) # w,h are oringinally based on the resized images, so you must divide it by the stride.
        anchors_wh=anchors_wh.reshape(self.anchor_num,2,1,1) # [anchor_num,2,1,1] 
        anchors_wh= np.tile(anchors_wh,[1,1,fh,fw]) # [anchor_num,2,h,w] 
        default_boxes=np.concatenate([default_boxes,anchors_wh],axis=1) # [anchor_num,4,h,w]
        default_boxes=default_boxes.transpose(0,2,3,1) # [anchor_num,h,w,4]
        default_boxes=default_boxes.reshape(-1,4) # [h*w*anchor_num,4] 

        if drop: 
            # drop the boundary anchors...
            temp_box=MyNet.ccwh2xyxy(default_boxes)
            mask=(temp_box[:,0]>=0) *(temp_box[:,1]>=0)*(temp_box[:,2]<=1) *(temp_box[:,3]<=1)
            return default_boxes,mask # mask is used for drop the rpn output...

        return default_boxes
        raise NotImplementedError('you must implement the function `get_anchors`')

    
class RPNMultiLoss(torch.nn.Module):
    def __init__(self):
        super(RPNMultiLoss,self).__init__()
    
    def forward(self,pos_cls,neg_cls,out_box,gt_box,n_reg,_lambda=10):
        assert len(pos_cls) == len(out_box) # must be the same
        n_cls=len(pos_cls)+len(neg_cls)
        
        # class loss
        cls_loss=-pos_cls[:,0].log().sum()-neg_cls[:,1].log().sum()

        # smooth l1...
        reg_loss=out_box-gt_box
        reg_loss=reg_loss.abs()
        reg_loss=torch.where(reg_loss<1,.5*reg_loss**2,reg_loss-.5).sum()

        loss=cls_loss/n_cls+reg_loss*_lambda/n_reg
        return loss
        raise NotImplementedError()


class FastRCnnLoss(torch.nn.Module):
    def __init__(self):
        super(FastRCnnLoss,self).__init__()
    
    def forward(self,pos_cls,pos_label,neg_cls,out_box,gt_box,_lambda=1):
        # let slot 0 denote the negative class, then you should plus one on the label:
        cls_loss=-pos_cls[[_ for _ in range(len(pos_cls))],(pos_label+1).long()].log().sum()-neg_cls[:,0].log().sum()
        
        # smooth l1...
        reg_loss=out_box-gt_box
        reg_loss=reg_loss.abs()
        reg_loss=torch.where(reg_loss<1,.5*reg_loss**2,reg_loss-.5).sum()

        loss=cls_loss+_lambda*reg_loss

        return loss


        raise NotImplementedError()

def main():
    print("my name is van")
    # let the random counld be the same
    np.random.seed(1234567)
    torch.manual_seed(1234567)
    torch.cuda.manual_seed(1234567)
    data_set=VOCDataset('d:/VOC2007_2012','train.txt',easy_mode=False)
    data_loader=DataLoader(data_set,batch_size=1,shuffle=True,drop_last=False)
   
    data_loader2=DataLoader(VOCDataset('d:/VOC2007_2012','train.txt',easy_mode=True),batch_size=1,shuffle=True,drop_last=False)
    net=MyNet(data_set)
    net.train()
    net.cuda()
    
    epoches=int(1e6)
    rpn_loss=RPNMultiLoss()
    rpn_opt=torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9,weight_decay=0.0005)
    fast_rcnn_loss=FastRCnnLoss()
    for epoch in range(epoches):
        
        # train the rpn
        print('******epoch %d*********' % (epoch))
        for i,(imgs,boxes,labels) in tqdm(enumerate(data_loader)):
            if next(net.parameters()).is_cuda:
                imgs=imgs.cuda()
                boxes=boxes.cuda()
            loss=net.train_rpn(imgs,boxes,rpn_loss,rpn_opt)
            print('rpn loss:%f'%(loss.data) )

            loss=net.train_fast_rcnn(imgs,boxes,labels,fast_rcnn_loss,rpn_opt)
            print('fast r-cnn loss:%f' %(loss.data))
            if epoch>6e4:
                for g in rpn_opt.param_groups:
                    g['lr'] = 0.0001
        for i,(imgs,boxes,labels) in tqdm(enumerate(data_loader2)):
            if next(net.parameters()).is_cuda:
                imgs=imgs.cuda()
                boxes=boxes.cuda()
            loss=net.train_fast_rcnn(imgs,boxes,labels,fast_rcnn_loss,rpn_opt)
            print('fast r-cnn loss:%f' %(loss.data))
        
        torch.save(net,'model.pkl')

if __name__ == '__main__':
    main()
    # test()
    