# coding=UTF-8

import torch
from torch.autograd import Variable
from config import cfg

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
        anchor_boxes=anchor_boxes.cuda(real_boxes.device.index)
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
        anchor_boxes=anchor_boxes.cuda(param_boxes.device.index)
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
        anchors=anchors.cuda(cfg.device_id)

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
