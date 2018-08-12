# coding=UTF-8
import numpy as np
import time

def xyxy2ccwh(boxes):
        """
        convert the box from format of 'xyxy' to 'ccwh'
        """
        boxes=np.array(boxes) # do not effect the original boxes
        boxes[:,2:]-=boxes[:,:2]
        boxes[:,:2]+=boxes[:,2:]/2
        return boxes
    
def ccwh2xyxy(boxes):
        """
        convert the box from format of 'ccwh' to 'xyxy'
        """
        boxes=np.array(boxes) # do not effect the original boxes
        boxes[:,:2]-=boxes[:,2:]/2
        boxes[:,2:]+=boxes[:,:2]
        return boxes
    
def intersection_overlap_union(A,B):
        """
        compute the iou
        input:
            A: [N,4], format is 'xyxy'
            B: [N,4], format is 'xyxy'
        return:
            iou: [N], iou[i] is the iot between A[i] and B[i]
        """
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
        

def iou_cartesian(A,B,fmt='ccwh'):
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
        if fmt not in ['ccwh','xyxy']:
            raise ValueError('fmt should be `ccwh` or `xyxy`')

        # do not effect the oringinal boxes
        if fmt=='ccwh':
            # convert (center_x,center_y,w,h) to (left,top,right,bottom)
            A=ccwh2xyxy(A)
            B=ccwh2xyxy(B)
        else:
            A=np.array(A) # copy
            B=np.array(B) # copy

        N,_=A.shape
        M,_=B.shape
        CarA=np.tile(A.reshape(N,1,4),[1,M,1])
        CarB=np.tile(B.reshape(1,M,4),[N,1,1])
        CarA=CarA.reshape(-1,4) # [N*M,4]
        CarB=CarB.reshape(-1,4) # [N*M.4]

        iou=intersection_overlap_union(CarA,CarB)
        C=iou.reshape(N,M)
        return C
    
def main():
    starttime=time.time()
    # A=np.random.randn(16200,4)
    # B=np.random.randn(4,4)
    # C=iou_cartesian(A,B)
    # print(C.shape)

    A=np.array([[0,0,1,1],[0.5,0.5,0.8,0.8]])
    C=iou_cartesian(A,A,fmt='xyxy')
    print(C)
    endtime = time.time()
    print(endtime - starttime)

if __name__ == '__main__':
    main()