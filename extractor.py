import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import cfg
from data import TrainDataset,TestDataset
from net.faster_rcnn.vgg16_caffe import  decom_vgg16 as vgg16_caffe

class ExtractorDataset(TrainDataset):
    def __init__(self):
        super(ExtractorDataset,self).__init__()
    
    def __getitem__(self,idx):
        img,box,label,scale,cur_img_size=TrainDataset.__getitem__(self,idx)
        return img,box,label,scale,idx,cur_img_size

class TestExtractorDataset(TestDataset):
    def __init__(self):
        super(TestExtractorDataset,self).__init__()
    
    def __getitem__(self,idx):
        img,src_img_size,cur_img_size,box,label,diff=TestDataset.__getitem__(self,idx)
        return img,src_img_size,cur_img_size,box,label,diff,idx

def main():
    for_trainig=False
    net,_=vgg16_caffe()
    net.train()

    if cfg.use_cuda:
        device_id=cfg.device_id
        net.cuda(device_id)
    
    if for_trainig:
        data_set=ExtractorDataset()
    else:
        data_set=TestExtractorDataset()

    data_loader=DataLoader(data_set,batch_size=1,shuffle=False,drop_last=False)
    if for_trainig:
        for _,(imgs,boxes,labels,scale,idx,img_size) in tqdm(enumerate(data_loader)):
            if cfg.use_cuda:
                imgs=imgs.cuda(device_id)
            feat=net(imgs)
            feat=feat.cpu().detach().numpy()
            boxes=boxes.numpy()
            labels=labels.numpy()
            scale=scale.numpy()
            idx=idx.numpy()

            for pos,num in enumerate(idx):
                fname='%s%d'% (cfg.train_feat_dir,num)
                content={'feat':feat[pos],
                    'box':boxes[pos],
                    'label':labels[pos],
                    'scale':scale[pos],
                    'img_size':img_size[pos]}
                torch.save(content,fname)
    else:
        for _,(imgs,src_img_sizes,cur_img_sizes,boxes,labels,diffs,indices) \
            in tqdm(enumerate(data_loader)):
            if cfg.use_cuda:
                imgs=imgs.cuda(device_id)
                
            feats=net(imgs)
            feats=feats.cpu().detach().numpy()
            src_img_sizes=src_img_sizes.numpy()
            cur_img_sizes=cur_img_sizes.numpy()
            boxes=boxes.numpy()
            labels=labels.numpy()
            diffs=diffs.numpy()
            
            for feat,src_img_size, cur_img_size,box,label,diff,idx in \
                zip(feats,src_img_sizes,cur_img_sizes,boxes,labels,diffs,indices):
                fname='%s%d'%(cfg.test_feat_dir,idx)
                content={'feat':feat,
                    'src_img_size':src_img_size,
                    'cur_img_size':cur_img_size,
                    'box':box,
                    'label':label, 
                    'diff':diff
                }
                torch.save(content,fname)


if __name__ == '__main__':
    main()