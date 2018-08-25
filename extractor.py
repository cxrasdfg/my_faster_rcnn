import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import cfg
from data import TrainDataset
from test3 import decom_vgg16

class ExtractorDataset(TrainDataset):
    def __init__(self):
        super(ExtractorDataset,self).__init__()
    
    def __getitem__(self,idx):
        img,box,label,scale,cur_img_size=TrainDataset.__getitem__(self,idx)
        return img,box,label,scale,idx,cur_img_size


def main():
    net,_=decom_vgg16()
    net.train()

    if cfg.use_cuda:
        device_id=cfg.device_id
        net.cuda(device_id)
    data_set=ExtractorDataset()
    data_loader=DataLoader(data_set,batch_size=1,shuffle=False,drop_last=False)

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
            fname='%s%d'% (cfg.feat_dir,num)
            content={'feat':feat[pos],
                'box':boxes[pos],
                'label':labels[pos],
                'scale':scale[pos],
                'img_size':img_size[pos]}
            torch.save(content,fname)


if __name__ == '__main__':
    main()