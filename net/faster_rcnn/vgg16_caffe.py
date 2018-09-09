# coding=UTF-8

import torch
from torchvision.models import vgg16
from config import cfg

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
    if cfg.freeze_top:    
        for layer in features[:10]:
            for p in layer.parameters():
                p.requires_grad = False

    return torch.nn.Sequential(*features), classifier