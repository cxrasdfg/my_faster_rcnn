# my_faster_rcnn

Simple and short codes with pytorch, just for me to understand the procedure of faster r-cnn, ~~could run on windows~~ but the training procedure does not work well, to be continued...

____
## TODO

- [X] Roughly Implementation :
    - [X] just put every thing in one file
    - [X] train basicly
    - [X] transform all the numpy codes to torch codes

- [-] Use SDB Tool
    - [-] use ChainerCV
    - [-] add data augmentation

- [-] Use Pretrained Caffe Model
    - [-] caffe normalization
    - [-] bbox transformation normalization

- [-] Add Evaluation
    - [-] use the pretrained caffe model(VGG16)
    - [-] catch up the paper's performance 

- [-] Code Refactor
    - [-] file categories
    - [-] deprecated code removal

## References:
+ [Faster R-CNN(Paper)](https://arxiv.org/abs/1506.01497)
+ [chenyuntc/simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)
+ [longcw/faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch)