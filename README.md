# my_faster_rcnn

Simple and short codes with pytorch, just for me to understand the procedure of faster r-cnn, ~~could run on windows~~ but the training procedure does not work well, to be continued...
____
## Detection Example
![Image text](res.png)

____
## Performance
I have tried my best to adjust the hyper-param and train for a lot of times..., but I only get 0.584 map 
on VOC 2007 test dataset..., so sad... 
____
## TODO

- [X] Roughly Implementation :
    - [X] just put every thing in one file
    - [X] train basicly
    - [X] transform all the numpy codes to torch codes

- [X] Use SDB Tool
    - [X] use ChainerCV
    - [X] add data augmentation

- [X] Use Pretrained Caffe Model
    - [X] caffe normalization
    - [X] bbox transformation normalization

- [-] Add Evaluation
    - [X] use the pretrained caffe model(VGG16)
    - [-] ~~catch up the paper's performance~~(Failed...Give up) 

- [X] Code Refactor
    - [X] file categories
    - [X] deprecated code removal

## References:
+ [Faster R-CNN(Paper)](https://arxiv.org/abs/1506.01497)
+ [chenyuntc/simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)
+ [longcw/faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch)