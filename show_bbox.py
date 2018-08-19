#coding=UTF-8

import cv2

_color_table={'gt':(255,0,0),'roi':(0,255,0),'pred':(0,0,255)}

def draw_bbox(img,bbox,color='gt'):
    assert color in ['gt','roi','pred']
    if len(list(bbox.shape))==1:
        bbox=bbox[None]
    assert bbox.shape[1]==4
    for x1,y1,x2,y2 in bbox:
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, _color_table[color],2, 1)
    

def show_img(img,wait_time=1):
    cv2.imshow("Testing", img)
    # Exit if ESC pressed
    tk = cv2.waitKey(wait_time) & 0xff
    if tk == 27: 
        return 

def tick_show(img,bbox,color='gt'):
    assert color in ['gt','roi','pred']
    if len(list(bbox.shape))==1:
        bbox=bbox[None]
    assert bbox.shape[1]==4
    for x1,y1,x2,y2 in bbox:
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, _color_table[color],2, 1)
        cv2.imshow("Testing", img)
        tk = cv2.waitKey(-1) & 0xff
        if tk == 27: 
            continue 
        