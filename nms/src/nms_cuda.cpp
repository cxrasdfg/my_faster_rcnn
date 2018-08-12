// ------------------------------------------------------------------
// Faster R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Shaoqing Ren
// ------------------------------------------------------------------
#include <ATen/ATen.h>
#include <THC/THC.h>
// #include <TH/TH.h>
#include <math.h>
#include <stdio.h>
#include "cuda/nms_kernel.h"
THCState *state = at::globalContext().thc_state;

#ifdef __cplusplus
extern "C"
#endif
int gpu_nms(THLongTensor * keep, THLongTensor* num_out, THCudaTensor * boxes, float nms_overlap_thresh) {
  printf("1. THCState * state: %p, THCudaTensor * boxes: %p\n",state,boxes);
  // boxes has to be sorted
  THArgCheck(THLongTensor_isContiguous(keep), 0, "boxes must be contiguous");
  THArgCheck(THCudaTensor_isContiguous(state, boxes), 2, "boxes must be contiguous");
  printf("check over...\n");
  // Number of ROIs
  printf("2. THCState * state: %p, THCudaTensor * boxes: %p\n",state,boxes);
  int boxes_num = THCudaTensor_size(state, boxes, 0);
  int boxes_dim = THCudaTensor_size(state, boxes, 1);

  printf("Number of ROIs %d, dim of boxes: %d\n",boxes_num,boxes_dim);
  float* boxes_flat = THCudaTensor_data(state, boxes);

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);
  THCudaLongTensor * mask = THCudaLongTensor_newWithSize2d(state, boxes_num, col_blocks);
  unsigned long long* mask_flat = (unsigned long long *)THCudaLongTensor_data(state, mask);

  printf("now begin to nms...\n");
  _nms(boxes_num, boxes_flat, mask_flat, nms_overlap_thresh);
  printf("function _nms is over...\n");

  THLongTensor * mask_cpu = THLongTensor_newWithSize2d(boxes_num, col_blocks);
  THLongTensor_copyCuda(state, mask_cpu, mask);
  THCudaLongTensor_free(state, mask);

  unsigned long long * mask_cpu_flat = (unsigned long long *) THLongTensor_data(mask_cpu);

  THLongTensor * remv_cpu = THLongTensor_newWithSize1d(col_blocks);
  unsigned long long* remv_cpu_flat = (unsigned long long *)THLongTensor_data(remv_cpu);
  THLongTensor_fill(remv_cpu, 0);

  long * keep_flat = (long*)THLongTensor_data(keep);
  long num_to_keep = 0;

  int i, j;
  for (i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv_cpu_flat[nblock] & (1ULL << inblock))) {
      keep_flat[num_to_keep++] = i;
      unsigned long long *p = &mask_cpu_flat[0] + i * col_blocks;
      for (j = nblock; j < col_blocks; j++) {
        remv_cpu_flat[j] |= p[j];
      }
    }
  }

  long * num_out_flat =(long*) THLongTensor_data(num_out);
  * num_out_flat = num_to_keep;

  THLongTensor_free(mask_cpu);
  THLongTensor_free(remv_cpu);

  return 1;
}
