// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include <torch/script.h>


at::Tensor ROIAlign_forward_cpu(const at::Tensor& input,
                                const at::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width,
                                const int sampling_ratio);


at::Tensor nms_cpu(const at::Tensor& dets,
                   const at::Tensor& scores,
                   const float threshold);

at::Tensor multi_label_nms_cpu(const at::Tensor& boxes,
                               const at::Tensor& scores,
                               const at::Tensor& max_output_boxes_per_class,
                               const at::Tensor& iou_threshold,
                               const at::Tensor& score_threshold);