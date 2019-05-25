import os
import numpy

from io import BytesIO

import requests
import torch

from PIL import Image
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from maskrcnn_benchmark.structures.image_list import ImageList

if __name__ == "__main__":
    # load config from file and command-line arguments

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg.merge_from_file(
        os.path.join(project_dir,
                     "configs/caffe2/e2e_faster_rcnn_R_50_FPN_1x_caffe2.yaml"))
    cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=0.7,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=480,
    )


def single_image_to_top_predictions(image):
    image = image.float() / 255.0
    image = image.permute(2, 0, 1)
    # we are loading images with OpenCV, so we don't need to convert them
    # to BGR, they are already! So all we need to do is to normalize
    # by 255 if we want to convert to BGR255 format, or flip the channels
    # if we want it to be in RGB in [0-1] range.
    if cfg.INPUT.TO_BGR255:
        image = image * 255
    else:
        image = image[[2, 1, 0]]

    # we absolutely want fixed size (int) here (or we run into a tracing error (or bug?)
    # or we might later decide to make things work with variable size...
    image = image - torch.tensor(cfg.INPUT.PIXEL_MEAN)[:, None, None]
    # should also do variance...
    image_list = ImageList(image.unsqueeze(0), [(int(image.size(-2)), int(image.size(-1)))])

    for param in coco_demo.model.parameters():
        param.requires_grad = False

    result, = coco_demo.model(image_list)
    scores = result.get_field("scores")
    keep = (scores >= coco_demo.confidence_threshold)
    result = (result.bbox[keep],
              result.get_field("labels")[keep],
              scores[keep])
    return result

class FRCNNModel(torch.nn.Module):
    def forward(self, image):
        return single_image_to_top_predictions(image)


def fetch_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

if __name__ == '__main__':
    pil_image = fetch_image(
        url="http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")

    # convert to BGR format
    image = torch.from_numpy(numpy.array(pil_image)[:, :, [2, 1, 0]])
    original_image = image

    if coco_demo.cfg.DATALOADER.SIZE_DIVISIBILITY:
        assert (image.size(0) % coco_demo.cfg.DATALOADER.SIZE_DIVISIBILITY == 0
                and image.size(1) % coco_demo.cfg.DATALOADER.SIZE_DIVISIBILITY == 0)

    with torch.no_grad():
        model = FRCNNModel()
        model.eval()

        def register_custom_op():
            # experimenting custom op registration.
            from torch.onnx.symbolic_helper import parse_args
            from torch.onnx.symbolic_opset9 import select, unsqueeze, squeeze, _cast_Long
            @parse_args('v', 'v', 'f')
            def symbolic_multi_label_nms(g, boxes, scores, iou_threshold):#, max_output_per_class, iou_threshold, score_threshold):
                boxes = unsqueeze(g, boxes, 0)
                scores = unsqueeze(g, unsqueeze(g, scores, 0), 0)
                max_output_per_class = g.op('Constant', value_t=torch.tensor([2000], dtype=torch.long))
                iou_threshold = g.op('Constant', value_t=torch.tensor([iou_threshold], dtype=torch.float))
                nms_out = g.op('NonMaxSuppression', boxes, scores, max_output_per_class, iou_threshold)
                return squeeze(g, select(g, nms_out, 1, g.op('Constant', value_t=torch.tensor([2], dtype=torch.long))), 1)

            @parse_args('v', 'v', 'f', 'i', 'i', 'i')
            def symbolic_roi_align(g, input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio):
                batch_indices = _cast_Long(g, squeeze(g, select(g, rois, 1, g.op('Constant', value_t=torch.tensor([0], dtype=torch.long))), 1), False)
                rois = select(g, rois, 1, g.op('Constant', value_t=torch.tensor([1,2,3,4], dtype=torch.long)))
                return g.op('RoiAlign', input, rois, batch_indices, spatial_scale_f=spatial_scale, output_height_i=pooled_height, output_width_i=pooled_width, sampling_ratio_i=sampling_ratio)

            from torch.onnx import register_custom_op_symbolic
            register_custom_op_symbolic('maskrcnn_benchmark::nms', symbolic_multi_label_nms, 10)
            register_custom_op_symbolic('maskrcnn_benchmark::roi_align_forward', symbolic_roi_align, 10)

        register_custom_op()

        torch.onnx.export(model, (image,), 'model.onnx', verbose=True, opset_version=10, strip_doc_string=False)
