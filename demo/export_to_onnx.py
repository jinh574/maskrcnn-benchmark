import os
import numpy

from io import BytesIO

from matplotlib import pyplot

import requests
import torch
import timeit

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
    # image = image.permute(2, 0, 1)
    # # we are loading images with OpenCV, so we don't need to convert them
    # # to BGR, they are already! So all we need to do is to normalize
    # # by 255 if we want to convert to BGR255 format, or flip the channels
    # # if we want it to be in RGB in [0-1] range.
    # if not cfg.INPUT.TO_BGR255:
    #     # image = image.float() * 255
    #     # else:
    #     image = image.float() / 255.0
    #     image = image[[2, 1, 0]]
    # else:
    #     image = image.float()

    # # we absolutely want fixed size (int) here (or we run into a tracing error (or bug?)
    # # or we might later decide to make things work with variable size...
    # image = image - torch.tensor(cfg.INPUT.PIXEL_MEAN)[:, None, None]
    # should also do variance...
    image_list = ImageList(image.unsqueeze(0), [(int(image.size(-2)), int(image.size(-1)))])

    for param in coco_demo.model.parameters():
        param.requires_grad = False

    result, = coco_demo.model(image_list)
    scores = result.get_field("scores")
    # keep = (scores >= coco_demo.confidence_threshold)
    # keep = (scores >= 0.1)
    # result = (result.bbox[keep],
    #           result.get_field("labels")[keep],
    #           scores[keep])
    # NOTE: if to keep all
    result = (result.bbox, result.get_field('labels'), scores)
    return result

class FRCNNModel(torch.nn.Module):
    def forward(self, image):
        return single_image_to_top_predictions(image)

@torch.jit.script
def my_paste_mask(mask, bbox, height, width, threshold=0.5, padding=1, contour=True, rectangle=False):
    # type: (Tensor, Tensor, int, int, float, int, bool, bool) -> Tensor
    #padded_mask = torch.constant_pad_nd(mask, (padding, padding, padding, padding))
    #scale = 1.0 + 2.0 * float(padding) / float(mask.size(-1))
    # center_x = (bbox[2] + bbox[0]) * 0.5
    # center_y = (bbox[3] + bbox[1]) * 0.5
    # w_2 = (bbox[2] - bbox[0]) * 0.5 * scale
    # h_2 = (bbox[3] - bbox[1]) * 0.5 * scale  # should have two scales?
    # bbox_scaled = torch.stack([center_x - w_2, center_y - h_2,
    #                            center_x + w_2, center_y + h_2], 0)

    # TO_REMOVE = 1
    # w = (bbox_scaled[2] - bbox_scaled[0] + TO_REMOVE).clamp(min=1).long()
    # h = (bbox_scaled[3] - bbox_scaled[1] + TO_REMOVE).clamp(min=1).long()

    # scaled_mask = torch.ops.maskrcnn_benchmark.upsample_bilinear(padded_mask.float(), h, w)

    # x0 = bbox_scaled[0].long()
    # y0 = bbox_scaled[1].long()
    # x = x0.clamp(min=0)
    # y = y0.clamp(min=0)
    # leftcrop = x - x0
    # topcrop = y - y0
    # w = torch.min(w - leftcrop, width - x)
    # h = torch.min(h - topcrop, height - y)

    # # mask = torch.zeros((height, width), dtype=torch.uint8)
    # # mask[y:y + h, x:x + w] = (scaled_mask[topcrop:topcrop + h,  leftcrop:leftcrop + w] > threshold)
    # mask = torch.constant_pad_nd((scaled_mask[topcrop:topcrop + h, leftcrop:leftcrop + w] > threshold),
    #                              (int(x), int(width - x - w), int(y), int(height - y - h)))   # int for the script compiler

    if contour:
        mask = mask.float()
        # poor person's contour finding by comparing to smoothed
        mask = (mask - torch.nn.functional.conv2d(mask.unsqueeze(0).unsqueeze(0),
                                                  torch.full((1, 1, 3, 3), 1.0 / 9.0), padding=1)[0, 0]).abs() > 0.001
    if rectangle:
        x = torch.arange(width, dtype=torch.long).unsqueeze(0)
        y = torch.arange(height, dtype=torch.long).unsqueeze(1)
        r = bbox.long()
        # work around script not liking bitwise ops
        rectangle_mask = ((((x == r[0]) + (x == r[2])) * (y >= r[1]) * (y <= r[3]))
                          + (((y == r[1]) + (y == r[3])) * (x >= r[0]) * (x <= r[2])))
        print('box', r)
        #print('rec mask', rectangle_mask)
        mask = rectangle_mask.clamp(max=1)
    return mask


# @torch.jit.script
# def add_annotations(image, labels, scores, bboxes, class_names=','.join(coco_demo.CATEGORIES), color=torch.tensor([255, 255, 255], dtype=torch.long)):
#     # type: (Tensor, Tensor, Tensor, Tensor, str, Tensor) -> Tensor
#     result_image = torch.ops.maskrcnn_benchmark.add_annotations(image, labels, scores, bboxes, class_names, color)
#     return result_image


@torch.jit.script
def combine_masks(image, labels, masks, scores, bboxes, threshold=0.5, padding=1, contour=False, rectangle=True, palette=torch.tensor([33554431, 32767, 2097151])):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, int, bool, bool, Tensor) -> Tensor
    height = image.size(0)
    width = image.size(1)
    image_with_mask = image.clone()
    for i in range(bboxes.size(0)):
        color = ((palette * labels[i]) % 255).to(torch.uint8)
        one_mask = my_paste_mask(bboxes[i], bboxes[i], height, width, threshold, padding, contour, rectangle)
        image_with_mask = torch.where(one_mask.unsqueeze(-1), color.unsqueeze(0).unsqueeze(0), image_with_mask)
    # image_with_mask = add_annotations(image_with_mask, labels, scores, bboxes)
    return image_with_mask


def fetch_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

if __name__ == '__main__':
    pil_image = fetch_image(
        url="http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")

    # convert to BGR format
    image = torch.from_numpy(numpy.array(pil_image)[:, :, [2, 1, 0]])
    original_image = image
    image = torch.nn.functional.upsample(image.permute(2, 0, 1).unsqueeze(0).to(torch.float), size=(960, 1280)).to(torch.uint8).squeeze(0).permute(1, 2, 0).to('cpu')#.to('cpu')

    image = image.permute(2, 0, 1)
    # we are loading images with OpenCV, so we don't need to convert them
    # to BGR, they are already! So all we need to do is to normalize
    # by 255 if we want to convert to BGR255 format, or flip the channels
    # if we want it to be in RGB in [0-1] range.
    if not cfg.INPUT.TO_BGR255:
        # image = image.float() * 255
        # else:
        image = image.float() / 255.0
        image = image[[2, 1, 0]]
    else:
        image = image.float()

    # we absolutely want fixed size (int) here (or we run into a tracing error (or bug?)
    # or we might later decide to make things work with variable size...
    image = image - torch.tensor(cfg.INPUT.PIXEL_MEAN)[:, None, None].to('cpu')

    # if coco_demo.cfg.DATALOADER.SIZE_DIVISIBILITY:
    #     assert (image.size(0) % coco_demo.cfg.DATALOADER.SIZE_DIVISIBILITY == 0
    #             and image.size(1) % coco_demo.cfg.DATALOADER.SIZE_DIVISIBILITY == 0)

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


        torch.onnx.export(model, (image,), 'model.onnx', verbose=True, opset_version=10, strip_doc_string=True, do_constant_folding=True)

        import onnx
        onnx_model = onnx.load('model.onnx')

        def update_inputs_outputs_dims(model, input_dims, output_dims):
            """
                This function updates the sizes of dimensions of the model's inputs and outputs to the values
                provided in input_dims and output_dims. if the dim value provided is negative, a unique dim_param
                will be set for that dimension.
            """
            def update_dim(tensor, dim, i, j, dim_param_prefix):
                dim_proto = tensor.type.tensor_type.shape.dim[j]
                if isinstance(dim, int):
                    if dim >= 0:
                        dim_proto.dim_value = dim
                    else:
                        dim_proto.dim_param = dim_param_prefix + str(i) + '_' + str(j)
                elif isinstance(dim, str):
                    dim_proto.dim_param = dim
                else:
                    raise ValueError('Only int or str is accepted as dimension value, incorrect type: {}'.format(type(dim)))

            for i, input_dim_arr in enumerate(input_dims):
                for j, dim in enumerate(input_dim_arr):
                    update_dim(model.graph.input[i], dim, i, j, 'in_')

            for i, output_dim_arr in enumerate(output_dims):
                for j, dim in enumerate(output_dim_arr):
                    update_dim(model.graph.output[i], dim, i, j, 'out_')

            onnx.checker.check_model(model)
            return model

        def remove_unused_floor(model):
            #model = shape_inference.infer_shapes(model)
            nodes = model.graph.node

            for i, n in enumerate(nodes):
                n.name = str(i)

            floor_nodes = [node for node in nodes if node.op_type=='Floor']

            for f in floor_nodes:
                in_id = f.input[0]
                out_id = f.output[0]
                in_n = [node for node in nodes if node.output == [in_id]][0]
                if in_n.op_type == 'Mul':
                    out_n = [node for node in nodes if node.input == [out_id]][0]
                    out_n.input[0] = in_n.output[0]
                    nodes.remove(f)

            return model
        onnx_model = remove_unused_floor(onnx_model)
        onnx_model = update_inputs_outputs_dims(onnx_model, [[3, 'height', 'width']], [['nbox', 4], ['nbox'], ['nbox']])
        onnx.save(onnx_model, 'updated_model.onnx')


        # pil_image = fetch_image(
        #     url='http://farm4.staticflickr.com/3153/2970773875_164f0c0b83_z.jpg')
        # image2 = torch.from_numpy(numpy.array(pil_image)[:, :, [2, 1, 0]])
        # image = image2
        # image = torch.nn.functional.upsample(image.permute(2, 0, 1).unsqueeze(0).to(torch.float), size=(960, 1280)).to(torch.uint8).squeeze(0).permute(1, 2, 0)

        """
        profiling
        """
        # repetition = 20
        # out = model(image) # image or image2
        # times = []
        # for _ in range(repetition):
        #     begin = timeit.time.perf_counter()
        #     out = model(image)
        #     times.append(timeit.time.perf_counter() - begin)
        # print("Avg run time of pytorch model:", sum(times)/len(times))

        # image = image.to('cpu')

        # # Convert the Numpy array to a TensorProto
        # from onnx import numpy_helper
        # tensor = numpy_helper.from_array(image.numpy())
        # with open('input_0.pb', 'wb') as f:
        #     f.write(tensor.SerializeToString())

        # import onnxruntime as rt
        # sess = rt.InferenceSession('updated_model.onnx')
        # ort_out = sess.run(None, {sess.get_inputs()[0].name: image.numpy()})
        # # print('torch output:', out)
        # # print('ort output:', ort_out)
        # # print(image.shape)
        # times = []
        # for _ in range(repetition):
        #     begin = timeit.time.perf_counter()
        #     ort_out = sess.run(None, {sess.get_inputs()[0].name: image.numpy()})
        #     times.append(timeit.time.perf_counter() - begin)
        # print("Avg run time of onnx model:", sum(times)/len(times))