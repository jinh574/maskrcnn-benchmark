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

import pytorch_export_patch

if __name__ == "__main__":
    # load config from file and command-line arguments
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg.merge_from_file(
        os.path.join(project_dir,
                     "configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"))
                    # "configs/caffe2/e2e_faster_rcnn_R_50_FPN_1x_caffe2.yaml"))
    # cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=0.7,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=480,
    )

    # prepare for onnx export
    coco_demo.model.prepare_onnx_export()


def single_image_to_top_predictions(image):
    # attempt to change shape from constant to tensor
    # from torch.onnx import operators
    # im_shape = operators.shape_as_tensor(image)
    # image_sizes = (im_shape[1].to(torch.float), im_shape[2].to(torch.float))

    # image_list = ImageList(image.unsqueeze(0), [image_sizes])
    image_list = ImageList(image.unsqueeze(0), [(int(image.size(-2)), int(image.size(-1)))])

    for param in coco_demo.model.parameters():
        param.requires_grad = False

    result, = coco_demo.model(image_list)
    scores = result.get_field("scores")
    masks = result.get_field("mask")
    # result = (result.bbox, result.get_field('labels'), scores)
    result = (result.bbox, result.get_field('labels'), scores, masks)
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

    """
    Preprocessing image.
    """
    # convert to BGR format
    image = torch.from_numpy(numpy.array(pil_image)[:, :, [2, 1, 0]])
    original_image = image
    image = torch.nn.functional.upsample(image.permute(2, 0, 1).unsqueeze(0).to(torch.float), size=(960, 1280)).to(torch.uint8).squeeze(0).permute(1, 2, 0).to('cuda')

    image = image.permute(2, 0, 1)

    if not cfg.INPUT.TO_BGR255:
        image = image.float() / 255.0
        image = image[[2, 1, 0]]
    else:
        image = image.float()

    # we absolutely want fixed size (int) here (or we run into a tracing error (or bug?)
    # or we might later decide to make things work with variable size...
    image = image - torch.tensor(cfg.INPUT.PIXEL_MEAN)[:, None, None].to('cuda')

    # model_path = 'faster_rcnn_R_50_FPN_1x.onnx'
    model_path = 'mask_rcnn_R_50_FPN_1x.onnx'

    with torch.no_grad():
        model = FRCNNModel()
        model.eval()
        model.cuda()

        torch.onnx.export(model, (image,), model_path, verbose=True, opset_version=10, strip_doc_string=True, do_constant_folding=True)

        pytorch_export_patch.postprocess_model(model_path)

        boxes, labels, scores, masks = model(image)

        print(masks.shape)

        """
        profiling
        """

        pil_image = fetch_image(
            url='http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg')
        image2 = torch.from_numpy(numpy.array(pil_image)[:, :, [2, 1, 0]])
        image = image2
        image = torch.nn.functional.upsample(image.permute(2, 0, 1).unsqueeze(0).to(torch.float), size=(960, 1280)).squeeze(0)#.permute(1, 2, 0)
        image = image.cuda()
        repetition = 2
        out = model(image) # image or image2
        times = []
        for _ in range(repetition):
            begin = timeit.time.perf_counter()
            out = model(image)
            times.append(timeit.time.perf_counter() - begin)
        print("Avg run time of pytorch model:", sum(times)/len(times))

        image = image.to('cpu')

        # Convert the Numpy array to a TensorProto
        from onnx import numpy_helper
        tensor = numpy_helper.from_array(image.numpy())
        with open('input_0.pb', 'wb') as f:
            f.write(tensor.SerializeToString())

        import onnxruntime as rt
        sess = rt.InferenceSession(model_path)
        ort_out = sess.run(None, {sess.get_inputs()[0].name: image.numpy()})
        # print('torch output:', out)
        # print('ort output:', ort_out)
        # print(image.shape)
        times = []
        for _ in range(repetition):
            begin = timeit.time.perf_counter()
            ort_out = sess.run(None, {sess.get_inputs()[0].name: image.numpy()})
            times.append(timeit.time.perf_counter() - begin)
        print("Avg run time of onnx model:", sum(times)/len(times))
