"""
Verified against

    pytorch-nightly: 1.2.0.dev20190604-py3.7_cuda10.0.130_cudnn7.5.1_0 pytorch
    torchvision:     from source 04188377c54aa9073e4c2496ddd9996da9fda629
    onnx:            1.5.0
    onnxruntime:     0.4.0
"""
import onnxruntime

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def preprocess(image):
    # Resize
    ratio = 800.0 / min(image.size[0], image.size[1])
    image = image.resize((int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR)

    # Convert to BGR
    image = np.array(image)[:, :, [2, 1, 0]].astype('float32')

    # HWC -> CHW
    image = np.transpose(image, [2, 0, 1])

    # Normalize
    mean_vec = np.array([102.9801, 115.9465, 122.7717])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]

    # Pad to be divisible of 32
    import math
    padded_h = int(math.ceil(image.shape[1] / 32) * 32)
    padded_w = int(math.ceil(image.shape[2] / 32) * 32)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, :image.shape[1], :image.shape[2]] = image
    image = padded_image

    return image

classes = [line.rstrip('\n') for line in open('coco_classes.txt')]

session = onnxruntime.InferenceSession('mask_rcnn_R_50_FPN_1x.onnx')

img = Image.open('frcnn_demo.jpg')
# img = Image.open('/home/bowbao/repos/mlperf_inference/cloud/single_stage_detector/val2017/000000001000.jpg')

"""
Preprocessing
"""
img_data = preprocess(img)



"""
Inference
"""
boxes, labels, scores, masks = session.run(None, {
    session.get_inputs()[0].name: img_data
})

print(boxes.shape)
print(labels.shape)
print(scores.shape)
print(masks.shape)
"""
Postprocessing
"""
def display_objdetect_image(image, boxes, labels, scores, masks, score_threshold=0.7):
    # Resize boxes
    ratio = 800.0 / min(image.size[0], image.size[1])
    boxes /= ratio
    #masks /= ratio

    _, ax = plt.subplots(1, figsize=(12,9))

    import pycocotools.mask as mask_util
    import cv2
    from maskrcnn_benchmark.utils import cv2_util

    image = np.array(image)#.astype('float32')


    for mask, box, label, score in zip(masks, boxes, labels, scores):
        if score <= score_threshold:
            break
        mask = mask[0, :, :, None]

        mask = cv2.resize(mask, (int(box[2])-int(box[0])+1, int(box[3])-int(box[1])+1))

        mask = mask > 0.5

        im_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        x_0 = max(int(box[0]), 0)
        x_1 = min(int(box[2]) + 1, image.shape[1])
        y_0 = max(int(box[1]), 0)
        y_1 = min(int(box[3]) + 1, image.shape[0])

        try:
            im_mask[int(y_0):int(y_1), int(x_0):int(x_1)] = mask[
                (y_0 - int(box[1])) : (y_1 - int(box[1])), (x_0 - int(box[0])) : (x_1 - int(box[0]))
            ]
        except Exception as e:
            print(e)

        im_mask = im_mask[:, :, None]
        contours, hierarchy = cv2_util.findContours(
            im_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        image = cv2.drawContours(image, contours, -1, 25, 3)

    ax.imshow(image)

    # Showing boxes with score > 0.7
    for box, label, score in zip(boxes, labels, scores):
        if score > score_threshold:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='b', facecolor='none')
            ax.annotate(classes[label] + ':' + str(np.round(score, 2)), (box[0], box[1]), color='w', fontsize=12)
            ax.add_patch(rect)
    plt.show()

display_objdetect_image(img, boxes, labels, scores, masks)
