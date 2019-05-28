import onnxruntime
import onnx
from onnx import shape_inference
import requests
from io import BytesIO
from PIL import Image
import numpy
#import onnx_helper


# remove floor for int type.
model = onnx.load('model.onnx')

def remove_unused_floor(model):
    #model = shape_inference.infer_shapes(model)
    nodes = model.graph.node

    floor_nodes = [node for node in nodes if node.op_type=='Floor']

    for f in floor_nodes:
        in_id = f.input[0]
        out_id = f.output[0]
        in_n = [node for node in nodes if node.output == [in_id]][0]
        if in_n.op_type == 'Mul':
            out_n = [node for node in nodes if node.input == [out_id]][0]
            out_n.input[0] = in_n.output[0]
            nodes.remove(f)

    onnx.save(model, 'updated_model.onnx')



remove_unused_floor(model)


sess = onnxruntime.InferenceSession('updated_model.onnx')

def fetch_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

pil_image = fetch_image(
    url="http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")
pil_image = fetch_image(
    url='http://farm4.staticflickr.com/3153/2970773875_164f0c0b83_z.jpg')

# convert to BGR format
image = numpy.array(pil_image)[:, :, [2, 1, 0]]

out = sess.run(None, {
    sess.get_inputs()[0].name: image
})

print(out)