import onnxruntime
import onnx
import onnx_helper


sess = onnxruntime.InferenceSession('model.onnx')

pil_image = fetch_image(
    url="http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")

# convert to BGR format
image = numpy.array(pil_image)[:, :, [2, 1, 0]]

out = sess.run(None, {
    sess.get_inputs()[0].name: image
})

print(out)