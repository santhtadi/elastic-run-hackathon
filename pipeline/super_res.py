import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore


class SuperRes:
    def __init__(self):
        self.net = IENetwork('./super_resolution_models/single-image-super-resolution-1033.xml',
                             './super_resolution_models/single-image-super-resolution-1033.bin')

    def increase_resolution(self, img: np.ndarray):
        inp_h, inp_w = img.shape[0], img.shape[1]
        out_h, out_w = inp_h * 3, inp_w * 3
        c1 = self.net.layers['79/Cast_11815_const']
        c1.blobs['custom'][4] = inp_h
        c1.blobs['custom'][5] = inp_w

        c2 = self.net.layers['86/Cast_11811_const']
        c2.blobs['custom'][2] = out_h
        c2.blobs['custom'][3] = out_w

        # Reshape network to specific size
        self.net.reshape({'0': [1, 3, inp_h, inp_w], '1': [1, 3, out_h, out_w]})

        # Load network to device
        ie = IECore()
        exec_net = ie.load_network(self.net, 'CPU')

        # Prepare input
        inp = img.transpose(2, 0, 1)  # interleaved to planar (HWC -> CHW)
        inp = inp.reshape(1, 3, inp_h, inp_w)
        inp = inp.astype(np.float32)

        # Prepare second input - bicubic resize of first input
        resized_img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        resized = resized_img.transpose(2, 0, 1)
        resized = resized.reshape(1, 3, out_h, out_w)
        resized = resized.astype(np.float32)

        outs = exec_net.infer({'0': inp, '1': resized})

        out = next(iter(outs.values()))

        out = out.reshape(3, out_h, out_w).transpose(1, 2, 0)
        out = np.clip(out * 255, 0, 255)
        out = np.ascontiguousarray(out).astype(np.uint8)

        return out


if __name__ == "__main__":
    # Read an image
    img = cv2.imread("./1148.jpeg")
    sr = SuperRes()
    sr.increase_resolution(img)
