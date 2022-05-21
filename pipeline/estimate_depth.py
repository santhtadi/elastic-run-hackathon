import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import matplotlib.pyplot as plt


class DepthEst:
    def __init__(self):
        self.net = IENetwork('./midas_net_fp32/midasnet.xml', './midas_net_fp32/midasnet.bin')
        ie = IECore()
        self.exec_net = ie.load_network(self.net, 'CPU')

    def get_depth(self, img_orig: np.ndarray, mask_out_path: str):
        # Prepare input
        img = cv2.resize(img_orig.copy(), (384, 384))  # - [123.675, 116.28, 103.53])/[51.525, 50.4, 50.625]
        inp = img.transpose(2, 0, 1)  # interleaved to planar (HWC -> CHW)
        inp = inp.reshape(1, 3, 384, 384)
        inp = inp.astype(np.float32)
        outs = self.exec_net.infer({'image': inp})
        out = next(iter(outs.values()))
        out = out.reshape(1, 384, 384).transpose(1, 2, 0)
        ok = np.max(out) - out
        mask = ok > 3500
        mask = np.array(mask * 255, dtype=np.uint8)
        H, W = img_orig.shape[:2]
        mask = cv2.resize(mask, (W, H))
        masked_image = cv2.bitwise_and(img_orig, img_orig, mask=mask)
        combined = np.hstack([masked_image, img_orig])
        cv2.imwrite(mask_out_path, combined)
        # plt.imshow(ok, cmap='autumn', interpolation='nearest')
        # plt.title("inv Heat Map")
        # plt.savefig(out_path)
        # plt.show()
        # out = np.clip(out * 255, 0, 255)
        # print(out.shape)
        return masked_image


if __name__ == "__main__":
    # Read an image
    img = cv2.imread("./77a4712a-IMG_20220407_111359.jpg")
    de = DepthEst()
    de.get_depth(img, "./masked_img.jpg")
