import importlib.util
import os.path
import sys
import time
import cv2
from difflib import SequenceMatcher

# comment out the below lines if using the pipeline directly

# Super Resolution
spec = importlib.util.spec_from_file_location("super_res", "../../pipeline/super_res.py")
super_res = importlib.util.module_from_spec(spec)
sys.modules["super_res"] = super_res
spec.loader.exec_module(super_res)
# Paddele OCR
spec = importlib.util.spec_from_file_location("ocr_detector", "../../pipeline/ocr_detect.py")
ocr_detect = importlib.util.module_from_spec(spec)
sys.modules["ocr_detect"] = ocr_detect
spec.loader.exec_module(ocr_detect)
# region proposal
spec = importlib.util.spec_from_file_location("region_proposal", "../../pipeline/region_proposal.py")
region_proposal = importlib.util.module_from_spec(spec)
sys.modules["region_proposal"] = region_proposal
spec.loader.exec_module(region_proposal)
# depth estimation
spec = importlib.util.spec_from_file_location("estimate_depth", "../../pipeline/estimate_depth.py")
estimate_depth = importlib.util.module_from_spec(spec)
sys.modules["estimate_depth"] = estimate_depth
spec.loader.exec_module(estimate_depth)
# embedding product matcher
spec = importlib.util.spec_from_file_location("embedding_product_matcher", "../../pipeline/embedding_product_matcher.py")
embedding_product_matcher = importlib.util.module_from_spec(spec)
sys.modules["embedding_product_matcher"] = embedding_product_matcher
spec.loader.exec_module(embedding_product_matcher)
# store size classification
spec = importlib.util.spec_from_file_location("store_size_classifier", "../../pipeline/store_size_classifier.py")
store_size_classifier = importlib.util.module_from_spec(spec)
sys.modules["store_size_classifier"] = store_size_classifier
spec.loader.exec_module(store_size_classifier)
# yolo detector
spec = importlib.util.spec_from_file_location("yolo_detector", "../../pipeline/yolo_detector.py")
yolo_detector = importlib.util.module_from_spec(spec)
sys.modules["yolo_detector"] = yolo_detector
spec.loader.exec_module(yolo_detector)
# -------------------------------------- #

from super_res import SuperRes
from ocr_detect import OCRDetect
from region_proposal import SelectRegions
from estimate_depth import DepthEst
from embedding_product_matcher import FindProducts
from store_size_classifier import StoreClassifier
from yolo_detector import YoloDetector


class PipelineX:
    def __init__(self):
        self.sr = SuperRes()
        print("pipeline initialized!")
        self.name = time.time()
        self.ocr = OCRDetect()
        self.ocr_image = ""
        self.rp = SelectRegions()
        self.product_names = [w for w in os.listdir("./products")]
        self.categories = []
        for prod in self.product_names:
            with open(f"./products/{prod}/cat.txt", 'r') as f:
                self.categories.append(f.read().strip())
        self.product_name_thresh = 0.4
        self.de = DepthEst()
        self.embed_batch_size = 16
        self.find_products = FindProducts()
        self.store_size = StoreClassifier()
        self.yolo_d = YoloDetector("./yolo_support/yolov3-tiny-obj.cfg",
                                   "./yolo_support/yolov3-tiny-obj.weights",
                                   classes=["product"])

    @staticmethod
    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    def process_image(self, image):
        self.name = time.time()
        ocr_products = []
        if not os.path.exists(f"./saved_images/{self.name}"):
            os.mkdir(f"./saved_images/{self.name}")
        cv2.imwrite(f"./saved_images/{self.name}/{self.name}.jpg", image)
        self.ocr_image = f"./saved_images/{self.name}/{self.name}.jpg"
        H, W = image.shape[:2]
        # SUPER RESOLUTION
        if H < 400 or W < 400:
            image = self.sr.increase_resolution(image)
            cv2.imwrite(f"./saved_images/{self.name}/SR_{self.name}.jpg", image)
            self.ocr_image = f"./saved_images/{self.name}/SR_{self.name}.jpg"
        # STORE SIZE CLASSIFICATION
        store_size = self.store_size.predict_store_size(image)
        # APPROACH-0: OCR
        boxes, txts, scores = self.ocr.identify_products(self.ocr_image,
                                                         f"./saved_images/{self.name}/ocr_{self.name}.jpg")
        # deleting unwanted texts
        for txt, conf in zip(txts, scores):
            for name, cat in zip(self.product_names, self.categories):
                if self.similar(txt, name) > self.product_name_thresh or conf > 0.7:
                    ocr_products.append((name, cat))
        ocr_products = list(set(ocr_products))
        # APPROACH-1: REGION PROPOSAL
        print("calculating regions.. ")
        crops, bboxes = self.rp.get_regions(image)
        # condition for depth estimation usage
        if len(crops) > 200:
            image = self.de.get_depth(image, f"./saved_images/{self.name}/depth_{self.name}.jpg")
            crops, bboxes = self.rp.get_regions(image)
        selected = self.find_products.match_products(crops)
        print(">>>>>>>>>> region proposal output: ", selected)
        # drawing region proposal output
        rpn_products = []
        region_out = image.copy()
        for s in selected:
            conf, ind, product, cat = s
            rpn_products.append((product, cat))
            x1, y1, x2, y2 = bboxes[ind]
            cv2.rectangle(region_out, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
            cv2.putText(region_out, f"conf: {conf:.2f} prod: {product}, cat: {cat}",
                        (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), thickness=2)
        cv2.imwrite(f"./saved_images/{self.name}/approach_1_{self.name}.jpg", region_out)
        print("RPN completed")
        # APPROACH-2: Object Detection Model
        print("detecting regions using yolo.. ")
        _, yolo_out = self.yolo_d.detect(image, conf=0.1, nms_thresh=0.4)
        # condition for depth estimation usage
        crops2 = []
        for cls, val in yolo_out.items():
            for (box, conf) in val:
                x1, y1, x2, y2 = box
                crop = image[y1:y2, x1:x2]
                h, w = crop.shape[:2]
                if h < 10 or w < 10:
                    print("error: ", box)
                    continue
                crops2.append(crop)
        selected2 = self.find_products.match_products(crops2)
        print(">>>>>>>>>> object detection output: ", selected2)
        # drawing region proposal output
        det_products = []
        det_out = image.copy()
        for s in selected2:
            conf, ind, product, cat = s
            conf_yolo = yolo_out["product"][ind][1]
            det_products.append((product, cat))
            x1, y1, x2, y2 = yolo_out["product"][ind][0]
            cv2.rectangle(det_out, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
            cv2.putText(det_out, f"yolo_conf: {conf_yolo:.2f} conf: {conf:.2f} prod: {product}, cat: {cat}",
                        (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), thickness=2)
        cv2.imwrite(f"./saved_images/{self.name}/approach_2_{self.name}.jpg", det_out)
        print("Object Detection completed")
        output = {"approach-2": {'image': 'base-64',
                                 'ocr': ocr_products,
                                 'store_size': store_size,
                                 'yolo': list(set(det_products))},
                  "approach-1": {'image': 'base-64',
                                 'ocr': ocr_products,
                                 'store_size': store_size,
                                 'rpn': list(set(rpn_products))}}
        print("FINAL OUTPUT: ", output)
        return output


if __name__ == "__main__":
    img = cv2.imread("./sample_image.jpg")
    pipeline = PipelineX()
    outt = pipeline.process_image(img)
    print(outt)
