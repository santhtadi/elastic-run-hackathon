import cv2
import random
import numpy as np
from keras.models import load_model


class SelectRegions:
    def __init__(self):
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self.count = 0
        self.model = load_model("./complete_incomplete_model")

    @staticmethod
    def preprocess_complete(img):
        img = cv2.resize(img, (224, 224))
        img = (img.astype(np.float32) / 127.0) - 1
        return img

    def get_regions(self, image_orig):
        image = cv2.resize(image_orig.copy(), (300, 300))
        H, W = image_orig.shape[:2]
        self.ss.setBaseImage(image)
        self.ss.switchToSelectiveSearchQuality()
        rects = self.ss.process()
        crops = []
        total = 0
        boxes = []
        for (x, y, w, h) in rects:
            if not (0.1*300 < w < 0.8*300 and 0.1*300 < h < 0.3*300):
                continue
            boxes.append([int(_) for _ in [x, y, w, h]])
        confidences = [1.0]*len(boxes)
        selected = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.5)
        output = image_orig.copy()
        out_boxes = []
        counter = 0
        for i in selected:
            cropsx = []
            # clone the original image so we can draw on it
            (x, y, w, h) = boxes[i]
            total += 1
            # draw the region proposal bounding box on the image
            x1, y1, x2, y2 = x, y, x + w, y + h
            x1, x2 = int(x1 * W / 300), int(x2 * W / 300)
            y1, y2 = int(y1 * H / 300), int(y2 * H / 300)
            crop = image_orig[y1:y2, x1:x2]
            counter += 1
            cropsx = self.preprocess_complete(crop)
            cropsx = np.expand_dims(cropsx, axis=0)
            preds = self.model.predict(cropsx)
            pred = np.argmax(preds)
            if pred == 0 and np.max(preds) > 0.4:
                crops.append(crop)
                out_boxes.append((x1, y1, x2, y2))
                self.count += 1
        return crops, out_boxes
