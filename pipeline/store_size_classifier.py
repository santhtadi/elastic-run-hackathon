import numpy as np
import cv2
from keras.models import load_model
from calculate_rows import GetRows


class StoreClassifier:
    def __init__(self):
        self.classifier = load_model("./size_classification_model")
        self.labels = ["small", "medium", "large"]
        self.map = {"small": 1, "medium": 2, "large": 3}
        self.gr = GetRows()

    def predict_store_size(self, image):
        image_orig = image.copy()
        image = np.array(cv2.resize(image, (224, 224)), dtype=np.float32) / 127.0 - 1
        image = np.expand_dims(image, axis=0)
        preds = self.classifier.predict(image)[0]
        pred = np.argmax(preds)
        conf = np.max(preds) * 100
        label = self.labels[pred]
        score = self.map[label] * 0.8 + self.gr.get_rows(image_orig) * 0.2
        print("store size score: ", score)
        if score < 1.5:
            return "small", score
        if score < 2.5:
            return "medium", score
        return "large", score
