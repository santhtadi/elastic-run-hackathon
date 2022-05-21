import numpy as np
import cv2
from keras.models import load_model


class StoreClassifier:
    def __init__(self):
        self.classifier = load_model("./size_classification_model")
        self.labels = ["small", "medium", "large"]

    def predict_store_size(self, image):
        image = np.array(cv2.resize(image, (224, 224)), dtype=np.float32) / 127.0 - 1
        image = np.expand_dims(image, axis=0)
        preds = self.classifier.predict(image)[0]
        pred = np.argmax(preds)
        conf = np.max(preds)*100
        return self.labels[pred], conf
