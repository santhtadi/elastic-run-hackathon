import os
import cv2
import numpy as np
import tensorflow
from scipy.spatial import distance
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess_input


class FeatureExtractor:
    def __init__(self, model: str = "mobilenet"):
        print("initializing feature extractor")
        self.model_type = model
        if model == "mobilenet":
            mobnet = MobileNet(weights="imagenet")
            out = tensorflow.keras.layers.Flatten()(mobnet.layers[-4].output)
            self.feature_extraction_model = tensorflow.keras.models.Model(inputs=mobnet.input,
                                                                          outputs=out)
        self.batch_size = 16

    @staticmethod
    def preprocess(img):
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def extract_features(self, images):
        all_features = []
        for x in range((len(images) // self.batch_size) + 2):
            inputs = []
            for i in range(x * self.batch_size, (x + 1) * self.batch_size):
                if i >= len(images):
                    break
                inputs.append(self.preprocess(images[i]))
            if len(inputs) == 0:
                break
            inputs = np.array(inputs)
            if self.model_type == "mobilenet":
                inputs = mobilenet_preprocess_input(inputs)
            features = self.feature_extraction_model.predict(inputs)
            all_features.extend(features)
        return np.array(all_features)


class FindProducts:
    def __init__(self):
        print("initializing find products")
        self.fe = FeatureExtractor()
        self.saved_features = []
        self.saved_names = []
        self.category = []
        self.prod_names = [w for w in os.listdir("./products")]
        for prod in self.prod_names:
            with open(f"./products/{prod}/cat.txt", 'r') as f:
                c = f.read().strip()
            images = [cv2.imread(f"./products/{prod}/{w}") for w in os.listdir(f"./products/{prod}") if
                      w.endswith(".jpg")]
            features = self.fe.extract_features(images)
            self.saved_features.extend(features)
            self.saved_names.extend([prod] * len(images))
            self.category.extend([c] * len(images))
        self.saved_features = np.array(self.saved_features)
        print(">>>>>>>>>>> saved_features: ", self.saved_features.shape)
        print(self.saved_names)

    def match_products(self, cropped_images):
        selected = []
        features = self.fe.extract_features(cropped_images)
        # can use faiss instead of this for faster matches
        for i, feat in enumerate(features):
            distances = []
            for saved_feat, saved_name, saved_cat in zip(self.saved_features, self.saved_names, self.category):
                distances.append((1 - distance.cosine(saved_feat, feat), i, saved_name, saved_cat))
            distances = sorted(distances, key=lambda x: x[0], reverse=True)
            if distances[0][0] > 0.4:
                selected.append(distances[0])
        return selected
