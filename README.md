# Introduction
Hello,

This Repository contains our (Team Pixel) submission for #POSSIBILITIES Hackathon by ElasticRun powered by Microsoft and hosted by HackerEarth!

# Problem Statement
The problem statement that we selected was to identify the store size, product categories and products in an image.

# Our Solution / Submission

## Environment Setup

All the libraries that we used are listed in [environment_spec.txt](./environment_spec.txt)

## Step-1: Using Super Resolution
The first and foremost thing to do in any computer vision application is image processing

The problem with many images in the dataset is the image size and the lack of contrasting features

We used a super resolution network from [OpenVINO](https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/README.md)

The inference code can be found at [super_res.py](./pipeline/super_res.py)

## Step-2: Using OCR

We used OCR to try and detect the brand names or product names in the images

### Problems Faced
Since we used ocr, we are able to read not just brand names but everything that's not a product as well.

![image with too many detections](./readme_images/ocr.jpg)

To solve this, we used a sequence matcher that maps every word with product names in our dataset to eliminate words that are not related

The output images are saved to [saved_images](./readme_images)

## Approach-1: Using Selective Search

### Selective Search
Selective Search is a Region proposal network used in object detection. It was first published in 2012 and saying that it has revolutionalized the world of computer vision is putting it lightly.

### Problems Faced

The products are not made of one single color or texture as considered by SS algorithm. 

So we have to eliminate many Regions proposed by SS.

![image with too many proposals](./readme_images/RPN.jpg)

In order to do that, we used Non Max Suppression from OpenCV to remove boxes that are overlapping too much.

The inference code for region proposal can be found at [region_proposal.py](./pipeline/region_proposal.py)

## Approach-2: Using Object Detector

### Tiny Yolov3 trained on Our Images and SKU110K grocery dataset

We collected images from our local stores and included images from SKU110K to train a tiny-yolov3 model.

The model is only capable of identifying grocery products in a shelf.

### Problems Faced

The problem we faced is the model convergence, the data is very dynamic and we felt the large amount of data could take a while to train.

So we used a small subset of the SKU110K dataset and combined it with our dataset to train the model.

The inference code can be found at [yolo_detector.py](./pipeline/yolo_detector.py)





