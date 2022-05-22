import json
import threading
import time

from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import numpy as np
from PIL import Image
import io
from ElasticLogger import ELogger

elogger = ELogger("./logs/log.txt")
# importing pipeline
import sys, cv2

thread_running = False
display_output = {}
a1_img = cv2.imencode(".png", cv2.imread("./sample_image.jpg"))
a2_img = cv2.imencode(".png", cv2.imread("./sample_image.jpg"))
outdict = {"approach-2": {'image': a2_img,
                          'ocr': ['ocr_products_a2'],
                          'store_size': ("large2", 0.9),
                          'yolo': list({"det_products2"})},
           "approach-1": {'image': a1_img,
                          'ocr': ['ocr_products_a1'],
                          'store_size': ("large1", 0.9),
                          'yolo': list({"rpn_products1"})}}
sys.path.append("../../pipeline")


def convert_format(data):
    out = {
        'approach-1': {
            'image': str(data['approach-1']['image']),
            'ocr': [{k: v} for k, v in data['approach-1']['ocr']],
            'store_size': {data['approach-1']['store_size'][0]: data['approach-1']['store_size'][1]},
            'rpn': [{k: v} for k, v in data['approach-1']['rpn']]
        },
        'approach-2': {
            'image': str(data['approach-2']['image']),
            'ocr': [{k: v} for k, v in data['approach-2']['ocr']],
            'store_size': {data['approach-2']['store_size'][0]: data['approach-2']['store_size'][1]},
            'yolo': [{k: v} for k, v in data['approach-2']['yolo']]
        },
    }
    return out


import pipeline


# Create your views here.
def index(request):
    return HttpResponse(request, "hi there")


def show_output(request):
    global outdict, thread_running
    if thread_running:
        return HttpResponse(request, {'status': "processing"})
    return HttpResponse(request, outdict)


def process_request():
    global elogger, pipe, store_image, outdict, thread_running
    thread_running = True
    elogger.print("started processing thread")
    elogger.print(pipe.sr)
    outdict = pipe.process_image(store_image)
    thread_running = False
    return


def get_output(request):
    global display_output
    return HttpResponse(display_output, content_type='application/json')


analyse_image_thread = threading.Thread(target=process_request, args=())
analyse_image_thread.daemon = True
pipe = pipeline.PipelineX()
store_image = np.array(np.ones((400, 400, 3)) * 255., dtype=np.uint8)


class SendImage(APIView):
    @staticmethod
    def check_validity(req):
        ret = True
        message = ""
        keys = [w for w in req.keys()]
        if "image" not in keys:
            ret = False
            message += "image is not appended, " \
                       "try appending the image in header files with key 'image', please refer to " \
                       "https://github.com/santhtadi/rest_api_in_django " \
                       "for more details ; "
        return ret, message

    # post is responsible for receiving files
    # develop det put and delete according to your need
    def post(self, request):
        global analyse_image_thread, store_image, thread_running, outdict, display_output
        # print the data in request to dashboard
        print(request.data)
        # convert the request data to a dictionary object in python
        req = dict(request.data)
        # check if all the required files are appended or not
        valid, error_message = self.check_validity(req)
        if not valid:
            return Response({"message": error_message}, status=status.HTTP_400_BAD_REQUEST)
        # read the image as bytes
        by = req['image'][0].read()
        # convert bytes as image using pillow library
        img = Image.open(io.BytesIO(by)).convert('RGB')
        # create an array using numpy
        image_in_rgb_format = np.array(img)
        # change RGB to BGR format for using with opencv library
        image_in_opencv_format = image_in_rgb_format[:, :, ::-1].copy()
        store_image = image_in_opencv_format.copy()
        thread_running = True
        analyse_image_thread = threading.Thread(target=process_request, args=())
        analyse_image_thread.daemon = True
        analyse_image_thread.start()
        while thread_running:
            print(".. processing", time.time())
            time.sleep(1)
        print("sending response")
        display_output = json.dumps(convert_format(outdict))
        # returning size of image as output
        return HttpResponse(display_output, content_type='application/json')
