from paddleocr import PaddleOCR, draw_ocr
from PIL import Image


class OCRDetect:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def identify_products(self, img_path, out_path):
        result = self.ocr.ocr(img_path, cls=True)
        image = Image.open(img_path).convert('RGB')
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        im_show = draw_ocr(image, boxes, txts, scores, font_path='./paddle_support/simfang.ttf')
        im_show = Image.fromarray(im_show)
        im_show.save(f'{out_path}')
        return boxes, txts, scores
