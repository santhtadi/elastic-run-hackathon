import cv2
import numpy as np
import math


class GetRows:
    def __init__(self):
        pass

    def get_rows(self, img):
        H, W = img.shape[:2]
        # Convert the image to gray-scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the edges in the image using canny detector
        edges = cv2.Canny(gray, 50, 200)
        cv2.erode(edges, np.ones((int(0.005*W), int(0.005*W))))
        # Detect points that form a line
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=int(0.9*W), maxLineGap=250)
        # Draw lines on the image
        lines_rows = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan((y2 - y1) / (x2 - x1)))
            if not 88 < 90 - angle < 92:
                continue
            lines_rows.append((line[0], (y1 + y2) / 2))
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        prev = None
        selected_lines = []
        lines_rows = sorted(lines_rows, key=lambda x: x[1])
        for line in lines_rows:
            if prev is None:
                prev = line
                continue
            d = line[1] - prev[1]
            if 0.01 * H < d < 0.02 * H:
                selected_lines.append(prev[0])
                selected_lines.append(line[0])
            prev = line
        for line in selected_lines:
            x1, y1, x2, y2 = line
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        rows = len(selected_lines)
        print(f"rows score: {rows/3}")
        if rows/4 >= 3:
            ret = 3
        else:
            ret = 1
        return ret
