import cv2
import numpy as np
import time
import pandas as pd

filename = 'NEW.mp4'
cap = cv2.VideoCapture(filename)

# baseline = None
baseline = cv2.imread("base.jpg")
baseline = baseline[140:400, 100:370]
baseline = cv2.medianBlur(baseline, 5)
baseline = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
baseline = cv2.medianBlur(baseline, 5)

# Create trackbars for color change
# Hue is from 0-179 for Opencv

subtractor = cv2.createBackgroundSubtractorMOG2()
# subtractor.getBackgroundImage(baseline)

hMin = 0
hMax = 185
sMin = 75
sMax = 160
vMax = 255
vMin = 190
red_lower = np.array([hMin, sMin, vMin])
red_upper = np.array([hMax, sMax, vMax])


class Laser:
    def __init__(self):
        self.last_point = np.array([0, 0])
        self.is_stable = 0

    def speed(self, center: np.ndarray):
        if np.mean(np.abs(center - self.last_point)) < 4:
            self.is_stable += 1
            self.last_point = self.last_point * 0.2 + center * 0.8
        else:
            self.is_stable = 0
            self.last_point = center

    def find_laser(self, _image) -> np.ndarray:
        contours, hierarchy = cv2.findContours(_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circle_ = np.array([])
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            area = cv2.contourArea(c)
            # print(approx)
            if len(approx) > 4 and 200 < area < 6000:
                ((x, y), r) = cv2.minEnclosingCircle(c)
                circle_ = np.array([x, y, r])

        return circle_


big_center = []
laser = Laser()
_enumerate = 0
laser_point = []
laser_means = []
point_save = []

try:
    while cap.isOpened():
        _enumerate += 1
        ret, image = cap.read()
        # if 27663 > _enumerate < 37575:
        #     print(_enumerate)
        #     continue
        if _enumerate % 2 != 0:
            continue

        # if _enumerate > 1000:
        #     break

        if image is None:
            break

        # writerVid.write(image)

        image = image[140:400, 100:370]

        image = cv2.medianBlur(image, 3)
        # image_ = image.copy()
        # red_layer = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[:, :, 0]
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # frame = cv2.medianBlur(frame, 5)
        frame_ = frame.copy()

        # Convert to HSV format and color threshold
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, red_lower, red_upper)
        # result = cv2.bitwise_and(image, image, mask=mask)
        mask = cv2.erode(mask, kernel=np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, kernel=np.ones((3, 3), np.uint8))
        # cv2.imshow('image', mask)

        # frame = cv2.erode(frame, kernel=np.ones((6, 6), np.uint8))
        frame = cv2.dilate(frame, kernel=np.ones((3, 3), np.uint8))
        # # cv2.Circle
        #
        # frame2 = cv2.erode(frame2, kernel=np.ones((4, 4), np.uint8))
        # frame2 = cv2.dilate(frame2, kernel=np.ones((3, 3), np.uint8))
        #
        # # circles = cv2.HoughCircles(frame2, cv2.HOUGH_GRADIENT, 1, 10,
        # #                            # param1=150, param2=80,
        # #                            minRadius=1, maxRadius=1000)
        # #
        # frame2 = 255 - frame2

        circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, 100,
                                   param1=150, param2=80,
                                   minRadius=10, maxRadius=1000)
        #
        if circles is not None:
            circles = np.uint16(np.around(circles))

            for i in circles[0, :]:
                center = (i[0], i[1])
                cv2.circle(image, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                big_center.append(center)
                cv2.circle(image, center, radius, (255, 0, 255), 1)

        circle = laser.find_laser(mask)

        # print(circles)
        if len(circle) > 0:
            laser.speed(circle[:2])
            circle = np.uint16(np.around(circle))

            if laser.is_stable:
                center = (circle[0], circle[1])
                cv2.circle(image, center, 1, (0, 100, 100), 3)
                # circle outline
                cv2.circle(image, center, circle[2], (255, 0, 255), 3)
                laser_point.append(center)
            else:
                if len(laser_point) > 20:
                    points = np.asarray(laser_point, dtype=np.float32)
                    laser_means.append(np.mean(points, axis=0).tolist())
                    print(laser_means[-1])
                    point_save.append(_enumerate)
                    # print(np.mean(points, axis=0) - center_of_circle)
                    laser_point = []
        cv2.putText(image, f"{len(laser_means)}", (700, 250), cv2.FONT_HERSHEY_DUPLEX, 2,
                    (255, 255, 255))
        # if _enumerate == 1:

        # print(image.shape)
        # cv2.circle(image, [image.shape[1] // 2, image.shape[0] // 2], 1, (255, 100, 100), 3)
        #

        result = np.abs(np.asarray(frame_, dtype=np.float32) - np.asarray(baseline, dtype=np.float32))
        result[result < 20] = 0
        result[result > 20] = 255

        cv2.imshow("Win", frame)
        # # cv2.imshow("Win2", cv2.bilateralFilter(baseline, 9, 75, 75))
        # # print(frame_.shape, baseline.shape)
        # # print(np.mean(np.abs(image_ - baseline)))
        # #
        # # cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('w'):
            # cv2.imwrite("base.jpg", image)
            break

        # if _enumerate > 500:
        #     break
    delta = np.array(laser_means) - np.mean(np.asarray(big_center, dtype=np.float32), axis=0)
    laser_means = np.asarray(laser_means)
    print(delta[:, 0])
    print(laser_means[:, 0])
    pd.DataFrame({
        "frame": point_save,
        "laser_x": laser_means[:, 0],
        "delta_x": delta[:, 0],
        "laser_y": laser_means[:, 1],
        "delta_y": delta[:, 1],
    }).to_excel(f"{filename.split('.')[0]}.xlsx")
    print(np.array(big_center).shape)
except Exception as e:
    print(e)

# writerVid.release()
cap.release()
cv2.destroyAllWindows()
