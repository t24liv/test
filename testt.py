print('Hello')
print('hello from pyCharm')
print('Hello from the other pc')

import cv2
import numpy as np
import os
from shutil import copyfile

# this code tests an image with 4 different color spaces to choose the optimal color space
# before use:
# make an output folder and specify its path

# change the input path here.
input_path = "/home/isu/PycharmProjects/building_classifier/input"

# change the output path here.
output_path = "/home/isu/PycharmProjects/building_classifier/output/"

# change test image extension here:
image_ext = ".jpg"

lab_output = output_path + "LAB"

if not os.path.exists(lab_output):
    os.makedirs(lab_output)


for filename in os.listdir(input_path):
    if filename.endswith(image_ext):
        bright = cv2.imread(os.path.join(input_path, filename))
        # this seems to be the optimal value for the RGB codes that
        # fits our need with a threshold of 3 and a LAB color space.
        # bgr = [249, 251, 252]
        # thresh = 3

        bgr = [249, 251, 252]
        thresh = 3

        #filename2 = "{} - {}, {}, {} - {}.jpg"

        brightLAB = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)

        # convert 1D array to 3D, then convert it to LAB
        lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0]
        minLAB = np.array([lab[0] - thresh, lab[1] - thresh, lab[2] - thresh])
        maxLAB = np.array([lab[0] + thresh, lab[1] + thresh, lab[2] + thresh])
        maskLAB = cv2.inRange(brightLAB, minLAB, maxLAB)
        resultLAB = cv2.bitwise_and(brightLAB, brightLAB, mask=maskLAB)
        # cv2.imwrite(os.path.join(lab_output, filename2.format(filename, str(bgr[0]), str(bgr[1]), str(bgr[2]), str(thresh))), resultLAB)

        grayImage = cv2.cvtColor(resultLAB, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(lab_output, filename), blackAndWhiteImage)

    else:
        continue
