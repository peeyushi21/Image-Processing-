import cv2
import numpy as np
from PIL import Image
import skimage
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import rotate
from skimage.feature import canny
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy.stats import mode

def solution(image_path):
    ############################
    ############################

    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
 image=cv2.imread(image_path)
 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


 edges = cv2.Canny(gray, 100, 200, apertureSize=7)


 tested_angles = np.deg2rad(np.arange(0.1, 180.0))
 h, theta, d = hough_line(edges, theta=tested_angles)
 accum, angles, dists = hough_line_peaks(h, theta, d)
 most_common_angle = mode(np.around(angles, decimals=2))[0]

 skew_angle = np.rad2deg(most_common_angle - np.pi/2)

 np.rad2deg(most_common_angle)

 (h, w) = image.shape[:2]
 center = (w // 2, h // 2)
 M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
 newImage = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


 gray2 = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
 edges2 = cv2.Canny(gray2, 100, 200, apertureSize=7)



 tested_angles = np.deg2rad(np.arange(0.1, 180.0))
 h2, theta2, d2 = hough_line(edges2, theta=tested_angles)
 accum2, angles2, dists2 = hough_line_peaks(h2, theta2, d2)

 most_common_angle2 = mode(np.around(angles2, decimals=2))[0]

 skew_angle2 = np.rad2deg(most_common_angle2 - np.pi/2)



 (h, w) = newImage.shape[:2]
 center2 = (w // 2, h // 2)
 M2= cv2.getRotationMatrix2D(center2, skew_angle2+skew_angle, 1.0)
 finalImage = cv2.warpAffine(image, M2, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

 grayf= cv2.cvtColor(finalImage, cv2.COLOR_BGR2GRAY)
 (T2, threshInv2) = cv2.threshold(grayf, 240, 255,
 cv2.THRESH_BINARY)


 window_size = 2
 step_size = 2
 height, width, _ = finalImage.shape
 pixel_sum_list = []
 for y in range(0, height - window_size + 1, step_size):
    row_sum = 0
    for x in range(width):
        for i in range(window_size):
            row_sum += np.sum(finalImage[y + i][x])
    pixel_sum_list.append(row_sum)


 arr=np.array(pixel_sum_list)
 p=0
 q=0
 for i in range(arr.shape[0]):
  if arr[i]!=arr[0]:
    p=i
    break
 for i in range(arr.shape[0]):
  if arr[arr.shape[0]-1-i]!=arr[0]:
    q=i
    break

 min_numbers = []
 for i in range(len(arr)):
    if len(min_numbers) < 3:
      min_numbers.append((arr[i], i))
    elif arr[i] < min_numbers[-1][0]:
      min_numbers[-1] = (arr[i], i)
      min_numbers.sort(key=lambda x: x[0])

 min_numbers.sort(key=lambda x: x[1])


 l=len(min_numbers)
 a=min_numbers[0][1]-p
 b=min_numbers[l-1][1]-q
 Angle=0
 if(a>b):
  Angle=180
 Angle

 M3= cv2.getRotationMatrix2D(center2, Angle, 1.0)
 FinalImage = cv2.warpAffine(finalImage, M3, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


 return FinalImage

