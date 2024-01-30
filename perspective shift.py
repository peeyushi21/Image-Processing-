import cv2
import numpy as np

# Usage
def solution(image_path):
 image= cv2.imread(image_path)
 p, q, channels = image.shape

 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 (T, threshInv) = cv2.threshold(gray, 10, 255,
 cv2.THRESH_BINARY_INV)

 border_pixels = []
 for i in range(threshInv.shape[0]):
  for j in range(threshInv.shape[1]):
    if i == 0 or i == threshInv.shape[0] - 1 or j == 0 or j == threshInv.shape[1] - 1:
      border_pixels.append((i, j))


 for i, j in border_pixels:
  threshInv[i, j] = 255

 edged = cv2.Canny(threshInv, 30, 200)

 corners = cv2.goodFeaturesToTrack(edged,4,0.01,10)
 corners = np.int0(corners)
 input_array=corners



 points = np.empty((4, 2))
 for i in range(4):
  points[i][0]=corners[i][0][0]
  points[i][1]=corners[i][0][1]

 points = points[points[:, 1].argsort()]

  # Sort the top two points by x-coordinate.
 top_two = points[:2]
 top_two = top_two[top_two[:, 0].argsort()]

  # Sort the bottom two points by x-coordinate.
 bottom_two = points[2:]
 bottom_two = bottom_two[bottom_two[:, 0].argsort()]

  # Combine the sorted points into a single array.
 src = np.concatenate((top_two, bottom_two))
 src=src.astype(np.float32)

 dst=np.float32([[0,0],[600,0],[0,600],[600,600]])

 dst=np.float32([[0,0],[600,0],[0,600],[600,600]])
 M = cv2.getPerspectiveTransform(src,dst)
 final = cv2.warpPerspective(image,M,(600,600))

 return final
