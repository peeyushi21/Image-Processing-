import cv2
import numpy as np

import math
# Usage
def solution(image_path):
    image1= cv2.imread(image_path)
    ######################################################################
    ######################################################################
    '''
    The pixel values of output should be 0 and 255 and not 0 and 1
    '''
    #####  WRITE YOUR CODE BELOW THIS LINE ###############################
    img=image1
    mean_shift=cv2.medianBlur(img, 3)  
    grayscale = cv2.cvtColor(mean_shift,cv2.COLOR_BGR2GRAY)
    b,g,r = cv2.split(mean_shift)
    flag=0
    for i in range(grayscale.shape[0]):
     for j in range(grayscale.shape[1]):
        if(b[i][j]>220):
          flag=1
          break
        elif (r[i][j]> 140 and b[i][j]< 80):
          grayscale[i][j]=255;
        else :
          grayscale[i][j]=0;
    def no_sun(gray):
      ker1 = np.ones((3, 3), np.uint8)
      dilation = cv2.dilate(gray, ker1, iterations=1)
      ker2 = np.ones((9, 9), np.uint8)  # Adjust the kernel size as needed
      filled_image = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, ker2)
      gray=filled_image
      fill = gray.copy()

      h, w = gray.shape[:2]
      mask = np.zeros((h+2, w+2), np.uint8)

    
      cv2.floodFill(fill, mask, (0,0), 255);

    
      next = 255-fill

   
      final = gray | next
      return final
    
    def sun(gray):
     for i in range(gray.shape[0]):
      for j in range(gray.shape[1]):
        gray[i][j]=0;
     return gray
    
    if(flag==1):
      image=sun(grayscale)
    else:
      image=no_sun(grayscale)

    ######################################################################  
    return image

