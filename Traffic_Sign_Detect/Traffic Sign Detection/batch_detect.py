import cv2
import numpy as np
import os
import libsvm
import argparse
from cPickle import load
from learn import extractSift, computeHistograms, writeHistogramsToFile
import copy


# Automatic Canny Edge Detection Using Median of Pixel Intensities
def auto_canny(image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)
 
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
 
        # return the edged image
        return edged

di=[]
for f in os.walk('tests/FullIJCNN2013/'):
        di.extend(f)

#Define Red
lower_red = np.array([0, 90, 60], dtype=np.uint8)
upper_red = np.array([10, 255, 255], dtype=np.uint8)
red = [lower_red, upper_red, 'red']

#Define Green
lower_green = np.array([60, 55, 0], dtype=np.uint8)
upper_green = np.array([100, 255, 120], dtype=np.uint8)
green = [lower_green, upper_green, 'green']

#Define Blue
lower_blue = np.array([90, 20, 60], dtype=np.uint8)
upper_blue = np.array([130, 255, 180], dtype=np.uint8)
blue = [lower_blue, upper_blue, 'blue']

for filename in di[2]:
        print filename

        img=cv2.imread("tests/FullIJCNN2013/"+filename)


        
        img_copy=img
        img_copy2= copy.copy(img)
        shape_img = copy.copy(img)
        #cv2.imshow("imageoriginal",img)

        # Converting to HSV 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        b,g,r = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(8,8))
        clahe = clahe.apply(r)
        img = cv2.merge((b,g,clahe))
        
    
        

        



        # Threshold the HSV image to get only red colors
        img1 = cv2.inRange(img, np.array([0, 100, 100]), np.array([10,255,255]))
        img2 = cv2.inRange(img, np.array([160, 100, 100]), np.array([180,255,255]))
        img3 = cv2.inRange(img, np.array([160, 40, 60]), np.array([180,70,80]))
        img4 = cv2.inRange(img, np.array([0, 150, 40]), np.array([20,190,75]))
        img5 = cv2.inRange(img, np.array([145, 35, 65]), np.array([170,65,90]))
        

        img = cv2.bitwise_or(img1,img3)
        img = cv2.bitwise_or(img,img2)
        img = cv2.bitwise_or(img,img4)
        img = cv2.bitwise_or(img,img5)

        


 
        cv2.imshow("red thresholded",img)
        ret,thresh = cv2.threshold(img,127,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        



        # Starting shape detection pipeline (incomplete)

        shape_img = cv2.GaussianBlur(shape_img,(5,5),0)
        shape_img = cv2.cvtColor(shape_img, cv2.COLOR_BGR2GRAY)
        shape_img=auto_canny(shape_img)

        #cv2.imshow("image",shape_img)

        # Deletion of non Candidates
        temp=[]
        for cnt in contours:
                area = cv2.contourArea(cnt)
                if area>120:
                        temp.append(cnt)

        contours=temp

        # drawing Rectangles



        
                
        # Classifying each rectangle
        final_detection=[]
        for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                pad_w, pad_h = int(0.15*w), int(0.05*h)
                roi=img_copy2[y-pad_h:y+h+pad_h,x-pad_w:x+w+pad_w]
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                area = cv2.contourArea(cnt)


                if area>100:
                       
                        cv2.imshow("image",roi)
                        cv2.imwrite('current.jpg',roi)

                        # Initializing The Classifier







                       
                        
                        
                        cv2.waitKey(5000)
                                



                                
        for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)

                
                
                cv2.rectangle(img_copy, (x,y), (x+w,y+h),(255,0,0), 2)
     
  
                        
 


        
        cv2.imshow("image",img_copy)
        cv2.waitKey(5000)


cv2.destroyAllWindows()


