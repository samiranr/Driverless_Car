import cv2
import numpy as np


img=cv2.imread("test1.jpg")
img_copy=img
cv2.imshow("imageoriginal",img)

# Converting to HSV for illuminatiion invariance
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)



# Threshold the HSV image to get only red colors
img1 = cv2.inRange(img, np.array([0, 100, 100]), np.array([10,255,255]))
img2 = cv2.inRange(img, np.array([160, 100, 100]), np.array([180,255,255]))
img = cv2.bitwise_or(img1,img2)



ret,thresh = cv2.threshold(img,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)





# SURF Starts here




trainimage = cv2.imread('fiftytemplate.png',0) # trainImage


surf = cv2.SURF(400)

kp, des = surf.detectAndCompute(trainimage,None)

print len (kp)






for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
	outpt=img_copy[y:y+h,x:x+w]
        area = cv2.contourArea(cnt)


	if area>100:
		cv2.imshow("image",outpt)
		cv2.waitKey(5000)


cv2.destroyAllWindows()


