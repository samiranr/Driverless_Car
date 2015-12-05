import cv2
import os

indir = 'pedestrians128x64'
# test set path

hog = cv2.HOGDescriptor()
hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )


        
for root, dirs, filenames in os.walk(indir):
    for f in filenames:
        img=cv2.imread(indir+"/"+f);
        cv2.imshow("input",img)
     
    

        found,w = hog.detectMultiScale(img, winStride=(4,4), padding=(40,40), scale=1.05) # Tune these parameters 
   
 
        for x, y, w, h in found:
            pad_w, pad_h = int(0.15*w), int(0.05*h)
            cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 255), 1)
    

        
        cv2.imshow("result",img )
        ch = 0xFF & cv2.waitKey(300) # Wait time before switching image
        # press q to skip
        if ch == 27:
            break
        

# When everything is done, release the capture

cv2.destroyAllWindows()




