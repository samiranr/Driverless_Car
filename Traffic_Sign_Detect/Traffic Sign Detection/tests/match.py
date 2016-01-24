import numpy as np
import cv2
import os

def match(img1,img2,t):
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    print matches
    if len(matches)>0:
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])


    return len(good)
'''

template= cv2.imread('test/stop/template.jpg',0)          # queryImage



indir = 'test/stop/positive/'

for root, dirs, filenames in os.walk(indir):
    for f in filenames:
         img=cv2.imread('test/stop/positive/'+f,0)
         print f,match(template,img,40)

print
print
print

indir = 'test/stop/negative/'

for root, dirs, filenames in os.walk(indir):
    for f in filenames:
         img=cv2.imread('test/stop/negative/'+f,0)
         print f,match(template,img,40)
'''
