from __future__ import print_function
import numpy as np
import time

import cv2

import sys
from glob import glob
import itertools as it


import cv

vidFile = cv.CaptureFromFile( 'videoplayback.mp4' )
nFrames = int(  cv.GetCaptureProperty( vidFile, cv.CV_CAP_PROP_FRAME_COUNT ) )
fps = cv.GetCaptureProperty( vidFile, 4 )
waitPerFrameInMillisec = int(20)

hog = cv2.HOGDescriptor()
hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh



def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)
        

for f in xrange( nFrames ):

    img=cv.QueryFrame( vidFile )
    print (img)
    img=np.asarray(img[:,:])

    found,w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)
    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and inside(r, q):
                break
        else:
            found_filtered.append(r)
        draw_detections(img, found)
        draw_detections(img, found_filtered, 3)
        #print('%d (%d) found' % (len(found_filtered), len(found)))


    cv2.imshow('Video',img )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()












