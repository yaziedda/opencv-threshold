import cv2
import numpy as np
import math
from subprocess import call

cap = cv2.VideoCapture(0)

while(cap.isOpened()):

    ret, img = cap.read()
    crop_img = img

    # To Greyscale
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # Global Thresholding
    ret1,global_threshold = cv2.threshold(grey,127,255,cv2.THRESH_BINARY)
    # im_global_threshold = cv2.resize(global_threshold, (500,250))

    # Otsu Tresholding
    # _, otsu = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, otsu = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    im_otsu = cv2.resize(otsu, (500,250))

    # Prewitt Tresholding
    blur = cv2.GaussianBlur(grey,(5,5),0)
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(blur, -1, kernelx)
    img_prewitty = cv2.filter2D(blur, -1, kernely)

    im_img_prewitty = cv2.resize(img_prewitty, (500,250))


    numpy_horizontal = np.hstack((grey, global_threshold, otsu, img_prewitty))
    numpy_horizontal_concat = np.concatenate((grey, global_threshold, otsu, img_prewitty), axis=1)
    numpy_horizontal_concat_v = cv2.resize(numpy_horizontal_concat, (1500, 200))
    
    cv2.imshow('Hasil', numpy_horizontal_concat_v)

    # check OpenCV version to avoid unpacking error
    (version, _, _) = cv2.__version__.split('.')

    print(version)

    # if version == '3':
    #     image, contours, hierarchy = cv2.findContours(otsu.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # elif version == '2':
    #     contours, hierarchy = cv2.findContours(otsu.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # elif version == '4':
    #     contours, hierarchy = cv2.findContours(otsu.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	
    # # find contour with max area
    # cnt = max(contours, key = lambda x: cv2.contourArea(x))
    # # create bounding rectangle around the contour
    # x,y,w,h = cv2.boundingRect(cnt)
    # cv2.rectangle(crop_img,(x,y),(x+w,y+h),(0,0,255),0)

    # # finding convex hull
    # hull = cv2.convexHull(cnt)

    # # drawing contours
    # drawing = np.zeros(crop_img.shape,np.uint8)
    # cv2.drawContours(drawing,[cnt],0,(0,255,0),0)
    # cv2.drawContours(drawing,[hull],0,(0,0,255),0)

    ret1,global_threshold = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    imS = cv2.resize(img, (700, 400))
    cv2.imshow('Video', imS)

    k = cv2.waitKey(10)
    if k == 27:
        break
