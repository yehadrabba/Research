#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

cap = cv2.VideoCapture('IF_Valpo/IMG_2721.mp4')


#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output1.avi',fourcc, 20.0, (640,480))

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


while(cap.isOpened()):
	ret, image = cap.read()

	image = imutils.resize(image, width=800)
	orig = image.copy()

	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4,4),
	padding=(8,8), scale=1.05)

	# draw the original bounding boxes
	for (x, y, w, h) in rects:
		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
		cv2.putText(image, "Persona", (xA, yA+10), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 255), 1)
	## show some information on the number of bounding boxes
	#filename = imagePath[imagePath.rfind("/") + 1:]
	#print("[INFO] {}: {} original boxes, {} after suppression".format(
	#	filename, len(rects), len(pick)))

	#if ret==True:

	#		frame = cv2.flip(image,0)

	        # write the flipped frame
	#		out.write(frame)

#			cv2.imshow('frame',frame)
	#		if cv2.waitKey(1) & 0xFF == ord('q'):
	#			break
	#image2 = imutils.resize(image, width=500)
	cv2.imshow('frame',image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
#out.release()


cv2.waitKey(0)
cv2.destroyAllWindows()
