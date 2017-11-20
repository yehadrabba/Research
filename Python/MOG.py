import cv2
import math
from skimage import color
import pickle
import imutils

cap = cv2.VideoCapture("IF_Valpo/IMG_2721.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
	ret, frame = cap.read()
	rect_frame = frame
	fgmask = fgbg.apply(frame)

	image, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	for c in contours:

			if cv2.contourArea(c) <300:
				continue
			cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

	#mask2 = imutils.resize(frame, width=400)

	#cv2.imshow('mask',frame)

	cv2.imshow('Background Subtraction',fgmask)
	#cv2.imshow("Video",rect_frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()