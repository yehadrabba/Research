import numpy as np
import cv2
import imutils
from imutils import perspective
#from imutils import contours
cap = cv2.VideoCapture('IF_Valpo/IMG_2721.mp4')
fgbg= cv2.createBackgroundSubtractorMOG2()


while(1):

	ret, frame = cap.read()
	fgmask = fgbg.apply(frame)
	#mask = np.ones(frame.shape[:2], dtype="uint8") * 255


	image, contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	for c in contours:
		if cv2.contourArea(c) > 10:
			continue
		box = cv2.minAreaRect(c)
		box = cv2.boxPoints(box)
		box = np.array(box, dtype="int")
		box = perspective.order_points(box)
		cv2.drawContours(frame, contours, -1, (0,255,0), 2)

		#final_frame = cv2.drawContours(frame.copy(), [box.astype("int")], -1, (0, 255, 0), 2)

	#dibujado_r = imutils.resize(dibujado, width=650)
    #cv2.imshow('frame', np.hstack([frame, fgmask_r]))
	#finalframe_r = imutils.resize(final_frame, width=650)
	#fgmaskr = imutils.resize(fgmask, width=500)

	cv2.imshow('final frame',frame)
	key = cv2.waitKey(10) & 0xFF




#def delete_noise(mask, orig):
#	image, contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#	for c in contours:
#		if cv2.contourArea(c) < 5:
#			continue
#		box = cv2.minAreaRect(c)
#		box = cv2.boxPoints(box)
#		box = np.array(box, dtype="int")
#		box = perspective.order_points(box)

#		cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
#	return orig





cap.release()
cv2.destroyAllWindows()
