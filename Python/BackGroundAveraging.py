import numpy as np
import cv2
import imutils

class BackGroundSubtractor:
	# When constructing background subtractor, we
	# take in two arguments:
	# 1) alpha: The background learning factor, its value should
	# be between 0 and 1. The higher the value, the more quickly
	# your program learns the changes in the background. Therefore, 
	# for a static background use a lower value, like 0.001. But if 
	# your background has moving trees and stuff, use a higher value,
	# maybe start with 0.01.
	# 2) firstFrame: This is the first frame from the video/webcam.
	def __init__(self,alpha,firstFrame):
		self.alpha  = alpha
		self.backGroundModel = firstFrame

	def getForeground(self,frame):
		# apply the background averaging formula:
		# NEW_BACKGROUND = CURRENT_FRAME * ALPHA + OLD_BACKGROUND * (1 - APLHA)
		self.backGroundModel =  frame * self.alpha + self.backGroundModel * (1 - self.alpha)

		# after the previous operation, the dtype of
		# self.backGroundModel will be changed to a float type
		# therefore we do not pass it to cv2.absdiff directly,
		# instead we acquire a copy of it in the uint8 dtype
		# and pass that to absdiff.

		return cv2.absdiff(self.backGroundModel.astype(np.uint8),frame)

cam = cv2.VideoCapture('IF_Valpo/IMG_2719.mp4')

# Just a simple function to perform
# some filtering before any further processing.
def denoise(frame):
    frame = cv2.medianBlur(frame,5)
    frame = cv2.GaussianBlur(frame,(5,5),0)
    
    return frame

ret,frame = cam.read()
if ret is True:
	backSubtractor = BackGroundSubtractor(0.001,denoise(frame))
	run = True
else:
	run = False

while(run):
	# Read a frame from the camera
	ret,frame = cam.read()

	# If the frame was properly read.
	if ret is True:
		# Show the filtered image
		#cv2.imshow('input',denoise(frame))

		# get the foreground
		foreGround = backSubtractor.getForeground(denoise(frame))
		foreGround_gray = cv2.cvtColor(foreGround,cv2.COLOR_BGR2GRAY)

		# Apply thresholding on the background and display the resulting mask
		ret, mask = cv2.threshold(foreGround_gray, 20, 255, cv2.THRESH_BINARY)
		# Note: The mask is displayed as a RGB image, you can
		# display a grayscale image by converting 'foreGround' to
		# a grayscale before applying the threshold.


		#thresh = cv2.dilate(thresh, None, iterations=2)

		kernel = np.ones((10,10),np.uint8)
		#kernely = np.ones((1,10),np.uint8)
		#kernelx = np.ones((10,1),np.uint8)
		#thresh = cv2.dilate(thresh,kernely,iterations = 4)
		thresh= cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		#thresh= cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernely)

		(_, cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		print cnts
		# loop over the contours
		for c in cnts:
			# if the contour is too small, ignore it

			if cv2.contourArea(c) <300:
				continue
			cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)



















		mask2 = imutils.resize(frame, width=400)

		cv2.imshow('mask',mask2)

		key = cv2.waitKey(10) & 0xFF
	else:
		break

	if key == 27:
		break

cam.release()
cv2.destroyAllWindows()