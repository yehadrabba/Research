import numpy as np
import cv2
from imutils import perspective
import imutils

cap = cv2.VideoCapture('IF_Valpo/IMG_2721.mp4')
#cap = cv2.VideoCapture('IF_Valpo/patio1d.avi')
cap.set(cv2.CAP_PROP_FPS, 0.1)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
fgbg = cv2.createBackgroundSubtractorMOG2()


#CROP DEFINITIONS

scaley=5
scalex=10

#y_len,x_len,_=img.shape
x_len=1010
y_len=510

y0=450
x0=166


nro_cell=0

#CELLS LISTS IN A DICCIONARY
cells_dict = dict()

# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('identificacion.avi',fourcc, 20.0, (640,480))

centros = []

def cropping_cells(frame,x0,y0,scalex,scaley,x_len,y_len):
	

			#CHANGE PIXELS VALUE OF CELL TO HIS MEAN
			#frame[y1:y2,x1:x2] = cell_mean
			#cv2.imshow(str(nro_cell),cell2)
			#print "nro celda = " + str(nro_cell) +" , Intensidad promedio= "  + str(cell_mean)

	return frame, nro_cell, cells_dict




#RUN VIDEO
while(1):
	ret, frame = cap.read()
	#frame = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
	#frame2 = frame.copy()

	#ROI
	#roi = frame[450:960,166:1176]

	#APPLY BACKGROUND SUBTRACTION
	fgmask = fgbg.apply(frame)

	#IMAGE PROCESSING
	fgmask = cv2.threshold(fgmask, 20, 255, cv2.THRESH_BINARY)[1]
	fgmask_open = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
	fgmask= cv2.morphologyEx(fgmask_open, cv2.MORPH_CLOSE, kernel2)
	fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)


	#THRESH CONTOURS > 500 AND DRAW THEM IN ORIGINAL FRAME
	image, contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	for c in contours:
		if cv2.contourArea(c) > 1000:
			#print c
			#cv2.drawContours(frame, contours, -1, (0,255,0), 2)
			box = cv2.minAreaRect(c)
			box = cv2.boxPoints(box)
			box = np.array(box, dtype="int")
			box = perspective.order_points(box)
			#cv2.drawContours(frame, contours, -1, (0,255,0), 2)
			cv2.drawContours(frame, [box.astype("int")], -1, (0, 255, 0), 2)

			M = cv2.moments(c)
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			centros.append([cX,cY])
			# draw the center of the shape on the image
			#print centros
	
	for x,y in centros:
		cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

	"""for y in range(5):
		for x in range(10):
			#CALCULE COORDINATE VALUES
			y1= ((y*y_len)/scaley)+y0
			y2 = ((y+1)*y_len)/scaley+y0 
			x1 = ((x*x_len)/scalex)+x0
			x2 = ((x+1)*x_len)/scalex+x0

			#CROP CELL
			cell=frame[y1:y2 ,x1:x2]
			#CALCULE CELL MEAN
			cell_mean = int(np.mean(cell))
			

			#CELLS COUNTER
			nro_cell=nro_cell+1
			
			#CREA LAS LISTAS CON TODOS LOS PROMEDIOS DE CADA CELDA EN UN DICCIONARIO (1 CELDA = 1 LISTA)
			a = 'Celda_'
			name_cell = a + str(nro_cell)
			if not name_cell in cells_dict:
				cells_dict[name_cell] = []
				cells_dict[name_cell].append(cell_mean)
			else:
				cells_dict[name_cell].append(cell_mean)


			#DRAW RECTANGLE IN CELL
			#cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 2)
"""


	fgmask_bgr = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2BGR)

	#cv2.add(frame2[450:960,166:1176],fgmask_bgr[450:960,166:1176])
	#frame2[450:960,166:1176] = fgmask_bgr[450:960,166:1176]
	fgmask_r = imutils.resize(fgmask_bgr, width=600)	
	frame_r = imutils.resize(frame, width=600)	

	#cv2.imshow('frame_r',frame_r)
	cv2.imshow('frame', np.hstack([frame_r, fgmask_r]))

	# write the flipped frame
	#out.write(frame)
	k = cv2.waitKey(1) & 0xff
	if k == 27:
		break
			
print cells_dict["Celda_1"]










cap.release()
cv2.destroyAllWindows()