import cv2


# get the video stream
cap = cv2.VideoCapture("highway_test2.mp4")
# create a named window (to show the video)
cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)	


# while capture is opened
while cap.isOpened():
		try:
			# Read frame from the video
			ret, frame = cap.read()
		except:
			continue

		if ret:	
			cv2.imshow("Detected lanes", frame)
			
		else:
			break

		# Press key q to stop
		if cv2.waitKey(1) == ord('q'):
			break

cap.release()
cv2.destroyAllWindows()