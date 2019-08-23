#This program does feature mapping of the video frames to the image dunamically
#In this program, initially the image was too big in size(5000*6000) around pixels and size was aroung 5MBs
#hence, the mactching results wasnt able to fit inside the screen
#hence, i had to first compress the image to about 500*600 pixels and then the matching results was able to fit in the laptop screen

import cv2
import numpy as np
 
# Create a VideoCapture object
cap = cv2.VideoCapture(0)
 
# Check if camera opened successfully
if (cap.isOpened() == False): 
	print("Unable to read camera feed")
 
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

img = cv2.imread("img_book.jpg", cv2.IMREAD_GRAYSCALE)
 
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

orb = cv2.ORB_create()

while(True):
	ret, frame = cap.read()
 	#frame = cv2.resize(frame,(200,100))

	if ret == True: 

		kp1, des1 = orb.detectAndCompute(img, None)
		kp2, des2 = orb.detectAndCompute(frame, None)

		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

		matches = bf.match(des1, des2)

		#sorts the matches according according to the best matches
		matches = sorted(matches, key = lambda x:x.distance)

		matching_result = cv2.drawMatches(img, kp1, frame, kp2, matches[:50], None, flags=2)
		#gives us the best 50 matches

		cv2.imshow('frame',frame)
		cv2.imshow("Matching result", matching_result)

		# Press Q on keyboard to stop recording
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	 
	# Break the loop
	else:
		break 
 
# When everything done, release the video capture and video write objects
cap.release()
out.release()
 
# Closes all the frames
cv2.destroyAllWindows() 
