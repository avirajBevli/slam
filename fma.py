#feature mapping between two images(a query image and a train image)
#this is not okay for real time detection because it uses too much computation power
import cv2
import numpy as np

img1 = cv2.imread("query_image.jpeg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("train_image.jpeg", cv2.IMREAD_GRAYSCALE)

# ORB Detector object initialization
#for ecample if you want to know more about the orb.create google ORB class reference
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

#the decriptors are atually an array of numbers that define our feature
#for example they can be 120 dimensional vectors

# Brute Force Matching method
#compares each descriptor of the first image with all the descriptors of the second image
#for orb, NORM_HAMMING should be used
#Second param is boolean variable, crossCheck which is false by default. 
#If it is true, Matcher returns only those matches with value (i,j)
#such that i-th descriptor in set A has j-th descriptor in set B as the best match 
#and vice-versa. 
#That is, the two features in both sets should match each other.
#It provides consistant result, and is a good alternative to ratio test proposed by D.Lowe in SIFT paper.

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)
# What is this Matcher Object?
# The result of matches = bf.match(des1,des2) line is a list of DMatch objects. This DMatch object has following attributes:

# DMatch.distance - Distance between descriptors. The lower, the better it is.
# DMatch.trainIdx - Index of the descriptor in train descriptors
# DMatch.queryIdx - Index of the descriptor in query descriptors
# DMatch.imgIdx - Index of the train image.


#sorts the matches according according to the best matches, ie sort them in order of their distance
matches = sorted(matches, key = lambda x:x.distance)
#print(matches)

print("the number of matches found are:")
print(len(matches))


# Like we used cv2.drawKeypoints() to draw keypoints, 
# cv2.drawMatches() helps us to draw the matches. 
# It stacks two images horizontally and 
# draw lines from first image to second image showing best matches.
matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
#gives us the best 50 matches
#without flag=2, the detected features will also be displayed

cv2.imshow("Img1", img1)
cv2.imshow("Img2", img2)
cv2.imshow("Matching result", matching_result)

cv2.waitKey(0)
cv2.destroyAllWindows()