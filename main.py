import cv2
import numpy as np

image = cv2.imread('lingkaran.jpg')

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100

#ser circularity filtering parameters
params.filterByCircularity = True
params.minCircularity = 0.9

params.filterByConvexity = True
params.minConvexity = 0.2

params.filterByInertia = True
params.minInertiaRatio = 0.01

#create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

#detector blobs
keypoints = detector.detect(image)

#red circles
blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 225), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
text = "Lingkaran yang terdeteksi :" + str(len(keypoints))
cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 100, 255), 2)

#memunculkan gambar yang telah terdeteksi
cv2.imshow('Panggil image', blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()