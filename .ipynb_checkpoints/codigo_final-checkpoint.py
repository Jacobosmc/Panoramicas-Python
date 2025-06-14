#=======LIBRERIAS=======#
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

#
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True, help="imagenes_entrada/")
ap.add_argument("-o", "--output", type=str, required=True, help="imagenes_salida/")
args = vars(ap.parse_args())

# grab the paths to the input images and initialize our images list
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["images"])))
images = []
# loop over the image paths, load each one, and add them to our
# images to stitch list
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	images.append(image)
	
# initialize OpenCV's image stitcher object and then perform the image
# stitching
print("[INFO] stitching images...")
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)

# if the status is '0', then OpenCV successfully performed image
# stitching
if status == 0:
	# write the output stitched image to disk
	cv2.imwrite(args["output"], stitched)
	# display the output stitched image to our screen
	cv2.imshow("Stitched", stitched)
	cv2.waitKey(0)
# otherwise the stitching failed, likely due to not enough keypoints)
# being detected
else:
	print("[INFO] image stitching failed ({})".format(status))