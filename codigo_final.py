#=======LIBRERIAS=======#
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2


# ESTO ES PARA USARLO POR COMANDOS --> DESPUES LO CAMBIAMOS PARA QUE SEA POR JUPYTER
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--images", type = str, required = True, help = "imagenes_entrada/")
# ap.add_argument("-o", "--output", type = str, required = True, help = "imagenes_salida/")
# ap.add_argument("-c", "--crop", type = int, default=0, help="whether to crop out largest rectangular region")
# args = vars(ap.parse_args())

# DIFERENTES ARGUMENTOS NECESARIOS:
# · RUTA DE LAS IMÁGENES QUE QUIERES USAR[ORDENADAS EN LA CARPETA],
# · RUTA DE SALIDA + NOMBRE DE LA IMAGEN,
# · SI QUIERES QUE SE RECORTE O NO
args = ["imagenes_entrada/dele/", "imagenes_salida/salida_mal.png"] # ESTO QUE LO HAGA EL USUARIO EN EL MOMENTO

print("[INFO] loading images...") # PRINTS EN CONSOLA PARA VER QUE VA

# Guardamos las imagenes en una lista
imagePaths = sorted(list(paths.list_images(args[0])))
images = []
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	images.append(image)
	
# Momento donde se unen las imágenes, pero pueden quedar mal
print("[INFO] stitching images...")  # PRINTS EN CONSOLA PARA VER QUE VA
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)

if status == 0: # Si sale 0 es que va bien la unión de las imágenes
	cv2.imwrite(args[1], stitched)
	cv2.imshow("Panorámica", stitched)
	cv2.waitKey(0)
	args.append(int(input("¿Quieres recortar la panorámica para que no tenga marcas negras raras?(0 es si, 1 es no):")))
	if args[2] == 0: #0 si quieres recortar, 1 si no
		# Podríamos hacer que el usuario prevea la imagen, y si quiere elegir que se corte que sea aquí

		print("[INFO] cropping...") # PRINTS EN CONSOLA PARA VER QUE VA
		stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0)) # Borde de 10 píxeles
		
		# convert the stitched image to grayscale and threshold it such that all pixels greater than zero are set to 
        # 255 (foreground) while all others remain 0 (background)
		gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
		thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
		
        # find all external contours in the threshold image then find the *largest* contour which will be 
        # the contour/outline of the stitched image
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		c = max(cnts, key=cv2.contourArea)
		# allocate memory for the mask which will contain the rectangular bounding box of the stitched image region
		mask = np.zeros(thresh.shape, dtype="uint8")
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
		
        # create two copies of the mask: one to serve as our actual minimum rectangular region and another to serve as a counter
		# for how many pixels need to be removed to form the minimum rectangular region
		minRect = mask.copy()
		sub = mask.copy()
		# keep looping until there are no non-zero pixels left in the subtracted image
		while cv2.countNonZero(sub) > 0:
			# erode the minimum rectangular mask and then subtract the thresholded image from the minimum rectangular mask
			# so we can count if there are any non-zero pixels left
			minRect = cv2.erode(minRect, None)
			sub = cv2.subtract(minRect, thresh)
			
        # find contours in the minimum rectangular mask and then extract the bounding box (x, y)-coordinates
		cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		c = max(cnts, key=cv2.contourArea)
		(x, y, w, h) = cv2.boundingRect(c)
		# use the bounding box coordinates to extract the our final stitched image
		stitched = stitched[y:y + h, x:x + w]
	# write the output stitched image to disk
	cv2.imwrite(args[1], stitched)
	# display the output stitched image to our screen
	cv2.imshow("Stitched", stitched)
	cv2.waitKey(0)
	
else: # SI VA MAL
	print("[INFO] image stitching failed ({})".format(status))