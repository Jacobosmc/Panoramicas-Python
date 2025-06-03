# AUTORES: Jacobo Sánchez-Malo Cabañero y Dani Díaz Lafuente

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
args = ["imagenes_entrada/balcon/", "imagenes_salida/pano_balcon.png", "0"] # ESTO QUE LO HAGA EL USUARIO EN EL MOMENTO

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
	
    # El usuario ve primero la imagen, y después se le pregunta si quiere recortarla o no
	cv2.imwrite(args[1], stitched)
	cv2.imshow("Panorámica", stitched)
	cv2.waitKey(0)
	while args[2] not in (0, 1):
		args[2] = input("¿Quieres recortar la panorámica para que no tenga marcas negras raras?(SI/NO):")
		if args[2].lower() == "si":
			args[2] = 0 
		elif args[2].lower() == "no":
			args[2] = 1
		else:
			args[2] = "Respuesta errónea"
		print(args[2])

	if args[2] == 0: #0 si quieres recortar, 1 si no

		print("[INFO] cropping...") # PRINTS EN CONSOLA PARA VER QUE VA
		stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0)) # Borde de 10 píxeles
		
		# Separa entre negro (del fondo) y la foto en si, poniendo a 0 el color del fondo y a 255 la de la imagen
		gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
		thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
		
        # Busca los contornos externos de la imagen umbralizada (blanco=foto y negro=fondo) 
		# y busca el contorno más grande para marcarlo como borde de la imagen
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		c = max(cnts, key=cv2.contourArea)

		# Creacion de la variable que contendrá la máscara
		mask = np.zeros(thresh.shape, dtype="uint8")
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
		
        # Usaremos 2 copias de la máscara de la panorámica
		minRect = mask.copy() # Con esta se irán erosinando los bordes 
		sub = mask.copy() # Esta otra contara la cantidad de pixeles que se quitarán usando ese rectángulo
		
		# bucle para eliminar todos los pixeles blancos, para saber hasta donde hay que cortar la imagen
		while cv2.countNonZero(sub) > 0:
			minRect = cv2.erode(minRect, None) # Rascamos parte del borde
			sub = cv2.subtract(minRect, thresh) # Quitamos esa parte
			# Así hasta quitar los pixeles que son foto (blancos)
			
        # Busca los contornos de la máscara recortada y saca las coordenadas para el borde
		cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		c = max(cnts, key=cv2.contourArea)
		(x, y, w, h) = cv2.boundingRect(c) # Coordenadas

		# Quitamos de la imagen el borde
		stitched = stitched[y:y + h, x:x + w]

		# Lo mismo de antes, para guardar y visualizar la imagen --> HACER QUE SE VEA EN EL JUPYTER
		cv2.imwrite(args[1], stitched)
		cv2.imshow("Stitched", stitched)
		cv2.waitKey(0)
			
else: # SI VA MAL
	print("[INFO] image stitching failed ({})".format(status))