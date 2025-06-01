import cv2
import numpy as np

# Cargar imágenes (asegúrate de que estén ordenadas para el stitching)
image1 = cv2.imread('imagenes/img1.jpg')
image2 = cv2.imread('imagenes/img2.jpg')

# Convertir a escala de grises
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detectar puntos clave y descriptores usando ORB
orb = cv2.ORB_create()

keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# Emparejar puntos clave usando BFMatcher (con Hamming porque es ORB)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Ordenar matches por distancia
matches = sorted(matches, key=lambda x: x.distance)

# Seleccionar los mejores matches
num_good_matches = int(len(matches) * 0.15)  # el 15% más cercano
good_matches = matches[:num_good_matches]

# Extraer las coordenadas de los puntos emparejados
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

# Calcular la homografía
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Aplicar la transformación a la primera imagen
height, width, channels = image2.shape
result = cv2.warpPerspective(image1, H, (width + image1.shape[1], height))
result[0:height, 0:width] = image2

# Mostrar y guardar el resultado
cv2.imshow('Stitched Image', result)
cv2.imwrite('stitched_result.jpg', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
