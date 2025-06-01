import cv2
import numpy as np
import glob

# Listar las imágenes que queremos coser (ordenadas)
image_files = sorted(glob.glob('imagenes/*.jpg'))  # Ajusta la ruta a tus imágenes

# Leer la primera imagen como base
base_image = cv2.imread(image_files[0])
base_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

# ORB detector
orb = cv2.ORB_create(5000)

# Recorrer el resto de las imágenes
for image_file in image_files[1:]:
    next_image = cv2.imread(image_file)
    next_gray = cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)

    # Detectar puntos clave y descriptores
    kp1, des1 = orb.detectAndCompute(base_gray, None)
    kp2, des2 = orb.detectAndCompute(next_gray, None)

    # Emparejar características
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:int(len(matches) * 0.15)]

    # Extraer coordenadas de puntos emparejados
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    # Calcular la homografía con RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Crear un lienzo grande para la imagen resultante
    h1, w1 = base_image.shape[:2]
    h2, w2 = next_image.shape[:2]
    result_width = w1 + w2
    result_height = max(h1, h2)

    # Warp la base_image al lienzo
    warped_image = cv2.warpPerspective(base_image, H, (result_width, result_height))

    # Pegar la nueva imagen sobre la resultante (con compensación)
    warped_image[0:h2, 0:w2] = next_image

    # Actualizar base_image y base_gray para el siguiente paso
    base_image = warped_image
    base_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

# Mostrar y guardar el resultado final
cv2.imshow('Panorama', base_image)
cv2.imwrite('stitched_panorama.jpg', base_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
