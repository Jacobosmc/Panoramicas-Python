import cv2
import glob

# Listar tus imágenes
image_files = sorted(glob.glob('imagenes/*.jpg'))  # Cambia la ruta

# Leerlas todas en una lista
images = [cv2.imread(f) for f in image_files]

# Crear el stitcher
stitcher = cv2.Stitcher_create()

# Intentar coserlas
status, stitched = stitcher.stitch(images)

# if status == cv2.Stitcher_OK:
#     print("¡Stitching exitoso!")
#     cv2.imshow("Panorama", stitched)
#     cv2.imwrite("stitched_panorama.jpg", stitched)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print(f"Error en el stitching: {status}")

# Leer la imagen final ya cosida
# stitched = cv2.imread("stitched_panorama.jpg")

# Convertir a escala de grises
gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)

# Crear una máscara binaria de los píxeles válidos (no negros)
# Aquí consideramos que negro puro (0) es fondo, >0 es imagen
_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# Encontrar los contornos
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Encontrar el rectángulo delimitador más grande
max_area = 0
best_rect = None
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    if area > max_area:
        max_area = area
        best_rect = (x, y, w, h)

# Recortar usando el rectángulo encontrado
if best_rect is not None:
    x, y, w, h = best_rect
    cropped = stitched[y:y+h, x:x+w]

    # Mostrar y guardar el resultado final
    cv2.imshow("Recorte", cropped)
    cv2.imwrite("stitched_panorama_cropped.jpg", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No se encontró un área válida para recortar.")