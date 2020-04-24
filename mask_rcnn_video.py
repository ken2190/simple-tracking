
# Librerías
import numpy as np
import imutils
import time
import cv2
import os


# Cargamos la máscara R-CNN donde vamos a cargar nuestra red preentrenada
labelsPath = os.path.sep.join(["mask-rcnn-coco","object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# Inicializamos los colores (random) para representar los diferentes objetos
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# Pasamos los paths de los pesos y configuración
weightsPath = os.path.sep.join(["mask-rcnn-coco",
	"frozen_inference_graph.pb"])
configPath = os.path.sep.join(["mask-rcnn-coco",
	"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# Cargamos nuestra red R-CNN con la base de datos Coco
print("[INFO] Cargando Mask R-CNN ...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# Iniciamos la captura del video
vs = cv2.VideoCapture("videos/Berlin.mp4")
writer = None

# Calculamos el total de frames del video
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} Frames totales: ".format(total))

# Exception por si no se pueden recuperar los frames
except:
	print("[INFO] No se detectó el número de Frames")
	total = -1

# Bucle para cada frame del video
while True:
	# Leemos siguiente frame
	(grabbed, frame) = vs.read()

	# Si no cogemos el frame continuamos al siguiente (Si procesamos un frame vacío da error, esto es para evitarlo)
	if not grabbed:
		break

	# Construimos una región y vamos pasando la red por ella , nos devuelve los limites de la caja y las coordenadas de los objetos
	blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	(boxes, masks) = net.forward(["detection_out_final",
		"detection_masks"])
	end = time.time()

	# Bucle para las distintas clases posibles
	for i in range(0, boxes.shape[2]):
		# extraemos el ID de la clase con probabilidad asociada a la predicción
		classID = int(boxes[0, 0, i, 1])
		confidence = boxes[0, 0, i, 2]

		# Filtramos las predicciones para que solo obtnegamos las que son más o menos fiables con confianza > 0.3 (Por ejemplo)
		if confidence > 0.3:
			# Escalamos las cajas para que se ajusten a la ventana
			(H, W) = frame.shape[:2]
			box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
			(startX, startY, endX, endY) = box.astype("int")
			boxW = endX - startX
			boxH = endY - startY

			# Extrae la segmentación en pixeles de los objetos,
			# redimensionamos la máscara con los límites de las cajas y añadimos un pequeño umbral
			mask = masks[i, classID]
			mask = cv2.resize(mask, (boxW, boxH),
				interpolation=cv2.INTER_NEAREST)
			mask = (mask > 0.5)

			# Extreamos las regiones de interes
			roi = frame[startY:endY, startX:endX][mask]

			# Obtenemos los colores de la clase detectada
			color = COLORS[classID]
			blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

			# Almacenamos la región de interés
			frame[startY:endY, startX:endX][mask] = blended

			# Dibujamos las cajas
			color = [int(c) for c in color]
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				color, 2)

			# Escribimos el nombre de la clase detectada y la probabilidad de la misma
			text = "{}: {:.4f}".format(LABELS[classID], confidence)
			cv2.putText(frame, text, (startX, startY - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	# Comprobamos si se ha iniciado la captura de video
	if writer is None:
		# Escribimos en el video de salida
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter("output/Berlin_output.avi", fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# Sacamos por pantalla los tiempos estimados
		if total > 0:
			elap = (end - start)
			print("[INFO] Tiempo que tarda un frame: {:.4f} s".format(elap))
			print("[INFO] Tiempo estimado: {:.4f}".format(
				elap * total))

	# write the output frame to disk
	writer.write(frame)

print("[INFO] Limpiando...")
writer.release()
vs.release()