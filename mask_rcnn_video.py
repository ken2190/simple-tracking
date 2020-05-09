# Librerías
import cv2
import numpy as np
import os
import sys
import time
import tensorflow as tf

from mrcnn import utils
from mrcnn import visualize 
import mrcnn.model as modellib

sys.path.append(os.path.join("coco"))  # Para acceder a la carpeta de la red coco
import coco

# Directorio de logs
MODEL_DIR = os.path.join("logs")

# Archivo de pesos de la red
COCO_MODEL_PATH = os.path.join("mask_rcnn_coco.h5")

# Descargar el archivo de pesos si no existe
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Cargamos nuestra mascara RCNN y cargamos los pesos en la máscara

model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)
model.load_weights(COCO_MODEL_PATH, by_name=True)



# Clases preentrenas en la red coco

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Función que devuelve un color aleatorio

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


colors = random_colors(len(class_names))
class_dict = {
    name: color for name, color in zip(class_names, colors)
}

# Función que convoluciona la máscara con las distintas imagenes

def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image

# Función que pinta en la imagen la caja, etiqueta y probabilidad de lo detectado

def display_instances(image, boxes, masks, ids, names, scores):
	
    n_instances = boxes.shape[0]

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        color = class_dict[label]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.4, color, 1
             )

    return image


# Capturamos el video y guardamos los fps

capture = cv2.VideoCapture("videos/Malaga.mp4")

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)

capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fps = capture.get(cv2.CAP_PROP_FPS)


# Calculamos el total de frames del video

try:
      prop = cv2.CAP_PROP_FRAME_COUNT
      total = int(capture.get(prop))
      print("[INFO] {} Frames totales: ".format(total))

# Exception por si no se pueden recuperar los frames

except:
         print("[INFO] No se detectó el número de Frames")
         total = -1

# Recorremos los frames del video

writer=None
	
while True:
	
    ret, frame = capture.read() # Capturamos frame

    if not ret:
        break
	
    start = time.time()  # Para estimar el tiempo total

# Aplicamos nuestra máscara en el frame

    results = model.detect([frame], verbose=0) 
    r = results[0]
    frame = display_instances(
        frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
    ) 

    end = time.time()  # Para estimar el tiempo total

    if writer is None:
		# Escribimos en el video de salida
		        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	        	writer = cv2.VideoWriter("output/Malaga_output.avi", fourcc, fps,
	      		(frame.shape[1], frame.shape[0]), True)
          	
            	# Sacamos por pantalla los tiempos estimados
	        	if total > 0:
	        		elap = (end - start)
	        		print("[INFO] Tiempo que tarda un frame: {:.4f} s".format(elap))
	        		print("[INFO] Tiempo estimado: {:.4f}".format(elap * total))
          
        # Escribimos en el disco
    writer.write(frame)

writer.release()
capture.release()
