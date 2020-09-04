import cv2
import numpy as np
import os
import math
import sys
import time
import tensorflow as tf

from mrcnn import utils
from mrcnn import visualize 
import mrcnn.model as modellib

sys.path.append(os.path.join("coco/"))  # Cargamos dataset coco (Common Objects in Context)
import coco

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="path to input image")
args = vars(parser.parse_args())


# Directorio de logs
MODEL_DIR = os.path.join("logs/")

# Ruta del archivo de pesos de la red rcnn
COCO_MODEL_PATH = os.path.join("mask_rcnn_coco.h5")

# Si no existe la ruta descargar los pesos a partir de la Release de Coco
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Configuramos numero de GPUs e imagenes por GPU
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# Cargamos nuestra mascara RCNN y cargamos los pesos en la máscara
def load_model():
  #Cargamos modelo RCNN (constructor)
  model = modellib.MaskRCNN(
      mode="inference", model_dir=MODEL_DIR, config=config
  )
  #Volcamos los pesos en el modelo RCNN
  model.load_weights(COCO_MODEL_PATH, by_name=True)

  # Clases contempladas en la dataset de coco
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

  return model, class_names

def asig_color():
  #Asigna un color a cada clase
  colors = [tuple(255 * np.random.rand(3)) for _ in range(512)]
  
  return colors

def lee_video(video_in):
  # Leemos el video de entrada
  capture = cv2.VideoCapture(video_in)

  #Ajustamos la resolción de los frames a 720x480
  capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
  capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  
  return capture

def calc_frames():

  #Sacamos fps
  fps = capture.get(cv2.CAP_PROP_FPS)

  print(int(fps))

  # Calculamos el total de frames del video
  try:
    prop = cv2.CAP_PROP_FRAME_COUNT
    total = int(capture.get(prop))
    print("[INFO] {} Frames totales: ".format(total))

  # Exception por si no se pueden recuperar los frames
  except:
    print("[INFO] No se detectó el número de Frames")
    total = -1
  return fps, total

def timer():
    push = time.time()  # Para sacar el tiempo actual
    return push

def diff_time (start, end, total):
    # Sacamos por pantalla los tiempos estimados
    if total > 0:
      elap = (end - start)
      print("[INFO] Tiempo que ha tardado el frame: {:.4f} s".format(elap))
      print("[INFO] Tiempo estimado: {:.4f}".format(elap * total)
	    
def draw_lbl(image, box, color, caption, text):
  y1, x1, y2, x2 = box
  image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
  image = cv2.putText(
      image, str(caption), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )
  image = cv2.putText(
      image ,text , (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )
  return image	  
	    
def detecta_personas(boxes, masks, ids, names, scores, class_detected):

  personas = []
  masks_personas = []
  scores_personas = []
  n_instances = boxes.shape[0]

  if not n_instances:
    print('NO INSTANCES TO DISPLAY')
  else:
    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue

        label = names[ids[i]]
        
        if label in class_detected: # Solo personas

          personas.append(boxes[i])
          masks_personas.append(masks[i])
          scores_personas.append(scores[i])

  return personas, masks_personas, scores_personas
	    
def distancia(a,b):
  distance = np.linalg.norm(a-b, axis = 1)
  min_value = distance.min()
  indx = np.where(distance==min_value)[0][0]
  
  return indx, min_value

def write_vid(writer,frame,video_out, fps):
  if writer is None:
  # Escribimos en el video de salida
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(video_out, fourcc, fps, (frame.shape[1], frame.shape[0]), True)
    
  return writer
# Variables previas

n_frame = 8
ref_n_frame_axies = []
ref_n_frame_label = []
ref_n_frame_axies_flatten = []
ref_n_frame_label_flatten = []
label_cnt = 1
frm_num = 1
min_distance = 50

input = str(args["input"])
video_in = "videos/" + input
video_out = "output/" + input.replace(".mp4", "_out.avi")
class_detected = 'person' # Detectaremos solo personas

writer = None
# Cargamos modelo

model,class_names = load_model()

# Leemos el video de entrada
capture = lee_video(video_in)

# Calculamos los frames
fps, total = calc_frames()

# Lista de Colores
colors= asig_color()

'''
Bucle de imágenes
'''

while True:
  
    ret, frame = capture.read() # Capturamos frame

    if ret == True: # En caso de capturarlo correctamente

      start = timer() # Ponemos en marcha timer
      '''
      Proceso cada imagen
      '''

      cur_frame_axies = []
      cur_frame_label = []

      # Hacemos la detección de Objetos

      results = model.detect([frame], verbose=0) 
      r = results[0] # Solo una imagen

      # Hacemos la detección de personas exclusivamente
      personas, people_masks, people_scores = detecta_personas( r['rois'],
      r['masks'], r['class_ids'], class_names, r['scores'], class_detected)

      n_instances = len(personas)
      if not n_instances:
          print('NO INSTANCES TO DISPLAY')

      for i in range(n_instances):
          if not np.any(personas[i]):
              continue

          score = people_scores[i]
          box = personas[i]
          lbl = float('nan')
          if (len(ref_n_frame_label_flatten) > 0):
            a = np.array(ref_n_frame_axies_flatten)
            b = np.array(box)
            idx, dist = distancia(a,b)

            if dist < 50:
              lbl = ref_n_frame_label_flatten[idx]

          if (math.isnan(lbl)):

              lbl = label_cnt
              label_cnt += 1

          cur_frame_label.append(lbl)
          cur_frame_axies.append(box)
          color = colors[lbl]  
          text = 'Persona: ' + str(lbl)
          image = draw_lbl(frame, box, color, score, text)
          
      if (len(ref_n_frame_axies) == n_frame):
        del ref_n_frame_axies[0]
        del ref_n_frame_label[0]

      ref_n_frame_label.append(cur_frame_label)
      ref_n_frame_axies.append(cur_frame_axies)
      ref_n_frame_axies_flatten = [a for ref_n_frame_axie in ref_n_frame_axies for a in ref_n_frame_axie]
      ref_n_frame_label_flatten = [b for ref_n_frame_lbl in ref_n_frame_label for b in ref_n_frame_lbl]


      end = timer() # Paramos timer
      writer = write_vid(writer, frame, video_out, fps)   # Escribimos en el video de salida

      diff_time (start, end, total) # Sacamos por pantalla diferencia de tiempos
          
        # Escribimos en el disco
      writer.write(frame)
    else:
      break

writer.release()
capture.release()
