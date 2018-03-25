import cv2
import h5py
import numpy as np

###########
# CV2 Utils
###########

def draw_bbox(frame, x1, y1, x2, y2, label="", color=(255, 0, 0)):
  cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

  # Label with background
  draw_label(frame, label, x1, y1, color)

def draw_label(frame, label, x1, y1, bg_color, font_scale=1.2, thickness=1):
  font = cv2.FONT_HERSHEY_SIMPLEX
  (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

  cv2.rectangle(frame, (x1 - 2, y1), (x1 + label_w + 20, y1 - label_h - 20), bg_color, cv2.FILLED)
  cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255))

def draw_progressbar(frame, progress, origin=(50,50), size=(600, 30), progress_color=(0, 255, 0), bg_color=(0, 0, 255)):
  x1, y1 = origin
  w, h = size
  x2, y2 = x1 + w, y1 + h

  # Draw background
  cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, cv2.FILLED)

  # Draw progress
  x_p = int((x2-x1)*progress)
  cv2.rectangle(frame, (x1, y1), (x1 + x_p, y2), progress_color, cv2.FILLED)

############
# Image utils
############

def load_image(path):
  img = cv2.imread(path, cv2.IMREAD_COLOR)
  # OpenCV loads images with color channels in BGR order. So we need to reverse them
  return convert_image(img)

def convert_image(img):
  img = img[...,::-1]
  img = np.around(img/255.0, decimals=12)
  # img = (img / 255.).astype(np.float32)
  return img

###############
# Embedding utils
###############

def write_embeddings(path, data):
  with h5py.File(path, 'w') as hf:
    hf.create_dataset('data',  data=data)

def read_embeddings(path):
  with h5py.File(path, 'r') as hf:
    return hf['data'][:]