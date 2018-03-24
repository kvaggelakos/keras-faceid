import sys
import glob
import numpy as np
import cv2
import argparse
from PIL import Image

from openface.model import create_model
from face_detector import FaceDetector
from utils import draw_bbox, draw_label

def build_model():
  model = create_model()
  model.load_weights('./openface/weights/nn4.small2.v1.h5')
  return model

def load_image(path):
  img = cv2.imread(path, cv2.IMREAD_COLOR)
  # OpenCV loads images with color channels in BGR order. So we need to reverse them
  return convert_image(img)

def convert_image(img):
  img = img[...,::-1]
  img = np.around(img/255.0, decimals=12)
  # img = (img / 255.).astype(np.float32)
  return img

# Builds an average embedding based on all images captured
def build_embedding(model):
  image_paths = glob.glob('./data/capture/*.jpg')
  embedding = np.zeros((len(image_paths), 128))
  for i, image_path in enumerate(image_paths):
    img = load_image(image_path)
    embedding[i] = model.predict(np.array([img]))
    sys.stdout.write('.')
    sys.stdout.flush()

  return np.average(embedding, axis=0)
  # return embedding[1]

def run(debug):
  model = build_model()
  embedding = build_embedding(model)

  cap = cv2.VideoCapture(0)

  # Check if camera opened successfully
  if (cap.isOpened()== False):
    print("Error opening video stream or file")

  face_detector = FaceDetector()

  while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

      # Detect image and write it
      faces = face_detector.detect_faces(frame)
      for face in faces:
        x, y, w, h = face
        cropped = frame[y:y+h, x:x+w]
        cropped = cv2.resize(cropped, (96, 96))
        cropped = np.around(convert_image(cropped), decimals=12)
        encoding = model.predict(np.array([cropped]))
        dist = np.linalg.norm(encoding - embedding)

        # Draw the box around the face
        if (dist < 0.7):
          draw_bbox(frame, x, y, x+w, y+h, label="It's you!", color=(0, 255, 0))
        else:
          draw_bbox(frame, x, y, x+w, y+h, label="A face")

        # Draw distance label for debugging
        if debug:
          draw_label(frame, "D: " + str(round(dist, 2)), frame.shape[0] - 50, 100, (255, 0, 0), font_scale=2.5, thickness=2)

      cv2.imshow('Frame', frame)

      # Press Q on keyboard to  exit
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    else:
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--debug",
                      help="Show debug outputs on the screen",
                      dest="debug",
                      action='store_true',
                      default=False)
  args = parser.parse_args()

  run(args.debug)