import sys
import numpy as np
import cv2
import argparse
from PIL import Image

from face_detector import FaceDetector
from utils import draw_bbox, draw_label, read_embeddings, convert_image
from model import build_model

def run(args):
  model = build_model()
  embedding = np.average(read_embeddings(args.emb_path), axis=0)

  cap = cv2.VideoCapture(0)

  if (cap.isOpened() == False):
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
        if args.debug:
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
  parser.add_argument("--emb-path",
                      help="The path to the embedding pickle",
                      dest="emb_path",
                      default='./data/embedding.h5')

  args = parser.parse_args()

  run(args)