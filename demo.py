import sys
import numpy as np
import cv2
import argparse
import glob
import os
from PIL import Image
import sklearn

from face_detector import FaceDetector
from utils import draw_bbox, draw_label, read_classifier, convert_image, read_only_embedding
from model import build_model

def run(args):
  model = build_model()

  (clf, class_names) = read_classifier(os.path.join(args.data_path, 'classifier.pickle'))
  # if classifier is none we only have one face
  if clf is None:
    verified_embedding, only_class = read_only_embedding(args.data_path)

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
        embedding = model.predict(np.array([cropped]))

        if clf is None:
          dist = np.linalg.norm(verified_embedding - embedding)
          match = dist < 0.7
          label = only_class if match else "Unknown"
          if args.debug:
            label += ' (d: {})'.format(round(dist, 2))
        else:
          predictions = clf.predict_proba(embedding)
          pred_class = np.argmax(predictions, axis=1)[0]
          score = round(np.max(predictions)*100, 2)
          match = score > 70
          name = class_names[pred_class]
          label = '{} ({}%)'.format(name, score)

        color = (0, 255, 0) if match else (0, 0, 255)

        draw_bbox(frame, x, y, x+w, y+h, label=label, color=color)

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
  parser.add_argument("--data-path",
                      help="The path to where data is stored from the capture",
                      dest="data_path",
                      default='./data')

  args = parser.parse_args()

  run(args)