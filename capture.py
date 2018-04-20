import cv2
import numpy as np
import time
import argparse
import os
import glob
import sys

from PIL import Image

from face_detector import FaceDetector
from utils import draw_bbox, draw_progressbar, load_image, convert_image, write_embeddings
from model import build_model, build_classifier

# Builds and writes embedding to file
def build_embedding(model, named_path):
  print('Building embedding and saving to disk...')
  image_paths = glob.glob('{}/*.jpg'.format(named_path))
  embedding = np.zeros((len(image_paths), 128))
  for i, image_path in enumerate(image_paths):
    img = load_image(image_path)
    embedding[i] = model.predict(np.array([img]))
    sys.stdout.write('.')
    sys.stdout.flush()

  write_embeddings(os.path.join(named_path, 'embedding.h5'), embedding)

def capture(named_path, data_path, count):
  cap = cv2.VideoCapture(0)

  # Check if camera opened successfully
  if (cap.isOpened()== False):
    print("Error opening video stream or file")


  captured_counter = 0
  face_detector = FaceDetector()
  model = build_model()

  while(cap.isOpened() and captured_counter < count):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

      # Show progress bar
      draw_progressbar(frame, (captured_counter / count))

      # Detect image and write it
      faces = face_detector.detect_faces(frame)
      if len(faces) > 0:

        # Per person path
        file_path = os.path.join(named_path, str(captured_counter + 1) + '.jpg')
        print('Writing capture: ' + file_path)

        face = faces[0] # Assume it's the only face
        x, y, w, h = face
        cropped = frame[y:y+h, x:x+w]
        cropped = cv2.resize(cropped, (96, 96))
        cv2.imwrite(file_path, cropped)
        captured_counter += 1
        draw_bbox(frame, x, y, x+w, y+h, label="Face detected")

      cv2.imshow('Frame', frame)

      # Press Q on keyboard to  exit
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # Break the loop
    else:
      break


  # When everything done, release the video capture object
  cap.release()
  cv2.destroyAllWindows()

  # Build and Write the embedding file for this person
  build_embedding(model, named_path)

  # Rebuild the classifier
  build_classifier(data_path)

  print('Done!')


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--data",
                      help="Sets the data dir",
                      dest="data_path",
                      default="./data/")
  parser.add_argument("--count",
                      help="The number of images to capture",
                      dest="count",
                      type=int,
                      default=10)
  parser.add_argument("--name",
                      help="The name of the person",
                      dest="name",
                      default="person")
  args = parser.parse_args()

  # Create output path if it doesn't exist

  named_path = os.path.join(args.data_path, args.name)

  if not os.path.exists(named_path):
    os.makedirs(named_path)

  capture(named_path, args.data_path, args.count)


