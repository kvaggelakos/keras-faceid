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
from model import build_model

# Builds an average embedding based on all images captured
def build_embedding(model, path):
  print('Building embedding and saving to disk...')
  image_paths = glob.glob('{}/*.jpg'.format(path))
  embedding = np.zeros((len(image_paths), 128))
  for i, image_path in enumerate(image_paths):
    img = load_image(image_path)
    embedding[i] = model.predict(np.array([img]))
    sys.stdout.write('.')
    sys.stdout.flush()

  # TODO: Avg embedding?

  write_embeddings(os.path.join(path, 'embedding.h5'), embedding)

def capture(args):
  cap = cv2.VideoCapture(0)

  # Check if camera opened successfully
  if (cap.isOpened()== False):
    print("Error opening video stream or file")


  captured_counter = 0
  face_detector = FaceDetector()
  model = build_model()

  while(cap.isOpened() and captured_counter < args.count):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

      # Show progress bar
      draw_progressbar(frame, (captured_counter / args.count))

      # Detect image and write it
      faces = face_detector.detect_faces(frame)
      if len(faces) > 0:
        file_path = os.path.join(args.output_dir, str(captured_counter) + '.jpg')
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

  # Write the embedding file
  build_embedding(model, args.output_dir)


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--output",
                      help="Sets the output dir",
                      dest="output_dir",
                      default="./data/")
  parser.add_argument("--count",
                      help="The number of images to capture",
                      dest="count",
                      type=int,
                      default=10)
  args = parser.parse_args()

  # Create output path if it doesn't exist
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  capture(args)


