import cv2
import numpy as np
import time
import argparse
import os

from face_detector import FaceDetector


def capture(output_dir, count):
  cap = cv2.VideoCapture(0)

  # Check if camera opened successfully
  if (cap.isOpened()== False):
    print("Error opening video stream or file")


  captured_counter = 0
  face_detector = FaceDetector()

  while(cap.isOpened() and captured_counter < count):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

      faces = face_detector.detect_faces(frame)
      if len(faces) > 0:
        file_path = os.path.join(output_dir, str(captured_counter) + '.jpg')
        print('Writing capture: ' + file_path)
        x, y, w, h = faces[0] # Assume it's the only face
        cropped = frame[y:y+h, x:x+w]
        cv2.imwrite(file_path, cropped)
        captured_counter += 1

      cv2.imshow('Frame', frame)

      # Press Q on keyboard to  exit
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # Break the loop
    else:
      break

  # When everything done, release the video capture object
  cap.release()

  # Closes all the frames
  cv2.destroyAllWindows()


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--output",
                      help="set the output dir",
                      dest="output_dir",
                      default="./data/capture/")
  parser.add_argument("--count",
                      help="How many images to capture",
                      dest="count",
                      default=10)
  args = parser.parse_args()

  # Create output path if it doesn't exist
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  capture(args.output_dir, args.count)


