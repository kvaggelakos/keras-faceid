import cv2
import numpy as np
import time
import argparse
import os


def capture(output_dir, length):
  cap = cv2.VideoCapture(0)

  # Check if camera opened successfully
  if (cap.isOpened()== False):
    print("Error opening video stream or file")


  start_time = time.time()
  frames_counter = 0
  image_counter = 0

  while(cap.isOpened() and (time.time() - start_time) < length):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

      # Display the resulting frame
      cv2.imshow('Frame',frame)

      # Capture image every 5 frames
      if frames_counter % 5 == 0:
        file_path = os.path.join(output_dir, str(image_counter) + '.jpg')
        print('Writing file: ' + file_path)
        cv2.imwrite(file_path, frame)
        image_counter += 1
      # Press Q on keyboard to  exit
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break

      frames_counter += 1
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
  parser.add_argument("--length",
                      help="How many seconds to capture for",
                      dest="length",
                      default=5)
  args = parser.parse_args()

  # Create output path if it doesn't exist
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  capture(args.output_dir, args.length)


