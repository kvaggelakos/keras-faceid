# import cv2
# import numpy as np
# import time
# import argparse
# import os

# from face_detector import FaceDetector
# from utils import draw_bbox

# def capture():
#   cap = cv2.VideoCapture(0)

#   # Check if camera opened successfully
#   if (cap.isOpened()== False):
#     print("Error opening video stream or file")


#   face_detector = FaceDetector()

#   while(cap.isOpened()):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if ret == True:

#       # Find the face
#       faces = face_detector.detect_faces(frame)
#       for (x,y,w,h) in faces:
#         draw_bbox(frame, x, y, x+w, y+h, label="Face detected")

#       cv2.imshow('Frame', frame)

#       # Press Q on keyboard to  exit
#       if cv2.waitKey(25) & 0xFF == ord('q'):
#         break
#     # Break the loop
#     else:
#       break

#   # When everything done, release the video capture object
#   cap.release()

#   # Closes all the frames
#   cv2.destroyAllWindows()


# if __name__ == "__main__":
#   capture()


