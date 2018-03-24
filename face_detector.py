import os
import cv2


class FaceDetector():

  def __init__(self):
    self.face_cascade = self._build_face_cascade()

  def _build_face_cascade(self):
    full_path = os.path.join(os.path.dirname(__file__), 'opencv', 'haarcascade_frontalface_default.xml')
    return cv2.CascadeClassifier(full_path)

  def detect_faces(self, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces