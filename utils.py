import cv2


def draw_bbox(frame, x1, y1, x2, y2, label="", color=(255, 0, 0)):
  cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
  cv2.putText(frame, label, (x1, y1 - 10), 0, 0.5, color)