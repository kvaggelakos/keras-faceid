import cv2


def draw_bbox(frame, x1, y1, x2, y2, label="", color=(255, 0, 0)):
  cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
  cv2.putText(frame, label, (x1, y1 - 10), 0, 0.5, color)

def draw_progressbar(frame, progress, origin=(50,50), size=(600, 30), progress_color=(0, 255, 0), bg_color=(0, 0, 255)):
  x1, y1 = origin
  w, h = size
  x2, y2 = x1 + w, y1 + h

  # Draw background
  cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, cv2.FILLED)

  # Draw progress
  x_p = int((x2-x1)*progress)
  cv2.rectangle(frame, (x1, y1), (x1 + x_p, y2), progress_color, cv2.FILLED)