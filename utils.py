import cv2


def draw_bbox(frame, x1, y1, x2, y2, label="", color=(255, 0, 0)):
  cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

  # Label with background
  draw_label(frame, label, x1, y1, color)

def draw_label(frame, label, x1, y1, bg_color, font_scale=1.2, thickness=1):
  font = cv2.FONT_HERSHEY_SIMPLEX
  (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

  cv2.rectangle(frame, (x1 - 2, y1), (x1 + label_w + 20, y1 - label_h - 20), bg_color, cv2.FILLED)
  cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255))

def draw_progressbar(frame, progress, origin=(50,50), size=(600, 30), progress_color=(0, 255, 0), bg_color=(0, 0, 255)):
  x1, y1 = origin
  w, h = size
  x2, y2 = x1 + w, y1 + h

  # Draw background
  cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, cv2.FILLED)

  # Draw progress
  x_p = int((x2-x1)*progress)
  cv2.rectangle(frame, (x1, y1), (x1 + x_p, y2), progress_color, cv2.FILLED)