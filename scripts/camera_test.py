import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # auto
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)

time.sleep(1)  # let exposure settle

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read frame")

cv2.imwrite("test_frame.jpg", frame)
print("Saved test_frame.jpg")

cap.release()

