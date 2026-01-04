import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read frame")

cv2.imwrite("test_frame.jpg", frame)
print("Saved test_frame.jpg")

cap.release()
