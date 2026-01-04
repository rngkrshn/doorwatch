import cv2

for i in range(0, 10):
    cap = cv2.VideoCapture(i)
    ok = cap.isOpened()
    ret, frame = cap.read() if ok else (False, None)
    print(i, "opened" if ok else "no", "frame" if ret else "no-frame")
    if ret:
        cv2.imwrite(f"scan_{i}.jpg", frame)
    cap.release()
