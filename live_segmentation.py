from ultralytics import YOLO
import cv2
import time
import numpy as np

# loading the model
model = YOLO("yolov8n-seg.pt")


# camera/vid-file
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


# loop through the video-frames
while cap.isOpened():
    # read a frame from vid
    success, frame = cap.read()
    if success:
        start = time.perf_counter()
        # remove classes to classify everything
        results = model(frame, classes=0)

        end = time.perf_counter()
        total_time = end-start
        fps = 1 / total_time

        # visualizing results on the frame
        annotated_frame = results[0].plot()

        # displaying it with text
        # object details
        org = [10, 20]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2

        cv2.putText(annotated_frame,
                    f"FPS: {int(fps)}", org, font, fontScale, color, thickness)

        cv2.imshow("YOLOV8 inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break  # end of vid is reachedd

cap.release()
cv2.destroyAllWindows()
