from ultralytics import YOLO
import cv2
import time
import numpy as np

model = YOLO('yolov8n.pt')
# with segmentation (uncomment those lines)
model1 = YOLO("yolov8n-seg.pt")

# result = model.track(source=0, show=True, tracker="bytetrack.yaml", classes=0)

cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=False,
                              tracker="bytetrack.yaml", classes=0)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # with segmentation-
        results1 = model1(annotated_frame, classes=0)

        annotated_frame = results1[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
