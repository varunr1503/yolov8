from ultralytics.models.sam import Predictor as SAMPredictor
from ultralytics import SAM
import cv2
import math
# Load a model
model = SAM('sam_b.pt')

# # Display model information (optional)
# model.info()

# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)


# while True:
#     success, img = cap.read()
#     results = model(img, stream=True)
#     annotated_frame = results[0].plot()
#     cv2.imshow('SAM model', annotated_frame)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# Create SAMPredictor
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
while True:
    success, img = cap.read()
    overrides = dict(conf=0.25, task='segment', mode='predict',
                     imgsz=1024, model="mobile_sam.pt")
    predictor = SAMPredictor(overrides=overrides)

    # Segment with additional args
    results = predictor(source=0, crop_n_layers=1,
                        points_stride=64, stream=True)
