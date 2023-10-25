
# This is client code to receive video frames over UDP
import cv2
import imutils
import socket
import numpy as np
import time
import base64

from collections import defaultdict
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
track_history = defaultdict(lambda: [])

BUFF_SIZE = 65536
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
host_name = socket.gethostname()
host_ip = '192.168.50.38'  # socket.gethostbyname(host_name)
print(host_ip)
port = 9999
message = b'Hello'

client_socket.sendto(message, (host_ip, port))
fps, st, frames_to_count, cnt = (0, 0, 20, 0)
while True:
    packet, _ = client_socket.recvfrom(BUFF_SIZE)
    data = base64.b64decode(packet, ' /')
    npdata = np.fromstring(data, dtype=np.uint8)
    frame = cv2.imdecode(npdata, 1)
    frame = cv2.putText(frame, 'FPS: '+str(fps), (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    results = model.track(frame, persist=True, classes=0)
    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()
    annotated_frame = results[0].plot()

    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))
        if len(track) > 30:
            track.pop(0)

        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False, color=(
            230, 230, 230), thickness=10)

    cv2.imshow("YOLOv8 Tracking", annotated_frame)
    # cv2.imshow("RECEIVING VIDEO", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        client_socket.close()
        break
    if cnt == frames_to_count:
        try:
            fps = round(frames_to_count/(time.time()-st))
            st = time.time()
            cnt = 0
        except:
            pass
    cnt += 1
