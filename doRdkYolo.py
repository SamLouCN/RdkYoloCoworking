import cv2
import numpy as np
from ultralytics_yolo_det import UltralyticsYOLODetect, UltralyticsYOLODetectConfig

config = UltralyticsYOLODetectConfig(
    model_path="models/08182351_best.bin",
    classes_num=4,
    score_thres=0.3,
    nms_thres=0.45,
    reg=16,
    resize_type=1,
    strides=[8, 16, 32]
)

detector = UltralyticsYOLODetect(config)
detector.set_scheduling_params(priority=0, bpu_cores=[0])

with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    boxes, scores ,cls_ids = detector.predict(frame)

    for box, score, cls_id in zip(boxes, scores, cls_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[cls_id]}: {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
