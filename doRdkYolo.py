import cv2
import numpy as np
import hbm_runtime

runtime = hbm_runtime.HB_HBMRuntime("models/08182351_best.bin")
model_name = runtime.model_names[0]

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
    nv12_data = bgr_to_nv12(frame, (736, 736))
    outputs = infer.forward([nv12_data])

    detections = parse_yolo_output(outputs, conf_threshold=0.3)

    for box in detections:
        x1, y1, x2, y2, conf, cls_id = box
        label = f"{class_names[cls_id]}: {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
