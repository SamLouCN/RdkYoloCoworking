import cv2
import numpy as np
import hbm_runtime

def bgr_to_nv12(frame, target_size):
    resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    yuv420 = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV_I420)
    height, width = target_size

    y_plane = yuv420[:height, :].reshape(-1)
    u_plane = yuv420[height:height + height//4, :].reshape(-1)
    v_plane = yuv420[height + height//4:, :].reshape(-1)

    uv_interleaved = np.empty((height * width // 2,), dtype=np.uint8)
    uv_interleaved[0::2] = u_plane
    uv_interleaved[1::2] = v_plane

    packed_nv12 = np.concatenate([y_plane, uv_interleaved])
    return packed_nv12.astype(np.uint8)

def nms_boxes(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def parse_yolo_output(outputs, ori_w, ori_h, conf_threshold=0.3, nms_threshold=0.45):
    if isinstance(outputs, dict):
        model_name = list(outputs.keys())[0]
        pred = outputs[model_name]['output0']
    else:
        pred = outputs[0]

    pred = np.transpose(pred.squeeze(0).squeeze(0), (1, 0))

    cx = pred[:, 0]
    cy = pred[:, 1]
    w = pred[:, 2]
    h = pred[:, 3]
    cls_logits = pred[:, 4:]

    cls_scores = 1.0 / (1.0 + np.exp(-cls_logits))
    max_scores = np.max(cls_scores, axis=1)
    cls_ids = np.argmax(cls_scores, axis=1)

    valid = max_scores > conf_threshold
    if not np.any(valid):
        return np.empty((0, 4)), np.empty(0), np.empty(0)
    
    cx_valid = cx[valid]
    cy_valid = cy[valid]
    w_valid = w[valid]
    h_valid = h[valid]
    scores = max_scores[valid]
    cls_ids = cls_ids[valid]
    
    scale_x = ori_w / input_w
    scale_y = ori_h / input_h

    cx_img = cx_valid * scale_x
    cy_img = cy_valid * scale_y
    w_img = w_valid * scale_x
    h_img = h_valid * scale_y

    x1 = cx_img - w_img/2
    y1 = cy_img - h_img/2
    x2 = cx_img + w_img/2
    y2 = cy_img + h_img/2
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    keep = nms_boxes(boxes, scores, nms_threshold)
    return boxes[keep], scores[keep], cls_ids[keep]

if __name__ == "__main__":
    
    runtime = hbm_runtime.HB_HBMRuntime("models/08182351_best.bin")
    model_name = runtime.model_names[0]
    input_name = runtime.input_names[model_name][0]
    
    with open("classes.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    input_h, input_w = 736,736

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        ori_h, ori_w = frame.shape[:2]
        nv12_data = bgr_to_nv12(frame, (input_w, input_h))
        
        inputs = {model_name: {input_name: nv12_data}}
        outputs = runtime.run(inputs)
        
        boxes, scores, cls_ids = parse_yolo_output(
            outputs, ori_w, ori_h,
            conf_threshold=0.3,
            nms_threshold=0.45
        )

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
