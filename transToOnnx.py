from ultralytics import YOLO

model = YOLO('D:\\Projects\\RdkYolo\\models\\08182351_best.pt')

model.export(
    format = 'onnx',
    imgsz = 720,
    opset = 11,
    simplify = True,
    dynamic = False,
    batch = 1
)