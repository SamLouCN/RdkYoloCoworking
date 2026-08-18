from ultralytics import YOLO
import torch
torch.backends.cudnn.enabled = False

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available}")
if torch.cuda.is_available:
    print(f"Graphics Card: {torch.cuda.get_device_name(1)}")
else:
    print(f"Using CPU for training.")
model = YOLO('yolov8n.pt')

if __name__ == '__main__':
    results = model.train(
        data='./data.yaml',
        epochs=100,
        
        device=1,
        batch=48,
        workers=4,
        
        imgsz=720,
        rect=False,
        
        patience=50,
        seed=42,
        project='runs/detect',
        name='model',
        exist_ok=True,
        
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.5,   
    )
    
    print("\n\nFinished")
    val_results = model.val()
    print(f"Val mAP50-95: {val_results.box.map:.4f}")

