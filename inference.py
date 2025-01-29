from ultralytics import YOLO

model = YOLO('/content/runs/detect/train2/weights/best.pt')
model('/content/20250122_180913.jpg', save=True)