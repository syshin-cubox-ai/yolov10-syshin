from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

model.export(format="onnx", simplify=True, device="cuda:0")
model.export(format="openvino", half=True)
