from ultralytics import YOLOv10

# model = YOLOv10("yolov8c-pose.yaml")
model = YOLOv10("runs/detect/train/weights/last.pt")
# model = YOLOv10("runs/pose/train/weights/last.onnx")
# model = YOLOv10("runs/detect/train/weights/last_openvino_model")
# model = YOLOv10("runs/pose/train/weights/last_int8_openvino_model")

results = model.predict(0, show=True, line_width=2, conf=0.6, device="cuda")
# results = model.predict("C:/Users/synml/Desktop/images", save=True, device="cpu", line_width=2)
# results = model.predict("ultralytics/assets/largest_selfie.jpg", save=True, line_width=2)
