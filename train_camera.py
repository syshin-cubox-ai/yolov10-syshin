import argparse

from ultralytics import YOLOv10

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    opt = parser.parse_args()

    model = YOLOv10("yolov10n.pt")

    model.train(data="camera.yaml", epochs=opt.epochs, imgsz=640, batch=128, optimizer="SGD",
                cache=False, device=[0, 1, 2, 3, 4, 5, 6, 7], workers=12, plots=True)
