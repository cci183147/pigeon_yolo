from ultralytics import YOLO
model = YOLO("/home/cci/pigeon/ultralytics/runs/detect/train2/weights/best.pt")
model.predict(
    source="/home/cci/pigeon/datasets/images_all/916.jpg",
    save=True,
    show=False,
    conf=0.25,
    iou=0.45,
)