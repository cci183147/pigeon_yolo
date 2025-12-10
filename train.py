from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(
    data="/home/cci/pigeon/datasets/pigeon_iris_yolo/data.yaml",
    epochs=10,
    imgsz=640,
    batch=-1,
    cache=False,
    workers=0,
)