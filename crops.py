# yolov11/batch_crop.py
from ultralytics import YOLO
import cv2, os, pandas as pd, tqdm

MODEL = "/home/cci/pigeon/ultralytics/runs/detect/train2/weights/best.pt"   # 或导出后的路径
IMG_DIR = "/home/cci/pigeon/datasets/images_all"
OUT_DIR = "/home/cci/pigeon/datasets/crops"
MAP_CSV = "/home/cci/pigeon/datasets/crop2img.csv"
os.makedirs(OUT_DIR, exist_ok=True)

model = YOLO(MODEL)
rows = []
for img_fn in tqdm.tqdm(os.listdir(IMG_DIR)):
    img_path = os.path.join(IMG_DIR, img_fn)
    try:
        res = model.predict(source=img_path, conf=0.25, iou=0.45, verbose=False)[0]
    except Exception as e:
        print("error", img_path, e); continue
    img = cv2.imread(img_path)
    if img is None: continue
    for i, box in enumerate(res.boxes.xyxy.tolist()):
        x1, y1, x2, y2 = map(int, box)
        score = float(res.boxes.conf[i]) if hasattr(res.boxes, 'conf') else None
        crop = img[y1:y2, x1:x2]
        if crop.size == 0: continue
        outname = f"{os.path.splitext(img_fn)[0]}_{i}.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, outname), crop)
        rows.append([outname, img_fn, i, x1, y1, x2, y2, score])

pd.DataFrame(rows, columns=["crop","src_image","box_idx","x1","y1","x2","y2","score"]).to_csv(MAP_CSV, index=False)
print("Saved crops:", len(rows))
