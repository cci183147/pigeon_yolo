import os
import json
import shutil
import random
from glob import glob

# -------------------------------
# 配置你的路径
# -------------------------------
ANNOT_DIR = "metaXG/metadataXG/anotations"
IMG_DIR = "images_all"           # 解压 1~12.zip 后放在这里
OUT_DIR = "pigeon_iris_yolo"

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# -------------------------------
# 创建目录结构
# -------------------------------
os.makedirs(f"{OUT_DIR}/images/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/images/val", exist_ok=True)
os.makedirs(f"{OUT_DIR}/images/test", exist_ok=True)

os.makedirs(f"{OUT_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{OUT_DIR}/labels/val", exist_ok=True)
os.makedirs(f"{OUT_DIR}/labels/test", exist_ok=True)

json_files = glob(os.path.join(ANNOT_DIR, "*.json"))
random.shuffle(json_files)

# 划分数据
N = len(json_files)
train_end = int(N * train_ratio)
val_end = int(N * (train_ratio + val_ratio))

splits = {
    "train": json_files[:train_end],
    "val": json_files[train_end:val_end],
    "test": json_files[val_end:]
}

# -------------------------------
# 转换 json → YOLO 格式
# -------------------------------
def convert_bbox(x1, y1, x2, y2, w, h):
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh

for split, files in splits.items():
    print(f"\nProcessing {split}, total {len(files)}...")

    for js in files:
        with open(js, "r") as f:
            data = json.load(f)

        img_name = data["img"]
        img_path = os.path.join(IMG_DIR, img_name)

        if not os.path.exists(img_path):
            print(f"Warning: image not found: {img_path}")
            continue

        # 加载宽高
        w = data["weidth"]
        h = data["height"]
        bbs = data["bbs"]

        # ---------------------------
        # 写 YOLO txt
        # ---------------------------
        label_path = f"{OUT_DIR}/labels/{split}/{img_name.replace('.jpg','.txt')}"
        with open(label_path, "w") as yolo_f:
            for bb in bbs:
                x1,y1,x2,y2 = bb["bbx"]
                cx,cy,bw,bh = convert_bbox(x1,y1,x2,y2,w,h)
                yolo_f.write(f"0 {cx} {cy} {bw} {bh}\n")

        # ---------------------------
        # 拷贝图片到对应目录
        # ---------------------------
        shutil.copy(img_path, f"{OUT_DIR}/images/{split}/{img_name}")

