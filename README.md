### 下载本项目
`git clone https://github.com/cci183147/pigeon_yolo`
`uv sync`
#### 数据处理
`python unzip.py`解压图片文件1~12.zip,并统一置于images_all中
`python yoloData.py` 将数据处理为yolo格式，输出于pigeon_iris_yolo
### Yolo模型训练
本项目使用**Yolov11n.pt**训练，请提前准备好
`python train.py` 参数如下:
	epochs=10,
    imgsz=640,
    batch=-1,
    cache=False,
    workers=0,`
得到模型**best.pt**
`python check.py` 处理错误图片
`python crops.py ` 获得虹膜切片，存放于crops中
`python crops2blood.py` 建立blood_id和对应切片的映射关系
`python crops2blood_clean` 清洗null值，得到crops_metadata_clean.csv

### Siamese embedding
##### 数据处理
`python pairs.py`得到pairs.csv作为模型训练数据
##### 模型训练
`python siamese/train.py`
模型参数如下：
	`EMB_SIZE = 128
	`BACKBONE = 'resnet50'   # change to resnet50 if GPU enough
	`BATCH = 32
	`EPOCHS = 10
	`LR = 1e-4
	`WEIGHT_DECAY = 1e-5
	`VAL_SPLIT = 0.1
	`MARGIN = 1.0 
用训练好的模型对所有图像生成 embedding
`python siamese/embed_all.py`
构建 FAISS 索引 — `retrieval/build_index.py`
### 使用效果
