from ML_model.model import Model

model = Model("yolov8x.pt", cuda_enable=False)

Model.create_yaml(
    names=[
        "big_fully_ripened",
        "big_half_ripened",
        "big_green",
        "small_fully_ripened",
        "small_half_ri+-pened",
        "small_green",
    ],
    nc=6,
    path="../data",
    train="train/images",
    val="val/images",
)

results = model.train(data="dataset.yaml", epochs=75, imgsz=640, batch=16)
