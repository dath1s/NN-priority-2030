import os

from ultralytics import YOLO
from ultralytics.engine.results import Results


class Model:
    def __init__(self, model_path: str, cuda_enable: bool = False) -> None:
        self.model: YOLO = YOLO(model_path)
        if cuda_enable:
            self.model.to("cuda")

    @staticmethod
    def create_yaml(names: list[str], nc: int, path: str, train: str, val: str) -> None:
        with open("dataset.yaml", mode="w+") as YAMLfile:
            join_str = "\n- "
            YAMLfile.write(
                f"names:\n- {join_str.join(names)}\n"
                f"nc : {nc}\n"
                f"path: {path}\n"
                f"train: {train}\n"
                f"val: {val}\n"
            )

    def train(self, data: str, epochs: int = 100, imgsz: int = 640, batch: int = 16) -> dict | None:
        if "dataset.yaml" in os.listdir():
            return self.model.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch)
        raise Exception("File 'dataset.yaml' does not exists. Use method Model().createYAML()")

    def predict(self, filepath: str) -> list[Results]:
        return self.model.predict(save=True, conf=0.5, source=filepath)
