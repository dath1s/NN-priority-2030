import os
import shutil
import time

from get_cam_img import getCamImg

from ML_model.model import Model

delay = 3600  # delay in seconds
i = 0

while True:
    filename = f"images/get/img{i}.jpg"
    out_filename = f"images/pred/img{i}.jpg"

    custom = Model("runs/detect/train/weights/best.pt")
    getCamImg(filename)
    custom.predict(filename)
    shutil.move(f"runs/detect/predict/img{i}.jpg", out_filename)
    os.rmdir("runs/detect/predict")
    i += 1
    time.sleep(delay)
