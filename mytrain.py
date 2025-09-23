from ultralytics import YOLO
import os
from pathlib import Path

if __name__ == '__main__':
# Load a model
#     model = YOLO(r"D:\Models\yolo11n.pt")
#     model = YOLO(r"D:\codes\ultralytics-main\ultralytics\cfg\models\11\myyolo11.yaml")
    model = YOLO(r"D:\codes\ultralytics-main\ultralytics\cfg\models\v3\yolov3.yaml")



    print(model)

    # def visdrone2yolo(dir):
    #     from PIL import Image
    #     from tqdm import tqdm
    #
    #     def convert_box(size, box):
    #         # Convert VisDrone box to YOLO xywh box
    #         dw = 1. / size[0]
    #         dh = 1. / size[1]
    #         return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh
    #
    #     (dir / 'labels').mkdir(parents=True, exist_ok=True)  # make labels directory
    #     # (os.path.join(dir,'labels')).mkdir(parents=True, exist_ok=True)  # make labels directory
    #     # pbar = tqdm((os.path.join(dir,'annotations')).glob('*.txt'), desc=f'Convertisng {dir}')
    #     pbar = tqdm((dir / 'annotations').glob('*.txt'), desc=f'Converting {dir}')
    #     for f in pbar:
    #         # img_size = Image.open((os.path.join(dir,'images',f.name)).with_suffix('.jpg')).size
    #         img_size = Image.open((dir / 'images' / f.name).with_suffix('.jpg')).size
    #         lines = []
    #         with open(f, 'r') as file:  # read annotation.txt
    #             for row in [x.split(',') for x in file.read().strip().splitlines()]:
    #                 if row[4] == '0':  # VisDrone 'ignored regions' class 0
    #                     continue
    #                 cls = int(row[5]) - 1
    #                 box = convert_box(img_size, tuple(map(int, row[:4])))
    #                 lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
    #                 with open(str(f).replace(f'{os.sep}annotations{os.sep}', f'{os.sep}labels{os.sep}'), 'w') as fl:
    #                     fl.writelines(lines)  # write label.txt

    dir=r"D:\Datas\VisDrone"
    dir = Path(dir)
    train_results = model.train(
        data=r"D:\codes\ultralytics-main\myVisDrone.yaml",  # path to dataset YAML
        epochs=100,  # number of training epochs
        imgsz=640,  # training image size
        device="cuda",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        amp=False,
# amp=True,
    )

    # Evaluate model performance on the validation set
    metrics = model.val()

    # Perform object detection on an image
    # results = model("path/to/image.jpg")
    # results[0].show()

    # Export the model to ONNX format
    path = model.export(format="onnx")  # return path to exported model


