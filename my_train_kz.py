from ultralytics import YOLO
import os
from pathlib import Path

if __name__ == '__main__':
# Load a model
#     model = YOLO(r"D:\Models\yolo11n.pt")
    model = YOLO(r"D:\codes\ultralytics-main\ultralytics\cfg\models\11\myyolo11.yaml")
    print(model)

    # Train the model
    train_results = model.train(
        data=r"D:\codes\ultralytics-main\my_kz.yaml",  # path to dataset YAML
        epochs=100,  # number of training epochs
        imgsz=640,  # training image size
        device="cuda",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        amp=False,
    )

    # Evaluate model performance on the validation set
    metrics = model.val()

    # Perform object detection on an image
    # results = model("path/to/image.jpg")
    # results[0].show()

    # Export the model to ONNX format
    path = model.export(format="onnx")  # return path to exported model




