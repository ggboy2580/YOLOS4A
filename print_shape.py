from ultralytics.nn.tasks import DetectionModel

if __name__ == "__main__":
    # 加载YOLO模型（配置文件或预训练权重）
    MODEL_PATH = r"D:\codes\ultralytics-main\ultralytics\cfg\models\12\yolo12-4head.yaml"  # 你的配置文件路径
    MODEL_PATH = (
        r"D:\codes\ultralytics-main\ultralytics\cfg\models\S4A\yolov8-SMALL-4head-SE-CA-ECA.yaml"  # 你的配置文件路径
    )

    DetectionModel(MODEL_PATH)
