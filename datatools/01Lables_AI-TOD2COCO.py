import os
import json
import cv2
from tqdm import tqdm


def yolo_to_coco(yolo_dir, output_json_path, classes):
    """
    将 YOLO 格式的标注文件转换为 COCO 格式
    Args:
        yolo_dir: YOLO 数据集根目录 (例如 D:\\Datas\\AI-TODyolo\\train)
        output_json_path: 输出的 COCO 标注文件路径
        classes: 类别列表
    """
    images_dir = os.path.join(yolo_dir, "images")
    labels_dir = os.path.join(yolo_dir, "labels")

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    coco_images = []
    coco_annotations = []
    ann_id = 0  # annotation id counter

    print(f"Converting {yolo_dir} to COCO format...")

    for idx, image_file in enumerate(tqdm(image_files)):
        # 获取图片基本信息
        image_path = os.path.join(images_dir, image_file)
        height, width = cv2.imread(image_path).shape[:2]

        image_id = idx
        coco_images.append({
            "id": image_id,
            "file_name": image_file,
            "height": height,
            "width": width
        })

        # 读取对应的标注文件
        label_path = os.path.join(labels_dir, os.path.splitext(image_file)[0] + ".txt")
        if not os.path.exists(label_path):
            continue  # 没有标注文件则跳过

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            cls_id, x_center, y_center, w, h = map(float, line.strip().split())
            cls_id = int(cls_id)

            # 将归一化坐标转为像素坐标
            x_center *= width
            y_center *= height
            w *= width
            h *= height

            x = x_center - w / 2
            y = y_center - h / 2

            coco_annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cls_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id += 1

    # 构造 COCO 数据集结构
    coco_format = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": i, "name": name} for i, name in enumerate(classes)]
    }

    # 保存为 JSON 文件
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f, indent=4)

    print(f"Saved COCO annotation file to: {output_json_path}")


if __name__ == "__main__":
    # === 1. 类别定义 ===
    AI_TOD_CLASSES = [
        "car",
        "truck",
        "bus",
        "van",
        "pedestrian",
        "cyclist",
        "tricycle",
        "traffic_light",
        "traffic_sign",
        "pole"
    ]

    # === 2. 数据集路径 ===
    root = r"D:\Datas\AI-TODyolo"
    output_dir = os.path.join(root, "coco")

    # === 3. 转换 train / val / test 三个子集 ===
    splits = ["train", "val", "test"]
    for split in splits:
        yolo_path = os.path.join(root, split)
        output_json = os.path.join(output_dir, f"AI-TOD_{split}_coco.json")
        yolo_to_coco(yolo_path, output_json, AI_TOD_CLASSES)
