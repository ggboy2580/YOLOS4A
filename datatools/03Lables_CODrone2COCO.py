import os
import json
import cv2
from tqdm import tqdm


def yolo_to_coco(yolo_dir, output_json_path, classes):
    """
    将 YOLO 检测标注格式转换为 COCO 格式
    Args:
        yolo_dir: 数据集子集路径 (例如 D:\\Datas\\CODrone\\train)
        output_json_path: 输出的 COCO JSON 文件路径
        classes: 类别名称列表
    """
    images_dir = os.path.join(yolo_dir, "images") if os.path.exists(os.path.join(yolo_dir, "images")) else yolo_dir
    labels_dir = os.path.join(yolo_dir, "labels") if os.path.exists(os.path.join(yolo_dir, "labels")) else yolo_dir

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    coco_images, coco_annotations = [], []
    ann_id = 0

    print(f"Converting {yolo_dir} to COCO format...")

    for idx, image_file in enumerate(tqdm(image_files)):
        image_path = os.path.join(images_dir, image_file)
        if not os.path.exists(image_path):
            continue

        # 获取图片尺寸
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: cannot read image {image_path}, skipped.")
            continue
        height, width = img.shape[:2]

        image_id = idx
        coco_images.append({
            "id": image_id,
            "file_name": image_file,
            "height": height,
            "width": width
        })

        # 对应的YOLO标签文件
        label_path = os.path.join(labels_dir, os.path.splitext(image_file)[0] + ".txt")
        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id, x_center, y_center, w, h = map(float, parts[:5])
            cls_id = int(cls_id)
            if cls_id >= len(classes):
                print(f"Warning: class id {cls_id} exceeds category list, skipped.")
                continue

            # 归一化坐标转为像素坐标
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

    # 构造COCO数据结构
    coco_dict = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": i, "name": name} for i, name in enumerate(classes)]
    }

    # 写入JSON文件
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f, indent=4)

    print(f"✅ COCO annotations saved to: {output_json_path}")


if __name__ == "__main__":
    # === CODrone 类别定义 ===
    CODRONE_CLASSES = [
        "car",
        "ignored",
        "traffic-sign",
        "people",
        "motor",
        "truck",
        "bus",
        "bicycle",
        "boat",
        "tricycle",
        "traffic-light",
        "ship",
        "bridge"
    ]

    # === 数据集路径配置 ===
    root = r"D:\Datas\CODrone"
    output_dir = os.path.join(root, "coco")

    # === 转换 train / val / test 三个子集 ===
    splits = ["train", "val", "test"]
    for split in splits:
        split_path = os.path.join(root, split)
        if not os.path.exists(split_path):
            print(f"⚠️  Skipping {split} (path not found)")
            continue
        output_json = os.path.join(output_dir, f"CODrone_{split}_coco.json")
        yolo_to_coco(split_path, output_json, CODRONE_CLASSES)
