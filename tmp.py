import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ultralytics import YOLO

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]


def visualize_segmentation(results, save_path=None):
    """
    可视化YOLO分割结果.

    参数:
        results: YOLO模型的预测结果
        save_path: 保存可视化结果的路径，None则不保存
    """
    # 遍历每张图片的结果
    for result in results:
        # 获取原图
        img = result.orig_img

        # 复制原图用于绘制
        visualized_img = img.copy()

        # 如果有分割掩码
        if result.masks is not None:
            # 获取掩码和类别
            masks = result.masks.data.cpu().numpy()
            result.boxes.cls.cpu().numpy().astype(int)

            # 为每个掩码生成随机颜色
            colors = np.random.randint(0, 255, size=(len(masks), 3), dtype=np.uint8)

            # 绘制每个掩码
            for i, mask in enumerate(masks):
                # 将掩码调整为与原图相同大小
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

                # 创建掩码的彩色版本
                color = colors[i]
                colored_mask = np.zeros_like(img, dtype=np.uint8)
                colored_mask[mask > 0.5] = color

                # 叠加掩码到原图上（半透明）
                visualized_img = cv2.addWeighted(visualized_img, 0.7, colored_mask, 0.3, 0)

        # 绘制边界框和类别标签
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # 边界框坐标
            confs = result.boxes.conf.cpu().numpy()  # 置信度
            clss = result.boxes.cls.cpu().numpy().astype(int)  # 类别

            for box, conf, cls_idx in zip(boxes, confs, clss):
                x1, y1, x2, y2 = map(int, box)
                class_name = result.names[cls_idx]

                # 绘制边界框
                cv2.rectangle(visualized_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 绘制类别和置信度标签
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(visualized_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 转换颜色空间（OpenCV默认BGR，Matplotlib需要RGB）
        visualized_img_rgb = cv2.cvtColor(visualized_img, cv2.COLOR_BGR2RGB)

        # 显示结果
        plt.figure(figsize=(10, 8))
        plt.imshow(visualized_img_rgb)
        plt.axis("off")
        plt.title("YOLO分割结果")
        plt.show()

        # 保存结果
        if save_path:
            # 创建保存目录
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # 转换回BGR格式保存
            visualized_img_bgr = cv2.cvtColor(visualized_img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, visualized_img_bgr)
            print(f"结果已保存至: {save_path}")


if __name__ == "__main__":
    # 加载模型 - 可以是配置文件或预训练权重文件
    # 注意：如果使用.yaml配置文件，需要确保已经训练过或指定了预训练权重
    MODEL_PATH = r"C:\Users\gaoge\Downloads\yolo11l-seg.pt"
    # 图像目录
    PIC_DIR = r"D:\BaiduSyncdisk\01bmpv\photos\大师作品\限制级\1 阿半今天很开心 NO.001 露背毛衣 [28P-157MB] 1"
    # 结果保存目录
    SAVE_DIR = "./segment_results"

    # 加载YOLO模型
    # 如果是首次使用，建议使用预训练模型如"yolov8n-seg.pt"进行测试
    model = YOLO(MODEL_PATH)

    # 确保保存目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 遍历图像目录
    for root, dirs, files in os.walk(PIC_DIR):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                pic_path = os.path.join(root, file)
                print(f"处理图像: {pic_path}")

                # 进行预测
                results = model(pic_path)

                # 生成保存路径
                save_name = f"seg_{os.path.basename(pic_path)}"
                save_path = os.path.join(SAVE_DIR, save_name)

                # 可视化并保存结果
                visualize_segmentation(results, save_path)
