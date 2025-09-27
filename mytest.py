from ultralytics import YOLO
from pathlib import Path
import time

if __name__ == "__main__":
    # 1. 加载模型
    model = YOLO(r"D:\Models\YOLOS4A-l.pt")

    FF=["CA","ECA","SE"]
    for ff in FF:
        if ff in str(model.model):
            print(ff)
    print(model.model)
    # 2. 计算参数量（单位：百万）
    num_params = sum(p.numel() for p in model.model.parameters()) / 1e6

    # 2. 设置数据集路径
    data_yaml = r"D:\Datas\CODrone\CODrone.yaml"

    # 3. 开始验证并计时
    start = time.time()
    metrics = model.val(data=data_yaml, split="test",save_json=True)  # split="val" 指定验证集
    elapsed = time.time() - start

    # 4. 提取关键指标
    # mAP50-95 (COCO style)
    mAP50_95 = metrics.box.map     # 相当于 metrics.box.map
    # mAP50
    mAP50 = metrics.box.map50
    # 小/中/大目标 AP
    AP_S = metrics.box.maps[0]     # small
    AP_M = metrics.box.maps[1]     # medium
    AP_L = metrics.box.maps[2]     # large
    # 推理速度 (Sec/img)
    sec_per_img = metrics.speed['inference'] / 1000.0  # speed 单位是 ms/img

    # 5. 提取指标
    # Precision / Recall
    precision = metrics.box.mp    # mean precision
    recall = metrics.box.mr       # mean recall

    # 5. 打印结果
    print("====== VisDrone Validation Results ======")
    print(f"Params(M)\tprecision\trecall\tmAP50-95\tmAP50\tAP_S\tAP_M\tAP_L\tSec/img")
    print(f"{num_params:.1f}\t{precision:.3f}\t{recall:.3f}\t{mAP50_95:.3f}\t{mAP50:.3f}\t{AP_S:.3f}\t{AP_M:.3f}\t{AP_L:.3f}\t{sec_per_img:.6f}")
    print(f"mAP50-95: {mAP50_95:.4f}")
    print(f"mAP50:    {mAP50:.4f}")
    print(f"AP_S:     {AP_S:.4f}")
    print(f"AP_M:     {AP_M:.4f}")
    print(f"AP_L:     {AP_L:.4f}")
    print(f"Sec/img:  {sec_per_img:.6f}")
    print(f"Total validation time: {elapsed:.2f} s")