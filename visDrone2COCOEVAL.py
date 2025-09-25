from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

ann_file = r"D:\Datas\VisDrone\coco\VisDrone2019-DET_val_coco.json"  # 转换后的 GT
pred_file = r"D:\codes\YOLOS4A\runs\detect\val9\predictions.json"  # YOLO 推理导出的 COCO 格式预测

coco_gt = COCO(ann_file)
coco_gt.dataset['info'] = {}
coco_dt = coco_gt.loadRes(pred_file)



coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
# === 修改面积划分为 VisDrone 规则 ===
coco_eval.params.areaRng = [
    [0, 1e5 ** 2],   # all
    [0, 400],        # small
    [400, 3600],     # medium
    [3600, 1e5 ** 2] # large
]
coco_eval.params.areaRngLbl = ['all', 'small', 'medium', 'large']
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()


# 提取 AP_S/M/L
AP_S, AP_M, AP_L = coco_eval.stats[3], coco_eval.stats[4], coco_eval.stats[5]
print(f"AP_S={AP_S:.3f}, AP_M={AP_M:.3f}, AP_L={AP_L:.3f}")
