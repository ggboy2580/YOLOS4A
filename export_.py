import os
import time
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 自动初始化CUDA

# =========================
# 配置参数
# =========================
pt_model_path = r"D:\Downloads\YOLOS4Ax_640.pt"
onnx_model_path = r"D:\Downloads\YOLOS4Ax_640.onnx"
trt_engine_dir = r"D:\Downloads\trt_engine"
data_yaml = r"D:\codes\YOLOS4A\myVisDrone.yaml"
save_dir = Path(r"D:\codes\YOLOS4A\runs\detect/trt_val")
calib_images_dir = r"D:\Datas\VisDrone\images\calib"  # 用于 INT8 校准的图片

os.makedirs(trt_engine_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

# =========================
# 1. 加载 PyTorch 模型并导出 ONNX
# =========================
print("=== Exporting ONNX model ===")
model = YOLO(pt_model_path).cuda()
dummy_input = torch.randn(1, 3, 640, 640).cuda()
model.model.eval()
torch.onnx.export(
    model.model,
    dummy_input,
    onnx_model_path,
    opset_version=12,
    input_names=["images"],
    output_names=["output"],
    dynamic_axes={"images": {0: "batch_size"}, "output": {0: "batch_size"}},
)
print(f"ONNX model saved: {onnx_model_path}")

# =========================
# 2. TensorRT Engine Helper
# =========================
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(onnx_file_path, engine_file_path, precision="fp32", calib_dataset=None):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        builder.max_workspace_size = 1 << 30  # 1GB

        # 精度设置
        if precision == "fp16":
            builder.fp16_mode = True
        elif precision == "int8":
            builder.int8_mode = True
            if calib_dataset is None:
                raise ValueError("INT8 requires calibration dataset")
            # INT8 校准器
            class CalibDataset(trt.IInt8EntropyCalibrator2):
                def __init__(self, image_files, batch_size=1):
                    super().__init__()
                    self.image_files = image_files
                    self.batch_size = batch_size
                    self.current_index = 0
                    self.device_input = cuda.mem_alloc(batch_size * 3 * 640 * 640 * 4)

                def get_batch_size(self):
                    return self.batch_size

                def get_batch(self, names):
                    if self.current_index >= len(self.image_files):
                        return None
                    batch_files = self.image_files[self.current_index:self.current_index+self.batch_size]
                    batch_data = []
                    for f in batch_files:
                        img = np.load(f).astype(np.float32) / 255.0
                        img = np.transpose(img, (2, 0, 1))
                        batch_data.append(img)
                    batch_data = np.array(batch_data, dtype=np.float32)
                    cuda.memcpy_htod(self.device_input, batch_data.ravel())
                    self.current_index += self.batch_size
                    return [int(self.device_input)]

                def read_calibration_cache(self, cache_file):
                    if os.path.exists(cache_file):
                        with open(cache_file, "rb") as f:
                            return f.read()
                    return None

                def write_calibration_cache(self, cache_file, cache):
                    with open(cache_file, "wb") as f:
                        f.write(cache)
            builder.int8_calibrator = CalibDataset(calib_dataset)

        # 解析 ONNX
        with open(onnx_file_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX")

        engine = builder.build_cuda_engine(network)
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        print(f"Saved TensorRT engine: {engine_file_path}")
        return engine

# =========================
# 3. 构建 / 加载 engine
# =========================
engine_files = {}
precisions = ["fp32", "fp16"]  # INT8 需要提供校准集，暂不演示
for prec in precisions:
    engine_path = os.path.join(trt_engine_dir, f"{prec}.engine")
    if os.path.exists(engine_path):
        engine_files[prec] = engine_path
        print(f"{prec} engine exists: {engine_path}")
    else:
        engine_files[prec] = engine_path
        build_engine(onnx_model_path, engine_path, precision=prec)

# =========================
# 4. TensorRT 推理函数
# =========================
def infer_trt(engine_file, input_image):
    # 输入：engine 文件路径，input_image：numpy [B,3,H,W] float32
    with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    # 输入输出
    input_shape = engine.get_binding_shape(0)
    output_shape = engine.get_binding_shape(1)
    d_input = cuda.mem_alloc(input_image.nbytes)
    d_output = cuda.mem_alloc(np.prod(output_shape) * 4)
    stream = cuda.Stream()
    cuda.memcpy_htod_async(d_input, input_image.ravel(), stream)
    start = time.time()
    context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
    stream.synchronize()
    end = time.time()
    fps = 1.0 / (end - start)
    return fps

# =========================
# 5. PyTorch FP32 / FP16 推理 FPS 测试
# =========================
dummy_img = torch.randn(1, 3, 640, 640).cuda()
model.model.eval()

# FP32
with torch.no_grad():
    torch.cuda.synchronize()
    t0 = time.time()
    out = model.model(dummy_img)
    torch.cuda.synchronize()
    t1 = time.time()
fps_fp32 = 1.0 / (t1 - t0)

# FP16
model.model.half()
dummy_img = dummy_img.half()
with torch.no_grad():
    torch.cuda.synchronize()
    t0 = time.time()
    out = model.model(dummy_img)
    torch.cuda.synchronize()
    t1 = time.time()
fps_fp16 = 1.0 / (t1 - t0)

# =========================
# 6. 打印结果
# =========================
print("=== FPS 对比 ===")
print(f"PyTorch FP32: {fps_fp32:.2f} FPS")
print(f"PyTorch FP16: {fps_fp16:.2f} FPS")
for prec in ["fp32","fp16"]:
    fps = infer_trt(engine_files[prec], dummy_img.cpu().numpy())
    print(f"TensorRT {prec.upper()}: {fps:.2f} FPS")
