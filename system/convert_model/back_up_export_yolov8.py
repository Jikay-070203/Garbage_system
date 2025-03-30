# Ultralytics YOLOv8 🚀 Export Script

import torch
from pathlib import Path
import argparse
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
from ultralytics.utils import yaml_save

def export_yolov8(
    weights='yolov8n.pt',
    imgsz=640,
    device='cpu',
    include=('torchscript', 'onnx'),
    half=False,
    simplify=False,
    dynamic=False,
    opset=17,
    int8=False,
    export_dir='exports'
):
    # ✅ Xử lý 'gpu' thành '0' (GPU đầu tiên)
    if device.lower() == 'gpu':
        device = '0'

    file = Path(weights)
    model = YOLO(file)

    export_kwargs = {
        'format': None,
        'half': half,
        'dynamic': dynamic,
        'simplify': simplify,
        'int8': int8,
        'opset': opset,
        'device': device,
        'imgsz': imgsz,
    }

    for fmt in include:
        print(f'🚀 Exporting to {fmt.upper()}...')
        export_kwargs['format'] = fmt
        model.export(**export_kwargs)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='model path')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--device', default='cpu', help='device to use for export (cpu, gpu, 0, 1, etc.)')
    parser.add_argument('--include', nargs='+', default=['torchscript', 'onnx'], help='export formats')
    parser.add_argument('--half', action='store_true', help='use FP16 precision')
    parser.add_argument('--simplify', action='store_true', help='simplify ONNX model')
    parser.add_argument('--dynamic', action='store_true', help='dynamic input shapes')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    parser.add_argument('--int8', action='store_true', help='INT8 quantization (only some formats)')
    parser.add_argument('--export-dir', type=str, default='exports', help='directory to save exported models')
    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    export_yolov8(**vars(opt))
