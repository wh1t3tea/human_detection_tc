import argparse
import os
from pathlib import Path
import shutil
import logging

from ultralytics import YOLO

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def export_to_onnx(
    model_name,
    model_path=None,
    opset=12,
    simplify=True,
    dynamic=True,
    batch=8,
    device='cpu'):
    """
    Export PyTorch YOLO model to ONNX format

    Args:
        model_name (str): Yolo model name
        output_path (str, optional): Path to save the ONNX model
        opset (int, optional): ONNX opset version
        simplify (bool, optional): Simplify ONNX model
        dynamic (bool, optional): Use dynamic axes
        batch (int, optional): Batch size for inference
        device (str, optional): Device to use for export (cpu, cuda)

    Returns:
        str: Path to the exported ONNX model
    """

    os.makedirs(os.path.join(MODELS_DIR, model_name, "torch"), exist_ok=True)
    os.makedirs(os.path.join(MODELS_DIR, model_name, "onnx"), exist_ok=True)

    torch_path = os.path.join(MODELS_DIR, model_name, "torch", f"{model_name}.pt")
    onnx_path = os.path.join(MODELS_DIR, model_name, "onnx", f"{model_name}.onnx")
    source_onnx_path = model_path.replace(".pt", ".onnx") if model_path else os.path.join(MODELS_DIR, model_name, "torch", f"{model_name}.onnx")

    model = YOLO(model_path if model_path else torch_path)

    export_options = {
        'format': 'onnx',
        'opset': opset,
        'simplify': simplify,
        'dynamic': dynamic,
        'device': device,
        'batch': batch
    }

    logger.info(f"Exporting to ONNX with options: {export_options}")
    result = model.export(**export_options)
    shutil.move(source_onnx_path, onnx_path)
    logger.info(f"ONNX model exported to: {onnx_path}")


def main():
    parser = argparse.ArgumentParser(description='Export YOLO PyTorch model to ONNX format')
    parser.add_argument('--model_path', default=None, type=str,
                        help='Path to the PyTorch model (.pt) or a model name (e.g., "yolov8n", "yolov8s")')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', default=True, help='Simplify ONNX model')
    parser.add_argument('--dynamic', action='store_true', default=True, help='Use dynamic axes')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for export (cpu, cuda)')
    parser.add_argument('--batch', type=int, default=8, help='Batch size for non-streaming inference. For streaming inference, set to 1.')

    args = parser.parse_args()

    export_to_onnx(
        model_name=args.model_name,
        model_path=args.model_path,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=args.dynamic,
        batch=args.batch,
        device=args.device
    )

if __name__ == '__main__':
    main()
