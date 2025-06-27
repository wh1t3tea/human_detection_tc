import argparse
import logging
import time
from typing import Dict, Any, List

import cv2
import numpy as np

from detectors.yolo_onnx import YOLODetector
from utils.config_utils import load_config
from utils.video_utils import process_video, process_video_batch
from utils.visualization import draw_detections, draw_fps, generate_color_palette


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Video Object Detection with YOLO ONNX")

    parser.add_argument("--input", type=str, required=True,
                      help="Path to input video")
    parser.add_argument("--output", type=str, required=True,
                      help="Path to output video")
    parser.add_argument("--codec", type=str, default="mp4v",
                      help="FourCC codec code for output video (e.g., 'mp4v', 'avc1', 'XVID')")

    parser.add_argument("--model", type=str, required=True,
                      help="Path to YOLO ONNX model")
    parser.add_argument("--model-config", type=str, default="config/model_config.yaml",
                      help="Path to model configuration file")
    parser.add_argument("--batch_size", type=int, default=1,
                      help="Batch size for inference")

    parser.add_argument("--conf", type=float, default=0.5,
                      help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7,
                      help="IoU threshold")
    parser.add_argument("--device", type=str, default="cpu",
                      help="Device for inference: 'cpu' or 'cuda'")
    parser.add_argument("--img-size", type=int, default=640,
                      help="Input image size")

    parser.add_argument("--vis-config", type=str, default="config/visual_config.yaml",
                      help="Path to visualization configuration file")
    parser.add_argument("--preview", action="store_true",
                      help="Show preview window during processing")

    parser.add_argument("--frame-step", type=int, default=1,
                      help="Process every Nth frame (1 = all frames)")

    return parser.parse_args()


def main():
    args = parse_args()

    model_config = load_config(args.model_config)
    vis_config = load_config(args.vis_config)

    detector = YOLODetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        img_size=args.img_size,
        batch_size=args.batch_size
    )

    if "class_names" in model_config:
        class_names = {int(k): v for k, v in model_config["class_names"].items()}
        detector.set_class_names(class_names)

    class_colors = vis_config.get("class_colors", {})
    class_colors = {int(k): v for k, v in class_colors.items()} if class_colors else {}

    if args.batch_size <= 1:
        def process_frame(frame, frame_idx):
            t_start = time.time()
            detections = detector.detect(frame)
            t_end = time.time()

            fps = 1.0 / (t_end - t_start) if (t_end - t_start) > 0 else 0

            result = draw_detections(
                frame,
                detections,
                class_colors=class_colors,
                show_labels=not vis_config.get("hide_labels", False),
                show_scores=not vis_config.get("hide_conf", False),
                box_thickness=vis_config.get("box_thickness", 2),
                text_scale=vis_config.get("text_font_scale", 0.5),
                text_thickness=vis_config.get("text_thickness", 2),
                text_color=vis_config.get("text_color", (255, 255, 255)),
                text_bg_color=vis_config.get("text_bg_color", (0, 0, 0)),
            )

            if vis_config.get("show_fps", True):
                result = draw_fps(result, fps)

            return result
    else:
        def process_batch(frames, frame_indices):
            batch_results = []

            t_start = time.time()
            batch_detections = detector.detect_batch(frames)
            processing_time = time.time() - t_start

            fps = len(frames) / processing_time if processing_time > 0 else 0

            for i, (frame, detections) in enumerate(zip(frames, batch_detections)):
                result = draw_detections(
                    frame,
                    detections,
                    class_colors=class_colors,
                    show_labels=not vis_config.get("hide_labels", False),
                    show_scores=not vis_config.get("hide_conf", False),
                    box_thickness=vis_config.get("box_thickness", 2),
                    text_scale=vis_config.get("text_font_scale", 0.5),
                    text_thickness=vis_config.get("text_thickness", 2),
                    text_color=vis_config.get("text_color", (255, 255, 255)),
                    text_bg_color=vis_config.get("text_bg_color", (0, 0, 0)),
                )

                if vis_config.get("show_fps", True):
                    batch_fps_text = f"Batch FPS: {fps:.1f}"
                    result = draw_fps(result, fps, custom_text=batch_fps_text)

                batch_results.append(result)

            return batch_results

    logger.info("=" * 60)
    logger.info("Video Object Detection with YOLO ONNX")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Confidence threshold: {args.conf}")
    logger.info(f"IoU threshold: {args.iou}")
    logger.info(f"Frame step: {args.frame_step}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("=" * 60)

    if args.batch_size <= 1:
        stats = process_video(
            video_path=args.input,
            output_path=args.output,
            process_frame=process_frame,
            show_preview=args.preview,
            frame_step=args.frame_step,
            codec=args.codec,
        )

        logger.info("=" * 60)
        logger.info("Processing Complete")
        logger.info("=" * 60)
        logger.info(f"Processed {stats['processed_frames']} frames")
        logger.info(f"Average processing time: {stats['avg_processing_time']*1000:.2f} ms")
        logger.info(f"Processing FPS: {stats['processing_fps']:.2f}")
        logger.info(f"Output saved to: {stats['output_path']}")
        logger.info("=" * 60)
    else:
        stats = process_video_batch(
            video_path=args.input,
            output_path=args.output,
            process_batch=process_batch,
            batch_size=args.batch_size,
            show_preview=args.preview,
            frame_step=args.frame_step,
            codec=args.codec,
        )

        logger.info("=" * 60)
        logger.info("Batch Processing Complete")
        logger.info("=" * 60)
        logger.info(f"Processed {stats['processed_frames']} frames with batch size {stats['batch_size']}")
        logger.info(f"Average processing time per frame: {stats['avg_processing_time']*1000:.2f} ms")
        logger.info(f"Average processing time per batch: {stats['avg_batch_processing_time']*1000:.2f} ms")
        logger.info(f"Processing FPS: {stats['batch_processing_fps']:.2f}")
        logger.info(f"Output saved to: {stats['output_path']}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
