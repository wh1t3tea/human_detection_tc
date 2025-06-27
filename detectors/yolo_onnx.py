import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

import cv2
import numpy as np
import onnxruntime as ort


logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLO object detector based on ONNX Runtime."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.7,
        device: str = "cpu",
        img_size: int = 640,
        batch_size: int = 1
    ):
        """Initialize YOLO detector.

        Args:
            model_path: Path to ONNX model
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ("cpu" or "cuda")
            img_size: Input image size
            batch_size: Batch size for inference
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.batch_size = batch_size

        self.session = self._create_session(device)

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.class_names = {}

    def _create_session(self, device: str) -> ort.InferenceSession:
        """Create ONNX Runtime inference session.

        Args:
            device: Device to run inference on ("cpu" or "cuda")

        Returns:
            ONNX Runtime session
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        providers = ['CPUExecutionProvider']
        if device.lower() == "cuda":
            providers = ['CUDAExecutionProvider'] + providers

        try:
            session = ort.InferenceSession(self.model_path, providers=providers)
            logger.info(f"Loaded ONNX model: {self.model_path}")
            logger.info(f"Provider: {session.get_providers()[0]}")
            return session
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")

    def set_class_names(self, class_names: Dict[int, str]) -> None:
        """Set class names mapping.

        Args:
            class_names: Dictionary mapping class IDs to names
        """
        self.class_names = class_names

    def preprocess(self, images: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Preprocess image(s) for inference, matching ultralytics preprocessing.

        Args:
            images: Single image (BGR) or list of images

        Returns:
            Preprocessed image(s) ready for model input
        """
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]

        batch = []

        for image in images:
            original_height, original_width = image.shape[:2]
            input_height, input_width = self.img_size, self.img_size

            scale = min(input_width / original_width, input_height / original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            pad_x = (input_width - new_width) // 2
            pad_y = (input_height - new_height) // 2

            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            img = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
            img[pad_y:pad_y + new_height, pad_x:pad_x + new_width, :] = resized

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0

            img = img.transpose(2, 0, 1)

            batch.append(img)

        return np.array(batch)

    def postprocess(
        self,
        outputs: Dict[str, np.ndarray],
        orig_shapes: List[Tuple[int, int]],
    ) -> List[Dict[str, Any]]:
        """Process model outputs to detection format.

        Args:
            outputs: Model outputs
            orig_shapes: List of original image shapes (height, width)

        Returns:
            List of dictionaries with detection results
        """
        predictions = outputs[self.output_names[0]]
        batch_size = predictions.shape[0]

        results = []
        for i in range(batch_size):
            result = self._process_yolov12_output(
                predictions[i:i+1], orig_shapes[i]
            )
            results.append(result)

        return results

    def _process_yolov12_output(
        self,
        predictions: np.ndarray,
        orig_shape: Tuple[int, int],
    ) -> Dict[str, Any]:
        """Process YOLOv12 outputs.

        Args:
            predictions: Model predictions with shape (1, 84, 8000)
            orig_shape: Original image shape (height, width)

        Returns:
            Dictionary with detection results
        """
        orig_height, orig_width = orig_shape
        input_height, input_width = self.img_size, self.img_size

        scale = min(input_width / orig_width, input_height / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        pad_x = (input_width - new_width) // 2
        pad_y = (input_height - new_height) // 2

        boxes = []
        scores = []
        class_ids = []

        logger.info(f"Processing YOLOv12 output with shape {predictions.shape}")

        transposed_preds = predictions.transpose(0, 2, 1)

        box_coords = transposed_preds[0, :, :4]
        obj_conf = transposed_preds[0, :, 4]

        cls_scores = transposed_preds[0, :, 4:]

        max_cls_scores = np.max(cls_scores, axis=1)
        max_cls_indices = np.argmax(cls_scores, axis=1)

        final_scores = obj_conf

        keep_idxs = final_scores > self.conf_threshold

        if np.any(keep_idxs):
            filtered_boxes = box_coords[keep_idxs]
            filtered_scores = final_scores[keep_idxs]
            filtered_classes = max_cls_indices[keep_idxs]

            for i in range(len(filtered_boxes)):
                cx, cy, w, h = filtered_boxes[i]

                cx_adj = cx - pad_x
                cy_adj = cy - pad_y

                if cx_adj < 0 or cx_adj > new_width or cy_adj < 0 or cy_adj > new_height:
                    continue

                cx_orig = cx_adj / scale
                cy_orig = cy_adj / scale
                w_orig = w / scale
                h_orig = h / scale

                x1 = cx_orig - w_orig/2
                y1 = cy_orig - h_orig/2
                x2 = cx_orig + w_orig/2
                y2 = cy_orig + h_orig/2

                x1 = max(0, min(x1, orig_width))
                y1 = max(0, min(y1, orig_height))
                x2 = max(0, min(x2, orig_width))
                y2 = max(0, min(y2, orig_height))

                if x2 - x1 < 5 or y2 - y1 < 5:
                    continue

                if filtered_classes[i] == 0:
                    boxes.append([float(x1), float(y1), float(x2), float(y2)])
                    scores.append(float(filtered_scores[i]))
                    class_ids.append(int(filtered_classes[i]))

        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.conf_threshold, self.iou_threshold
        ) if boxes else []

        result_boxes = []
        result_scores = []
        result_class_ids = []
        result_class_names = []

        if len(indices) > 0:
            if isinstance(indices, np.ndarray) and indices.ndim == 1:
                selected_indices = indices
            else:
                selected_indices = [i[0] if isinstance(i, (list, np.ndarray)) else i for i in indices]

            for idx in selected_indices:
                result_boxes.append(boxes[idx])
                result_scores.append(scores[idx])
                result_class_ids.append(class_ids[idx])

                class_name = self.class_names.get(class_ids[idx], f"Class {class_ids[idx]}")
                result_class_names.append(class_name)

        detections = {
            "boxes": np.array(result_boxes) if result_boxes else np.empty((0, 4)),
            "scores": np.array(result_scores),
            "classes": np.array(result_class_ids),
            "class_names": result_class_names,
            "count": len(result_boxes),
        }

        return detections

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """Run detection on a single image.

        Args:
            image: Input image (BGR)

        Returns:
            Dictionary with detection results
        """
        orig_shape = image.shape[:2]
        input_tensor = self.preprocess(image)

        outputs = self.session.run(
            self.output_names, {self.input_name: input_tensor}
        )

        output_dict = {name: outputs[i] for i, name in enumerate(self.output_names)}

        detections = self.postprocess(output_dict, [orig_shape])[0]

        return detections

    def detect_batch(self, images: List[np.ndarray], batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run detection on a batch of images.

        Args:
            images: List of input images (BGR)
            batch_size: Batch size override (default: use class batch_size)

        Returns:
            List of dictionaries with detection results
        """
        if batch_size is None:
            batch_size = self.batch_size

        results = []
        orig_shapes = [img.shape[:2] for img in images]

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_shapes = orig_shapes[i:i+batch_size]

            input_tensor = self.preprocess(batch_images)

            outputs = self.session.run(
                self.output_names, {self.input_name: input_tensor}
            )

            output_dict = {name: outputs[i] for i, name in enumerate(self.output_names)}

            batch_results = self.postprocess(output_dict, batch_shapes)
            results.extend(batch_results)

        return results
