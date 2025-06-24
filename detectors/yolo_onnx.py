#!/usr/bin/env python3
"""
YOLO object detector implementation using ONNX Runtime.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

import cv2
import numpy as np
import onnxruntime as ort


# Configure module logger
logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLO object detector based on ONNX Runtime."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        img_size: int = 640,
        class_filter: Optional[List[int]] = None,
    ):
        """Initialize YOLO detector.

        Args:
            model_path: Path to ONNX model
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ("cpu" or "cuda")
            img_size: Input image size
            class_filter: List of class IDs to detect (None = all classes)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.class_filter = class_filter

        # Create ONNX Runtime session
        self.session = self._create_session(device)

        # Get model metadata
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Initialize class names with placeholder (need to be provided or loaded)
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

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference.

        Args:
            image: Input image (BGR)

        Returns:
            Preprocessed image ready for model input
        """
        # Resize
        input_height, input_width = self.img_size, self.img_size
        img = cv2.resize(image, (input_width, input_height))

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize [0-255] to [0-1]
        img = img.astype(np.float32) / 255.0

        # Channel first (HWC -> NCHW)
        img = img.transpose(2, 0, 1)

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        return img

    def postprocess(
        self,
        outputs: Dict[str, np.ndarray],
        orig_shape: Tuple[int, int],
    ) -> Dict[str, Any]:
        """Process model outputs to detection format.

        Args:
            outputs: Model outputs
            orig_shape: Original image shape (height, width)

        Returns:
            Dictionary with detection results
        """
        # Determine the format of outputs based on the number of dimensions
        # and shape of the output tensors
        if len(self.output_names) == 1:
            # YOLOv8 format - single output, shape: (1, n_boxes, n_classes+5)
            predictions = outputs[self.output_names[0]]
            return self._process_yolov8_output(predictions, orig_shape)
        else:
            # Multiple outputs - likely YOLOv5/v7 format
            return self._process_yolov5_output(outputs, orig_shape)

    def _process_yolov8_output(
        self,
        predictions: np.ndarray,
        orig_shape: Tuple[int, int],
    ) -> Dict[str, Any]:
        """Process YOLOv8 outputs.

        Args:
            predictions: Model predictions
            orig_shape: Original image shape (height, width)

        Returns:
            Dictionary with detection results
        """
        # Get image dimensions
        orig_height, orig_width = orig_shape
        input_height, input_width = self.img_size, self.img_size

        # Extract boxes, scores, and classes
        boxes = []
        scores = []
        class_ids = []

        # YOLOv8 output shape: (1, num_boxes, num_classes+5)
        for i in range(predictions.shape[1]):
            # Get score and class
            class_scores = predictions[0, i, 4:]
            class_id = np.argmax(class_scores)
            score = class_scores[class_id] * predictions[0, i, 4]

            # Filter by confidence threshold
            if score < self.conf_threshold:
                continue

            # Filter by class if specified
            if self.class_filter is not None and class_id not in self.class_filter:
                continue

            # Get bounding box coordinates
            x, y, w, h = predictions[0, i, :4]

            # Convert to corner coordinates and scale to original image
            x1 = (x - w/2) / input_width * orig_width
            y1 = (y - h/2) / input_height * orig_height
            x2 = (x + w/2) / input_width * orig_width
            y2 = (y + h/2) / input_height * orig_height

            # Add to results
            boxes.append([x1, y1, x2, y2])
            scores.append(float(score))
            class_ids.append(int(class_id))

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.conf_threshold, self.iou_threshold
        )

        # Prepare results
        result_boxes = []
        result_scores = []
        result_class_ids = []
        result_class_names = []

        for i in indices:
            # Handle both OpenCV 4.5+ (returns a flat array) and older versions
            idx = i if isinstance(i, int) else i[0]

            result_boxes.append(boxes[idx])
            result_scores.append(scores[idx])
            result_class_ids.append(class_ids[idx])

            # Get class name if available
            class_name = self.class_names.get(class_ids[idx], f"Class {class_ids[idx]}")
            result_class_names.append(class_name)

        # Create detection dictionary
        detections = {
            "boxes": np.array(result_boxes),
            "scores": np.array(result_scores),
            "classes": np.array(result_class_ids),
            "class_names": result_class_names,
            "count": len(result_boxes),
        }

        return detections

    def _process_yolov5_output(
        self,
        outputs: Dict[str, np.ndarray],
        orig_shape: Tuple[int, int],
    ) -> Dict[str, Any]:
        """Process YOLOv5/v7 outputs.

        Args:
            outputs: Model outputs
            orig_shape: Original image shape (height, width)

        Returns:
            Dictionary with detection results
        """
        # This implementation depends on the exact output format
        # For YOLOv5/v7, typically one output is the boxes and another is scores
        # This is a placeholder - would need to be adjusted based on the model
        raise NotImplementedError(
            "YOLOv5/v7 format not implemented. Please use YOLOv8 ONNX models."
        )

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """Run detection on an image.

        Args:
            image: Input image (BGR)

        Returns:
            Dictionary with detection results
        """
        # Get original image shape
        orig_shape = image.shape[:2]  # (height, width)

        # Preprocess image
        input_tensor = self.preprocess(image)

        # Run inference
        outputs = self.session.run(
            self.output_names, {self.input_name: input_tensor}
        )

        # Convert outputs to dictionary
        output_dict = {name: outputs[i] for i, name in enumerate(self.output_names)}

        # Postprocess outputs
        detections = self.postprocess(output_dict, orig_shape)

        return detections
