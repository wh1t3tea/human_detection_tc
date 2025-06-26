import logging
from typing import Dict, List, Tuple, Optional, Any, Union

import cv2
import numpy as np


logger = logging.getLogger(__name__)


def draw_bbox(
    image: np.ndarray,
    bbox: List[float],
    color: Tuple[int, int, int],
    thickness: int = 2,
    line_type: int = cv2.LINE_AA,
) -> np.ndarray:
    """Draw a bounding box on an image.

    Args:
        image: Input image (BGR)
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        color: RGB color (B, G, R)
        thickness: Line thickness
        line_type: Line type

    Returns:
        Image with bounding box
    """
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, line_type)
    return image


def draw_text(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.5,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    background: Optional[Tuple[int, int, int]] = None,
    padding: int = 5,
    line_type: int = cv2.LINE_AA,
) -> np.ndarray:
    """Draw text on an image with optional background.

    Args:
        image: Input image (BGR)
        text: Text to draw
        position: Position (x, y) of text
        font_scale: Font scale
        color: RGB color (B, G, R)
        thickness: Line thickness
        font: Font type
        background: Background color (B, G, R), None for no background
        padding: Padding around text (if background is used)
        line_type: Line type

    Returns:
        Image with text
    """
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    if background is not None:
        x, y = position
        cv2.rectangle(
            image,
            (x - padding, y - text_height - padding - baseline),
            (x + text_width + padding, y + padding),
            background,
            -1,
            line_type
        )

    cv2.putText(
        image,
        text,
        position,
        font,
        font_scale,
        color,
        thickness,
        line_type
    )

    return image


def generate_color_palette(num_classes: int = 80) -> Dict[int, Tuple[int, int, int]]:
    """Generate a color palette for object classes.

    Args:
        num_classes: Number of classes

    Returns:
        Dictionary mapping class IDs to colors
    """
    np.random.seed(42)
    colors = {}

    for i in range(num_classes):
        hue = i / num_classes
        saturation = 0.8 + np.random.uniform(-0.2, 0.2)
        value = 0.8 + np.random.uniform(-0.2, 0.2)

        hsv = np.array([[[hue * 179, saturation * 255, value * 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]

        colors[i] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    return colors


def draw_detections(
    image: np.ndarray,
    detections: Dict[str, Any],
    class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    box_thickness: int = 2,
    text_scale: float = 0.5,
    text_thickness: int = 2,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    text_bg_color: Optional[Tuple[int, int, int]] = (0, 0, 0),
    show_labels: bool = True,
    show_scores: bool = True,
) -> np.ndarray:
    """Draw detection results on image.

    Args:
        image: Input image
        detections: Detection results
        class_colors: Dictionary mapping class IDs to colors
        box_thickness: Bounding box thickness
        text_scale: Text font scale
        text_thickness: Text thickness
        text_color: Text color
        text_bg_color: Text background color (None for no background)
        show_labels: Whether to show class labels
        show_scores: Whether to show confidence scores

    Returns:
        Image with drawn detections
    """
    result = image.copy()

    if detections.get("count", 0) == 0:
        return result

    if class_colors is None:
        num_classes = max(detections["classes"]) + 1 if len(detections["classes"]) > 0 else 80
        class_colors = generate_color_palette(num_classes)

    for i in range(detections["count"]):
        box = detections["boxes"][i]
        score = detections["scores"][i]
        class_id = detections["classes"][i]
        class_name = detections["class_names"][i] if "class_names" in detections else f"Class {class_id}"

        x1, y1, x2, y2 = [int(coord) for coord in box]

        color = class_colors.get(class_id, (0, 255, 0))

        cv2.rectangle(result, (x1, y1), (x2, y2), color, box_thickness)

        if show_labels or show_scores:
            label_parts = []

            if show_labels:
                label_parts.append(class_name)

            if show_scores:
                label_parts.append(f"{score:.2f}")

            label = " ".join(label_parts)

            text_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
            )

            if text_bg_color is not None:
                cv2.rectangle(
                    result,
                    (x1, y1 - text_size[1] - baseline),
                    (x1 + text_size[0], y1),
                    text_bg_color,
                    -1,
                )

            cv2.putText(
                result,
                label,
                (x1, y1 - baseline // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                text_color,
                text_thickness,
                cv2.LINE_AA,
            )

    return result


def draw_fps(
    image: np.ndarray,
    fps: float,
    position: Tuple[int, int] = (10, 30),
    text_color: Tuple[int, int, int] = (0, 255, 0),
    text_scale: float = 0.7,
    text_thickness: int = 2,
    bg_color: Optional[Tuple[int, int, int]] = None,
    custom_text: Optional[str] = None,
) -> np.ndarray:
    """Draw FPS counter on image.

    Args:
        image: Input image
        fps: FPS value
        position: Text position (x, y)
        text_color: Text color
        text_scale: Text font scale
        text_thickness: Text thickness
        bg_color: Background color (None for no background)
        custom_text: Optional custom text to display instead of "FPS: X.X"

    Returns:
        Image with FPS counter
    """
    result = image.copy()

    fps_text = custom_text if custom_text is not None else f"FPS: {fps:.1f}"

    (text_width, text_height), baseline = cv2.getTextSize(
        fps_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
    )

    if bg_color is not None:
        bg_rect_p1 = (position[0] - 5, position[1] - text_height - 5)
        bg_rect_p2 = (position[0] + text_width + 5, position[1] + 5)
        cv2.rectangle(result, bg_rect_p1, bg_rect_p2, bg_color, -1)

    cv2.putText(
        result,
        fps_text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        text_color,
        text_thickness,
        cv2.LINE_AA,
    )

    return result


def draw_text_overlay(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int] = (10, 30),
    text_color: Tuple[int, int, int] = (0, 255, 0),
    text_scale: float = 0.7,
    text_thickness: int = 2,
    bg_color: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """Draw text overlay on image.

    Args:
        image: Input image
        text: Text to display
        position: Text position (x, y)
        text_color: Text color
        text_scale: Text font scale
        text_thickness: Text thickness
        bg_color: Background color (None for no background)

    Returns:
        Image with text overlay
    """
    result = image.copy()

    lines = text.split('\n')

    (_, text_height), baseline = cv2.getTextSize(
        lines[0], cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
    )
    line_height = text_height + baseline + 5

    for i, line in enumerate(lines):
        line_pos = (position[0], position[1] + i * line_height)

        (text_width, _), _ = cv2.getTextSize(
            line, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
        )

        if bg_color is not None:
            bg_rect_p1 = (line_pos[0] - 5, line_pos[1] - text_height - 5)
            bg_rect_p2 = (line_pos[0] + text_width + 5, line_pos[1] + 5)
            cv2.rectangle(result, bg_rect_p1, bg_rect_p2, bg_color, -1)

        cv2.putText(
            result,
            line,
            line_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            text_color,
            text_thickness,
            cv2.LINE_AA,
        )

    return result
