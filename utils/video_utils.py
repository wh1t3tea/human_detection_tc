#!/usr/bin/env python3
"""
Video processing utilities.
"""

import os
import time
import logging
from typing import Callable, Dict, Any, Tuple

import cv2
import numpy as np
from tqdm import tqdm


# Configure module logger
logger = logging.getLogger(__name__)


def get_video_properties(video_path: str) -> Dict[str, Any]:
    """Get video properties.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary of video properties
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    # Get properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": frame_count / fps if fps > 0 else 0,
    }


def process_video(
    video_path: str,
    output_path: str,
    process_frame: Callable[[np.ndarray, int], np.ndarray],
    show_preview: bool = False,
    frame_step: int = 1,
    preview_scale: float = 1.0,
    codec: str = "mp4v",
) -> Dict[str, Any]:
    """Process a video file.

    Args:
        video_path: Path to input video
        output_path: Path to output video
        process_frame: Function to process each frame
        show_preview: Whether to show preview window
        frame_step: Process every Nth frame
        preview_scale: Scale factor for preview window
        codec: FourCC codec code

    Returns:
        Dictionary with processing statistics
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process video
    processed_frames = 0
    current_frame = 0
    total_time = 0.0

    # Create progress bar
    pbar = tqdm(total=frame_count, desc="Processing video")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames if frame_step > 1
            if current_frame % frame_step == 0:
                # Process frame
                t_start = time.time()
                processed_frame = process_frame(frame, current_frame)
                t_end = time.time()

                processing_time = t_end - t_start
                total_time += processing_time

                # Write frame
                out.write(processed_frame)
                processed_frames += 1

                # Show preview
                if show_preview:
                    if preview_scale != 1.0:
                        preview_width = int(width * preview_scale)
                        preview_height = int(height * preview_scale)
                        preview = cv2.resize(
                            processed_frame,
                            (preview_width, preview_height)
                        )
                    else:
                        preview = processed_frame

                    cv2.imshow("Preview", preview)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            current_frame += 1
            pbar.update(1)

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")

    finally:
        # Release resources
        pbar.close()
        cap.release()
        out.release()

        if show_preview:
            cv2.destroyAllWindows()

    # Calculate stats
    avg_time = total_time / processed_frames if processed_frames > 0 else 0
    processing_fps = 1.0 / avg_time if avg_time > 0 else 0

    # Return statistics
    stats = {
        "input_path": video_path,
        "output_path": output_path,
        "frame_count": frame_count,
        "processed_frames": processed_frames,
        "total_processing_time": total_time,
        "avg_processing_time": avg_time,
        "processing_fps": processing_fps,
    }

    return stats


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_step: int = 1,
    max_frames: int = None,
    image_format: str = "jpg",
    quality: int = 95,
) -> int:
    """Extract frames from a video.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        frame_step: Extract every Nth frame
        max_frames: Maximum number of frames to extract
        image_format: Output image format ('jpg', 'png')
        quality: Image quality for JPG (0-100)

    Returns:
        Number of extracted frames
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    # Process video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    extracted_frames = 0

    # Create progress bar
    pbar = tqdm(total=frame_count, desc="Extracting frames")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Extract frame if it's a multiple of frame_step
            if current_frame % frame_step == 0:
                if max_frames is not None and extracted_frames >= max_frames:
                    break

                # Save frame
                if image_format.lower() == "jpg":
                    filename = os.path.join(output_dir, f"frame_{current_frame:06d}.jpg")
                    cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                else:
                    filename = os.path.join(output_dir, f"frame_{current_frame:06d}.png")
                    cv2.imwrite(filename, frame)

                extracted_frames += 1

            current_frame += 1
            pbar.update(1)

    except KeyboardInterrupt:
        logger.info("Extraction interrupted by user")

    finally:
        pbar.close()
        cap.release()

    return extracted_frames
