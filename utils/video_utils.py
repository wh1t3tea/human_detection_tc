import os
import time
import logging
import subprocess
from typing import Callable, Dict, Any, Tuple, List, Optional

import cv2
import numpy as np
from tqdm import tqdm


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
        codec: FourCC codec code (default: "mp4v", alternatives: "avc1", "XVID")

    Returns:
        Dictionary with processing statistics
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Always create a temporary file first
    temp_output_path = output_path + ".temp.mp4"
    actual_output_path = output_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    codecs_to_try = [codec, "mp4v", "XVID", "MJPG", "X264"]
    out = None

    for current_codec in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*current_codec)
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

            if out.isOpened():
                logger.info(f"Using codec '{current_codec}' for temporary output video")
                break
            else:
                out.release()
        except Exception as e:
            logger.warning(f"Failed to initialize VideoWriter with codec '{current_codec}': {e}")

    if out is None or not out.isOpened():
        raise RuntimeError(f"Failed to create output video file: {temp_output_path}. Tried codecs: {codecs_to_try}")

    processed_frames = 0
    current_frame = 0
    total_time = 0.0

    pbar = tqdm(total=frame_count, desc="Processing video")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame % frame_step == 0:
                t_start = time.time()
                processed_frame = process_frame(frame, current_frame)
                t_end = time.time()

                processing_time = t_end - t_start
                total_time += processing_time

                out.write(processed_frame)
                processed_frames += 1

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
    except Exception as e:
        logger.error(f"Error during video processing: {e}")
        raise

    finally:
        pbar.close()
        cap.release()

        if out is not None:
            out.release()
            logger.info(f"Video writer released for: {temp_output_path}")

        if show_preview:
            cv2.destroyAllWindows()

    # Post-process with ffmpeg to maintain quality
    if os.path.exists(temp_output_path):
        try:
            # Get the file size of the original video to calculate target bitrate
            original_size = os.path.getsize(video_path)
            video_duration = get_video_properties(video_path)["duration"]
            target_bitrate = int((original_size * 8) / video_duration)  # in bits per second

            # Ensure a minimum bitrate for quality
            target_bitrate = max(target_bitrate, 1000000)  # minimum 1 Mbps

            # Convert to kbps for ffmpeg
            bitrate_str = f"{target_bitrate//1000}k"
            logger.info(f"Target bitrate: {bitrate_str} (based on original video size)")

            cmd = [
                "ffmpeg", "-y",
                "-i", temp_output_path,
                "-c:v", "libx264",
                "-b:v", bitrate_str,
                "-preset", "medium",
                actual_output_path
            ]

            logger.info("Re-encoding video with ffmpeg to match original quality...")
            subprocess.run(cmd, check=True)
            logger.info(f"FFmpeg re-encoding completed: {actual_output_path}")

            # Delete temporary file
            os.remove(temp_output_path)
            logger.info(f"Temporary file removed: {temp_output_path}")

        except Exception as e:
            logger.error(f"Failed to re-encode with ffmpeg: {e}")
            logger.warning("Using basic output without size optimization")

            # If ffmpeg failed, rename the temp file to the actual output
            if os.path.exists(temp_output_path):
                if os.path.exists(actual_output_path):
                    os.remove(actual_output_path)
                os.rename(temp_output_path, actual_output_path)

    # Ensure output_path is set to the actual output path for stats
    output_path = actual_output_path

    avg_time = total_time / processed_frames if processed_frames > 0 else 0
    processing_fps = 1.0 / avg_time if avg_time > 0 else 0

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


def process_video_batch(
    video_path: str,
    output_path: str,
    process_batch: Callable[[List[np.ndarray], List[int]], List[np.ndarray]],
    batch_size: int = 4,
    show_preview: bool = False,
    frame_step: int = 1,
    preview_scale: float = 1.0,
    codec: str = "mp4v",
) -> Dict[str, Any]:
    """Process a video file using batch processing.

    Args:
        video_path: Path to input video
        output_path: Path to output video
        process_batch: Function to process batch of frames
        batch_size: Number of frames to process at once
        show_preview: Whether to show preview window
        frame_step: Process every Nth frame
        preview_scale: Scale factor for preview window
        codec: FourCC codec code (default: "mp4v", alternatives: "avc1", "XVID")

    Returns:
        Dictionary with processing statistics
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Always create a temporary file first
    temp_output_path = output_path + ".temp.mp4"
    actual_output_path = output_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    codecs_to_try = [codec, "mp4v", "XVID", "MJPG", "X264"]
    out = None

    for current_codec in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*current_codec)
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

            if out.isOpened():
                logger.info(f"Using codec '{current_codec}' for temporary output video")
                break
            else:
                out.release()
        except Exception as e:
            logger.warning(f"Failed to initialize VideoWriter with codec '{current_codec}': {e}")

    if out is None or not out.isOpened():
        raise RuntimeError(f"Failed to create output video file: {temp_output_path}. Tried codecs: {codecs_to_try}")

    processed_frames = 0
    current_frame = 0
    total_time = 0.0

    pbar = tqdm(total=frame_count, desc="Processing video in batches")

    try:
        while cap.isOpened():
            batch_frames = []
            batch_indices = []

            while len(batch_frames) < batch_size and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if current_frame % frame_step == 0:
                    batch_frames.append(frame)
                    batch_indices.append(current_frame)

                current_frame += 1
                pbar.update(1)

            if not batch_frames:
                break

            t_start = time.time()
            processed_frames_batch = process_batch(batch_frames, batch_indices)
            t_end = time.time()

            processing_time = t_end - t_start
            total_time += processing_time

            for i, processed_frame in enumerate(processed_frames_batch):
                out.write(processed_frame)
                processed_frames += 1

                if show_preview and i == len(processed_frames_batch) - 1:
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

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Error during batch video processing: {e}")
        raise

    finally:
        pbar.close()
        cap.release()

        if out is not None:
            out.release()
            logger.info(f"Video writer released for: {temp_output_path}")

        if show_preview:
            cv2.destroyAllWindows()

    # Post-process with ffmpeg to maintain quality
    if os.path.exists(temp_output_path):
        try:
            # Get the file size of the original video to calculate target bitrate
            original_size = os.path.getsize(video_path)
            video_duration = get_video_properties(video_path)["duration"]
            target_bitrate = int((original_size * 8) / video_duration)  # in bits per second

            # Ensure a minimum bitrate for quality
            target_bitrate = max(target_bitrate, 1000000)  # minimum 1 Mbps

            # Convert to kbps for ffmpeg
            bitrate_str = f"{target_bitrate//1000}k"
            logger.info(f"Target bitrate: {bitrate_str} (based on original video size)")

            cmd = [
                "ffmpeg", "-y",
                "-i", temp_output_path,
                "-c:v", "libx264",
                "-b:v", bitrate_str,
                "-preset", "medium",
                actual_output_path
            ]

            logger.info("Re-encoding video with ffmpeg to match original quality...")
            subprocess.run(cmd, check=True)
            logger.info(f"FFmpeg re-encoding completed: {actual_output_path}")

            # Delete temporary file
            os.remove(temp_output_path)
            logger.info(f"Temporary file removed: {temp_output_path}")

        except Exception as e:
            logger.error(f"Failed to re-encode with ffmpeg: {e}")
            logger.warning("Using basic output without size optimization")

            # If ffmpeg failed, rename the temp file to the actual output
            if os.path.exists(temp_output_path):
                if os.path.exists(actual_output_path):
                    os.remove(actual_output_path)
                os.rename(temp_output_path, actual_output_path)

    # Ensure output_path is set to the actual output path for stats
    output_path = actual_output_path

    avg_time = total_time / processed_frames if processed_frames > 0 else 0
    processing_fps = 1.0 / avg_time if avg_time > 0 else 0

    batch_avg_time = total_time / (processed_frames / batch_size) if processed_frames > 0 else 0
    batch_processing_fps = processed_frames / total_time if total_time > 0 else 0

    stats = {
        "input_path": video_path,
        "output_path": output_path,
        "frame_count": frame_count,
        "processed_frames": processed_frames,
        "batch_size": batch_size,
        "total_processing_time": total_time,
        "avg_processing_time": avg_time,
        "avg_batch_processing_time": batch_avg_time,
        "processing_fps": processing_fps,
        "batch_processing_fps": batch_processing_fps,
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

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    extracted_frames = 0

    pbar = tqdm(total=frame_count, desc="Extracting frames")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if current_frame % frame_step == 0:
                if max_frames is not None and extracted_frames >= max_frames:
                    break

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
