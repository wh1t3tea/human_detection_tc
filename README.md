# Video Human Detection with YOLO ONNX

A clean implementation of video human detection using YOLO models with ONNX Runtime.

## Features

- Process video files with YOLO object detection
- Optimize inference with ONNX Runtime
- Customize visualization settings (boxes, labels, colors)
- No Ultralytics dependency for inference

## Installation

### Requirements

- Python 3.8+
- OpenCV 4.5+
- ONNX Runtime 1.13+
- NumPy 1.20+

### Setup

1. Clone this repository:
```bash
git clone https://github.com/username/detection_tc.git
cd detection_tc
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Video Detection

Run human detection on a video file:

```bash
python main.py --input path/to/video.mp4 --output path/to/output.mp4 --model path/to/model.onnx
```

#### Options:

- `--input`: Path to input video
- `--output`: Path to output video
- `--model`: Path to YOLO ONNX model
- `--model-config`: Path to model configuration file
- `--vis-config`: Path to visualization configuration file
- `--preview`: Show preview window during processing
- `--frame-step`: Process every Nth frame (1 = all frames)

#### Detection Parameters:

- `--conf`: Confidence threshold (default: 0.25)
- `--iou`: IoU threshold (default: 0.45)
- `--device`: Device for inference: 'cpu' or 'cuda'
- `--img-size`: Input image size (default: 640)
- `--classes`: Filter by class IDs

### Exporting Models to ONNX

To convert a YOLO model to ONNX format (requires installation of ultralytics):

```bash
# Install ultralytics
pip install ultralytics

# Export model
python tools/export_onnx.py --model yolo12n

```

#### Options:

- `--model`: Path to YOLO model or model name (e.g., "yolov8n")
- `--output`: Path to save ONNX model
- `--opset`: ONNX opset version
- `--simplify`: Simplify ONNX model
- `--dynamic`: Use dynamic axes
- `--batch`: Batch size
- `--device`: Device to use for export (cpu, cuda)

## Configuration

The project uses YAML configuration files for both model and visualization settings.

### Model Configuration

`config/model_config.yaml` contains settings for the model:

```yaml
# Model settings
img_size: 640
device: "cpu"

# Inference settings
conf_threshold: 0.25
iou_threshold: 0.45

class_names:
  0: "person"
```

### Visualization Configuration

`config/visual_config.yaml` contains settings for visualization:

```yaml
# Bounding box settings
box_thickness: 2
box_color: null  # null means auto-color by class

# Text settings
text_thickness: 2
text_color: [255, 255, 255]  # White
text_bg_color: [0, 0, 0]     # Black
text_font_scale: 0.5

# Display settings
hide_labels: false
hide_conf: false
show_fps: true

# Class-specific colors
class_colors:
  0: [0, 255, 255]    # Yellow
```

## Project Structure

```
detection_tc/
├── config/                 # Configuration files
│   ├── model_config.yaml   # Model settings
│   └── visual_config.yaml  # Visualization settings
├── detectors/              # Object detectors
│   └── yolo_onnx.py        # YOLO ONNX detector implementation
├── tools/                  # Command-line tools
│   └── export/             # Model export tools
│       └── export_onnx.py  # ONNX export tool
├── utils/                  # Utility modules
│   ├── video/              # Video processing utilities
│   │   └── video_utils.py  # Video loading/processing
│   └── vis/                # Visualization utilities
│       └── visualization.py # Drawing functions
├── main.py                 # Main application
├── requirements.txt        # Dependencies without ultralytics
└── README.md               # This file
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
