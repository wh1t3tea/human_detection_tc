import ultralytics
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="yolov12n.pt")
    parser.add_argument("--data_path", type=str, default="data/other/cityperson/data.yaml")
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()
    model = ultralytics.YOLO(args.model_name)
    model.val(
        data=args.data_path,
        imgsz=args.img_size,
        device=args.device,
        workers=8,
        batch=16,
    )


if __name__ == "__main__":
    main()
