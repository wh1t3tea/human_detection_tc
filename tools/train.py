import ultralytics
import os
import argparse


def main(args):
    model = ultralytics.YOLO(args.model_name)
    model.train(
        data=args.data_path,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        device=args.device,
        workers=args.workers,
        lr0=args.lr0
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="yolov12n.pt")
    parser.add_argument("--data_path", type=str, default="data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr0", type=float, default=0.01)
    main(parser.parse_args())
