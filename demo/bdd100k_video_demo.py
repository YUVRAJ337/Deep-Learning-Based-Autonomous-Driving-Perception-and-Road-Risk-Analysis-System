import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from PIL import Image, ImageDraw, ImageFont

from data.bdd100k_dataset import (
    normalize_bdd100k_image,
    _image_tensor,
)
from models.detection.train_detector import load_detector


CLASS_NAMES = [
    "bike", "bus", "car", "motor", "person",
    "rider", "traffic light", "traffic sign",
    "train", "truck",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/raw/demo_input.mp4",
        help="Input driving video",
    )
    parser.add_argument(
        "--output",
        default="outputs/bdd100k_detection_demo.mp4",
        help="Output annotated video",
    )
    parser.add_argument(
        "--config",
        default="configs/bdd100k_detector.yaml",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=30,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    checkpoint = Path(cfg["ckpt_path"])
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Detector checkpoint not found: {checkpoint}"
        )

    print(f"Device: {device}")
    print(f"Loading checkpoint: {checkpoint}")

    model = load_detector(cfg, str(checkpoint), device)
    model.score_threshold = args.threshold
    model.nms_threshold = 0.5
    model.max_detections = 50
    model.eval()

    print("Detector loaded successfully.")

    reader = imageio.get_reader(str(input_path))
    metadata = reader.get_meta_data()
    fps = float(metadata.get("fps", 24.0) or 24.0)

    max_frames = (
        int(args.max_seconds * fps)
        if args.max_seconds > 0 else None
    )

    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )

    frame_count = 0
    total_detections = 0

    try:
        for frame_index, frame in enumerate(reader):
            if max_frames is not None and frame_index >= max_frames:
                break

            # Convert video frame to an RGB PIL image.
            original = Image.fromarray(frame).convert("RGB")
            original = normalize_bdd100k_image(original)

            original_width, original_height = original.size

            # Match the detector's training input size.
            resized = original.resize(
                (800, 448),
                Image.Resampling.BILINEAR,
            )

            image_tensor = _image_tensor(resized).unsqueeze(0).to(device)

            # Run detection.
            with torch.inference_mode():
                boxes, scores, labels = model(image_tensor)

            boxes = boxes[0].detach().cpu().numpy()
            scores = scores[0].detach().cpu().numpy()
            labels = labels[0].detach().cpu().numpy()

            # Draw predictions on the original-resolution frame.
            canvas = original.copy()
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.load_default()

            scale_x = original_width / 800.0
            scale_y = original_height / 448.0

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box

                x1 = int(np.clip(x1 * scale_x, 0, original_width - 1))
                y1 = int(np.clip(y1 * scale_y, 0, original_height - 1))
                x2 = int(np.clip(x2 * scale_x, 0, original_width - 1))
                y2 = int(np.clip(y2 * scale_y, 0, original_height - 1))

                class_id = int(label)
                class_name = (
                    CLASS_NAMES[class_id]
                    if 0 <= class_id < len(CLASS_NAMES)
                    else f"class_{class_id}"
                )

                caption = f"{class_name} {float(score):.2f}"

                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline="red",
                    width=max(2, original_width // 500),
                )

                text_box = draw.textbbox((x1, y1), caption, font=font)
                text_y = max(0, y1 - (text_box[3] - text_box[1]) - 4)

                draw.rectangle(
                    [
                        x1,
                        text_y,
                        x1 + (text_box[2] - text_box[0]) + 6,
                        text_y + (text_box[3] - text_box[1]) + 4,
                    ],
                    fill="red",
                )

                draw.text(
                    (x1 + 3, text_y + 2),
                    caption,
                    fill="white",
                    font=font,
                )

            total_detections += len(boxes)

            # Add a project label.
            draw.text(
                (15, 15),
                "BDD100K | ResNet-18 + FPN",
                fill="yellow",
                font=font,
                stroke_width=1,
                stroke_fill="black",
            )

            writer.append_data(np.asarray(canvas))
            frame_count += 1

            if frame_count % 30 == 0:
                print(
                    f"Processed {frame_count} frames | "
                    f"Detections so far: {total_detections}"
                )

    finally:
        reader.close()
        writer.close()

    print("\nVIDEO INFERENCE COMPLETE")
    print(f"Frames processed: {frame_count}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections/frame: "
          f"{total_detections / max(frame_count, 1):.2f}")
    print(f"Saved video: {output_path.resolve()}")


if __name__ == "__main__":
    main()
