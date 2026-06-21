import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from inference_utils import calculate_frame_metrics, predict_sequence, read_paired_sequences
from runtime_utils import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_CLEAN_PATH,
    DEFAULT_WEIGHT_PATH,
    PROJECT_ROOT,
    load_model,
    resolve_device,
    validate_weight_file,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run LDE-MAR on one paired cine-MRI DICOM sequence.")
    parser.add_argument("--clean-path", type=Path, default=DEFAULT_CLEAN_PATH)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_WEIGHT_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "example")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--style-dim", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    validate_weight_file(args.model_path)
    clean, artifact = read_paired_sequences(args.clean_path, args.artifact_path, args.img_size)
    model = load_model(args.model_path, device, args.img_size, args.style_dim)
    model.eval()

    prediction = predict_sequence(model, artifact, device, args.batch_size)
    metrics = calculate_frame_metrics(prediction, clean)
    directories = {
        name: args.output_dir / name for name in ("artifact", "clean", "prediction")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
        for old_image in directory.glob("*.png"):
            old_image.unlink()

    for index in range(clean.size(0)):
        images = {
            "artifact": artifact[index, 0].numpy(),
            "clean": clean[index, 0].numpy(),
            "prediction": prediction[index, 0].numpy(),
        }
        for name, image in images.items():
            plt.imsave(directories[name] / f"{index:04d}.png", image, cmap="gray", vmin=-1, vmax=1)

    mean = metrics.mean(axis=0)
    print(f"Device: {device}")
    print(f"Frames: {clean.size(0)}")
    print(f"MAE: {mean[0]:.6f} | SSIM: {mean[1]:.6f} | PSNR: {mean[2]:.6f}")
    print(f"Saved images: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as error:
        raise SystemExit(f"error: {error}")
