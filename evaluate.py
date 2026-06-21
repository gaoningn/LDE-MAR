import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from dcmreader import collect_sequence_pairs
from inference_utils import calculate_frame_metrics, predict_sequence, read_paired_sequences
from runtime_utils import (
    DEFAULT_DATA_ROOT,
    DEFAULT_WEIGHT_PATH,
    PROJECT_ROOT,
    load_model,
    resolve_device,
    validate_weight_file,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LDE-MAR on paired cine-MRI DICOM sequences.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_WEIGHT_PATH)
    parser.add_argument("--log-file", type=Path, default=PROJECT_ROOT / "outputs" / "test_metrics.tsv")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--style-dim", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    validate_weight_file(args.model_path)
    model = load_model(args.model_path, device, args.img_size, args.style_dim)
    model.eval()

    pairs = collect_sequence_pairs(args.data_root, args.max_subjects)
    rows = []
    all_metrics = []
    for index, (subject, clean_path, artifact_path) in enumerate(pairs, start=1):
        print(f"[{index}/{len(pairs)}] {subject}: {clean_path.name} <-> {artifact_path.name}")
        clean, artifact = read_paired_sequences(clean_path, artifact_path, args.img_size)
        prediction = predict_sequence(model, artifact, device, args.batch_size)
        frame_metrics = calculate_frame_metrics(prediction, clean)
        mean = frame_metrics.mean(axis=0)
        rows.append((subject, clean_path.name, artifact_path.name, len(frame_metrics), *mean))
        all_metrics.append(frame_metrics)
        print(f"  MAE={mean[0]:.6f}, SSIM={mean[1]:.6f}, PSNR={mean[2]:.6f}")

    overall = np.concatenate(all_metrics, axis=0).mean(axis=0)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("w", encoding="utf-8") as handle:
        handle.write(f"# time\t{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.write(f"# data_root\t{args.data_root.resolve()}\n")
        handle.write(f"# model_path\t{args.model_path.resolve()}\n")
        handle.write(f"# device\t{device}\n")
        handle.write("subject\toriginal_group\tartificial_group\tframes\tMAE\tSSIM\tPSNR\n")
        for row in rows:
            handle.write(
                f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t{row[4]:.6f}\t"
                f"{row[5]:.6f}\t{row[6]:.6f}\n"
            )
        handle.write(
            f"FRAME_WEIGHTED_MEAN\t-\t-\t{sum(row[3] for row in rows)}\t"
            f"{overall[0]:.6f}\t{overall[1]:.6f}\t{overall[2]:.6f}\n"
        )

    print("\n=== Frame-weighted mean ===")
    print(f"MAE: {overall[0]:.6f} | SSIM: {overall[1]:.6f} | PSNR: {overall[2]:.6f}")
    print(f"Saved metrics: {args.log_file}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as error:
        raise SystemExit(f"error: {error}")
