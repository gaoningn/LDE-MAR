from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from dcmreader import read_dcm


def read_paired_sequences(clean_path, artifact_path, image_size=256):
    clean = read_dcm(clean_path, target_size=(image_size, image_size))
    artifact = read_dcm(artifact_path, target_size=(image_size, image_size))
    if clean.size(0) != artifact.size(0):
        raise ValueError(
            f"Paired sequences have different frame counts: {Path(clean_path)} ({clean.size(0)}) vs "
            f"{Path(artifact_path)} ({artifact.size(0)})"
        )
    if clean.size(0) == 0:
        raise ValueError(f"Paired sequences contain no frames: {clean_path}, {artifact_path}")
    return clean, artifact


def predict_sequence(model, artifact, device, batch_size=1):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    source = artifact[0:1].to(device)
    predictions = []
    with torch.inference_mode():
        for start in range(0, artifact.size(0), batch_size):
            target = artifact[start:start + batch_size].to(device)
            prediction, _ = model.pure_motion_img(source, target)
            predictions.append(prediction.detach().cpu())
    return torch.cat(predictions, dim=0)


def calculate_frame_metrics(prediction, clean):
    if prediction.shape != clean.shape:
        raise ValueError(f"Prediction shape {prediction.shape} does not match clean shape {clean.shape}")
    values = []
    for index in range(clean.size(0)):
        predicted_frame = prediction[index, 0].numpy()
        clean_frame = clean[index, 0].numpy()
        values.append(
            (
                float(np.mean(np.abs(predicted_frame - clean_frame))),
                float(structural_similarity(predicted_frame, clean_frame, data_range=2.0)),
                float(peak_signal_noise_ratio(clean_frame, predicted_frame, data_range=2.0)),
            )
        )
    metrics = np.asarray(values, dtype=np.float64)
    if not np.isfinite(metrics).all():
        raise RuntimeError("Evaluation produced non-finite MAE, SSIM, or PSNR values")
    return metrics
