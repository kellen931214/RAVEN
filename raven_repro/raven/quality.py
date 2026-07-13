"""Optional heavyweight quality metrics with explicit model provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def openclip_text_image_scores(
    image_paths: Sequence[str | Path],
    prompts: Sequence[str],
    device: str = "cuda",
    model_name: str = "ViT-bigG-14",
    pretrained: str = "laion2b_s39b_b160k",
) -> dict:
    if len(image_paths) != len(prompts) or not image_paths:
        raise ValueError("CLIP requires equally sized, non-empty image and prompt lists")
    import open_clip
    import torch
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    scores = []
    model.eval()
    with torch.no_grad():
        for path, prompt in zip(image_paths, prompts):
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            text = tokenizer([prompt]).to(device)
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            scores.append(float((image_features * text_features).sum().cpu().item()))
    return {
        "model_name": model_name,
        "pretrained": pretrained,
        "metric": "prompt-image cosine similarity",
        "scores": scores,
        "mean": sum(scores) / len(scores),
    }


def clean_fid(reference_dir: str | Path, attacked_dir: str | Path, device: str = "cuda") -> dict:
    from cleanfid import fid

    value = fid.compute_fid(str(reference_dir), str(attacked_dir), device=device)
    return {
        "implementation": "clean-fid",
        "feature_extractor": "clean-fid default InceptionV3",
        "reference_dir": str(Path(reference_dir).resolve()),
        "attacked_dir": str(Path(attacked_dir).resolve()),
        "value": float(value),
    }


def torchmetrics_fid(
    reference_paths: Sequence[str | Path],
    attacked_paths: Sequence[str | Path],
    device: str = "cuda",
) -> dict:
    if not reference_paths or not attacked_paths:
        raise ValueError("FID requires non-empty reference and attacked image sets")
    if len(reference_paths) != len(attacked_paths):
        raise ValueError("paired FID audit expects equal reference and attacked counts")
    import numpy as np
    import torch
    from PIL import Image
    from torchmetrics.image.fid import FrechetInceptionDistance

    metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    for real, fake in zip(reference_paths, attacked_paths):
        real_tensor = torch.from_numpy(np.asarray(Image.open(real).convert("RGB")).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        fake_tensor = torch.from_numpy(np.asarray(Image.open(fake).convert("RGB")).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        metric.update(real_tensor.to(device), real=True)
        metric.update(fake_tensor.to(device), real=False)
    return {
        "implementation": "torchmetrics",
        "feature_extractor": "InceptionV3 2048",
        "value": float(metric.compute().cpu().item()),
    }
