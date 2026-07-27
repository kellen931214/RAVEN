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


# FID protocol. The primary reported value uses the TensorFlow FID protocol:
# the original TF Inception-2015-12-05 graph features with TensorFlow-compatible
# bilinear resizing, which is what the watermarking literature reports as "FID".
# clean-fid's ``legacy_tensorflow`` mode is that protocol, so no TensorFlow
# runtime is needed next to the torch/diffusers attack pipeline.
FID_PRIMARY_MODE = "legacy_tensorflow"
# Recorded alongside the primary value so results stay directly comparable with
# earlier runs, which reported clean-fid's own ``clean`` mode.
FID_SECONDARY_MODES: tuple[str, ...] = ("clean",)
FID_MODES: dict[str, str] = {
    "legacy_tensorflow": (
        "TF Inception-2015-12-05 pool3 features with TensorFlow-compatible "
        "bilinear resizing (original TensorFlow FID protocol)"
    ),
    "legacy_pytorch": (
        "pytorch-fid ported Inception-2015-12-05 weights with PIL bilinear resizing"
    ),
    "clean": (
        "clean-fid default: Inception-2015-12-05 features with clean-fid "
        "anti-aliased bicubic resizing"
    ),
}


def require_fid_mode(mode: str) -> str:
    """Fail closed on an unregistered FID mode instead of silently using another."""
    if mode not in FID_MODES:
        raise ValueError(f"unknown FID mode {mode!r}; known modes: {sorted(FID_MODES)}")
    return mode


def fid_protocol_descriptor(
    mode: str = FID_PRIMARY_MODE,
    secondary_modes: Sequence[str] = FID_SECONDARY_MODES,
) -> str:
    """Stable provenance string for the FID protocol actually used.

    Part of the recorded quality-config hash, so changing the primary or
    secondary FID protocol makes prior run configs fail closed instead of
    silently mixing two FID definitions under one metric name.
    """
    require_fid_mode(mode)
    secondary = [require_fid_mode(name) for name in secondary_modes if name != mode]
    text = f"clean-fid {mode} watermarked-vs-raven"
    if secondary:
        text += " (also recorded: " + ", ".join(sorted(secondary)) + ")"
    return text


def clean_fid(
    reference_dir: str | Path,
    attacked_dir: str | Path,
    device: str = "cuda",
    mode: str = FID_PRIMARY_MODE,
    secondary_modes: Sequence[str] = FID_SECONDARY_MODES,
) -> dict:
    """FID between two staged folders, primary value under the TF FID protocol."""
    import importlib.metadata

    from cleanfid import fid

    require_fid_mode(mode)
    modes = [mode, *[name for name in secondary_modes if name != mode]]
    values: dict[str, float] = {}
    for name in modes:
        values[require_fid_mode(name)] = float(
            fid.compute_fid(str(reference_dir), str(attacked_dir), device=device, mode=name)
        )
    return {
        "implementation": "clean-fid",
        "clean_fid_version": importlib.metadata.version("clean-fid"),
        "mode": mode,
        "primary_mode": mode,
        "secondary_modes": [name for name in modes if name != mode],
        "protocol": fid_protocol_descriptor(mode, secondary_modes),
        "feature_extractor": FID_MODES[mode],
        "mode_values": values,
        "mode_feature_extractors": {name: FID_MODES[name] for name in modes},
        "reference_dir": str(Path(reference_dir).resolve()),
        "attacked_dir": str(Path(attacked_dir).resolve()),
        "value": values[mode],
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
