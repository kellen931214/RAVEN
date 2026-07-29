"""HSQR standalone verification / calibration runner tests — Issue #5.

Model-free tests for the runner layer: CLI surfaces, deterministic image
enumeration, duplicate rejection, per-image error containment, ROC/threshold
bookkeeping and the generation resume gates. The diffusion inversion itself is
stubbed, because what is under test here is the runner's bookkeeping — the
detector math is covered element-wise in ``test_hsqr_official_parity.py``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from PIL import Image

from utils.wm import runner_common, sfw_bundle, sfw_runtime
from utils.wm.hsqr_provider import (
    HSQR_SCORE_DEFINITION,
    OFFICIAL_BASE_KEY_SEED,
    OFFICIAL_PROFILE_NAME,
    HSQRProvider,
)
from utils.wm.sfw_bundle import SfwBundle, SfwBundleError


CPU = torch.device("cpu")
LATENT_SHAPE = (1, 4, 64, 64)
MODEL_KWARGS = {
    "modelid_target": "stabilityai/stable-diffusion-2-1-base",
    "model_revision": None,
    "scheduler_target": "DDIM",
    "resolution": 512,
}


def official_provider(**overrides) -> HSQRProvider:
    kwargs = dict(
        latent_shape=LATENT_SHAPE, device=CPU, hsqr_profile=OFFICIAL_PROFILE_NAME,
        hsqr_base_key_seed=OFFICIAL_BASE_KEY_SEED, hsqr_key_index=0, **MODEL_KWARGS,
    )
    kwargs.update(overrides)
    return HSQRProvider(**kwargs)


def write_image(path: Path, colour=(10, 20, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), colour).save(path)
    return path


class TestImageEnumeration(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_directory_order_is_deterministic_and_by_name(self):
        for name in ("b.png", "a.png", "c.jpg", "notes.txt"):
            write_image(self.tmp / name) if name.endswith((".png", ".jpg")) else \
                (self.tmp / name).write_text("x")
        images = sfw_runtime.enumerate_images(self.tmp)
        self.assertEqual([p.name for p in images], ["a.png", "b.png", "c.jpg"])

    def test_single_file_is_supported(self):
        path = write_image(self.tmp / "one.png")
        self.assertEqual(sfw_runtime.enumerate_images(path), [path])

    def test_missing_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            sfw_runtime.enumerate_images(self.tmp / "nope")

    def test_duplicate_inputs_are_rejected(self):
        path = write_image(self.tmp / "one.png")
        runner_common.assert_unique_inputs([path])
        with self.assertRaises(RuntimeError):
            runner_common.assert_unique_inputs([path, path])
        with self.assertRaises(RuntimeError):
            runner_common.assert_unique_inputs([path, self.tmp / "./one.png"])


class StubInversion:
    """Stands in for the diffusion inversion: maps an image to a fixed latent."""

    def __init__(self, latents_by_sha=None, default=None, fail_on=()):
        self.latents_by_sha = latents_by_sha or {}
        self.default = default
        self.fail_on = set(fail_on)
        self.calls = []

    def __call__(self, image, pipe_provider_target, num_inference_steps=None):
        self.calls.append(image)
        index = len(self.calls) - 1
        if index in self.fail_on:
            raise RuntimeError("stub inversion failure")
        latent = self.default
        return {
            "z0_torch": latent,
            "zT_torch": latent,
            "inversion_steps": 50,
            "recovered_latent_sha256": sfw_bundle.sha256_tensor(latent),
        }


class TestScoreImage(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.provider = official_provider()
        self.sample = self.provider.build_sample_latents(2024)

    def _score(self, latent, threshold_info, image_name="img.png", fail=False):
        path = write_image(self.tmp / image_name)
        stub = StubInversion(default=latent, fail_on=(0,) if fail else ())
        with mock.patch.object(self.provider, "invert_pil_image", stub):
            return sfw_runtime.score_image(self.provider, object(), path, threshold_info)

    def test_raw_distance_and_canonical_score_are_always_emitted(self):
        info = self.provider.resolve_threshold()
        row = self._score(self.sample["watermarked_latent"], info)
        self.assertEqual(row["status"], "ok")
        self.assertIsNotNone(row["hsqr_l1_distance"])
        self.assertEqual(row["hsqr_score"], -row["hsqr_l1_distance"])
        self.assertEqual(row["score_definition"], HSQR_SCORE_DEFINITION)
        self.assertEqual(row["score_direction"], "higher_is_watermarked")
        self.assertEqual(row["comparison_operator"], ">=")
        # No compatible threshold -> undecided, not "not embedded".
        self.assertIsNone(row["detection_success"])

    def test_watermark_identity_is_recorded_on_every_row(self):
        row = self._score(self.sample["watermarked_latent"], self.provider.resolve_threshold())
        self.assertEqual(row["selected_key_index"], 0)
        self.assertEqual(row["selected_key_seed"], 7433)
        self.assertEqual(row["payload_text"], "HSQR7433")
        self.assertEqual(row["selected_pattern_sha256"], self.provider.pattern_sha256())
        self.assertIsNotNone(row["image_sha256"])

    def test_decision_uses_the_score_against_the_threshold(self):
        provider = official_provider(hsqr_threshold=-60.0)
        self.provider = provider
        info = provider.resolve_threshold()
        wm = self._score(self.sample["watermarked_latent"], info, "wm.png")
        clean = self._score(self.sample["clean_latent"], info, "clean.png")
        self.assertTrue(wm["detection_success"])
        self.assertFalse(clean["detection_success"])
        self.assertEqual(wm["threshold"], -60.0)
        self.assertEqual(wm["distance_threshold"], 60.0)

    def test_inversion_failure_is_an_error_not_a_false_negative(self):
        row = self._score(self.sample["watermarked_latent"], self.provider.resolve_threshold(),
                          fail=True)
        self.assertEqual(row["status"], "error")
        self.assertIsNone(row["detection_success"])
        self.assertIsNone(row["hsqr_score"])
        self.assertIn("stub inversion failure", row["error"])

    def test_non_finite_latent_is_an_error_not_a_false_negative(self):
        broken = self.sample["watermarked_latent"].clone()
        broken[0, 0, 0, 0] = float("nan")
        row = self._score(broken, self.provider.resolve_threshold())
        self.assertEqual(row["status"], "error")
        self.assertIsNone(row["detection_success"])

    def test_shape_mismatch_is_an_error_not_a_false_negative(self):
        row = self._score(torch.zeros(1, 4, 32, 32), self.provider.resolve_threshold())
        self.assertEqual(row["status"], "error")
        self.assertIsNone(row["detection_success"])
        self.assertIn("latent shape", row["error"])

    def test_every_image_is_scored_with_its_own_latent(self):
        """Regression: a directory run must not reuse the first image's result."""
        paths = [write_image(self.tmp / f"{i}.png", (i, i, i)) for i in range(3)]
        latents = [
            self.sample["watermarked_latent"],
            self.sample["clean_latent"],
            self.provider.build_sample_latents(99)["clean_latent"],
        ]
        rows = []
        for path, latent in zip(paths, latents):
            stub = StubInversion(default=latent)
            with mock.patch.object(self.provider, "invert_pil_image", stub):
                rows.append(sfw_runtime.score_image(
                    self.provider, object(), path, self.provider.resolve_threshold()
                ))
        distances = [row["hsqr_l1_distance"] for row in rows]
        self.assertEqual(len(set(distances)), 3)
        self.assertLess(distances[0], distances[1])
        self.assertEqual([row["image_path"] for row in rows], [p.as_posix() for p in paths])


class TestRoc(unittest.TestCase):
    def test_official_roc_uses_negative_distance_scores(self):
        positives = [-30.0, -31.0, -29.5, -32.0]
        negatives = [-70.0, -71.0, -69.0, -72.0]
        roc = sfw_runtime.official_roc(positives, negatives, target_fpr=0.01)
        self.assertEqual(roc["score_definition"], HSQR_SCORE_DEFINITION)
        self.assertEqual(roc["score_direction"], "higher_is_watermarked")
        self.assertEqual(roc["comparison_operator"], ">=")
        self.assertEqual(roc["roc_auc"], 1.0)
        self.assertEqual(roc["tpr_at_target_fpr"], 1.0)
        self.assertEqual(roc["empirical_fpr"], 0.0)
        self.assertEqual(roc["positive_count"], 4)
        self.assertEqual(roc["negative_count"], 4)
        self.assertEqual(roc["target_fpr"], 0.01)

    def test_non_finite_score_is_rejected(self):
        with self.assertRaises(SfwBundleError):
            sfw_runtime.official_roc([float("inf")], [-1.0], target_fpr=0.01)

    def test_empty_cohort_is_rejected(self):
        with self.assertRaises(SfwBundleError):
            sfw_runtime.official_roc([], [-1.0], target_fpr=0.01)

    def test_report_label_requires_the_official_profile(self):
        self.assertEqual(
            sfw_runtime.cohort_report_label("paper_eval", True), "official_paper_evaluation"
        )
        self.assertEqual(
            sfw_runtime.cohort_report_label("paper_eval", False), "legacy_or_ablation_mode"
        )
        self.assertEqual(
            sfw_runtime.cohort_report_label("calibrate", True),
            "calibrated_deployment_verification",
        )


class TestResumeGates(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_run_manifest_with_a_different_config_is_rejected(self):
        path = self.tmp / "run_manifest.json"
        self.assertIsNone(
            runner_common.assert_run_manifest_compatible(path, "abc", method="HSQR")
        )
        path.write_text(json.dumps({"run_config_sha256": "abc"}), encoding="utf-8")
        self.assertIsNotNone(
            runner_common.assert_run_manifest_compatible(path, "abc", method="HSQR")
        )
        with self.assertRaises(RuntimeError):
            runner_common.assert_run_manifest_compatible(path, "def", method="HSQR")

    def test_resume_rejects_a_changed_seed_prompt_or_pattern(self):
        expected = {
            "sample_seed": 1,
            "prompt_sha256": "p",
            "run_config_sha256": "r",
            "selected_pattern_sha256": "s",
        }
        runner_common.assert_resumable("000000", dict(expected), expected, method="HSQR")
        for field, value in (
            ("sample_seed", 2), ("prompt_sha256", "q"),
            ("run_config_sha256", "z"), ("selected_pattern_sha256", "t"),
        ):
            stale = dict(expected)
            stale[field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                runner_common.assert_resumable("000000", stale, expected, method="HSQR")

    def test_resume_rejects_a_missing_field(self):
        with self.assertRaises(RuntimeError):
            runner_common.assert_resumable("000000", {}, {"sample_seed": 1}, method="HSQR")


class TestCliSurfaces(unittest.TestCase):
    def test_generation_parser_accepts_the_documented_command(self):
        import run_watermark

        args = run_watermark.build_parser().parse_args([
            "--wm_type", "HSQR",
            "--hsqr_profile", "official_sfwmark_sd21",
            "--hsqr_key_index", "0",
            "--num", "10",
            "--seed", "42",
            "--out_dir", "out/hsqr_generation",
        ])
        self.assertEqual(args.wm_type, "HSQR")
        self.assertEqual(args.hsqr_profile, "official_sfwmark_sd21")
        self.assertEqual(args.hsqr_key_index, 0)
        self.assertEqual(args.num, 10)
        self.assertEqual(args.seed, 42)

    def test_generation_parser_defaults_stay_legacy_for_the_formal_generators(self):
        import run_watermark

        args = run_watermark.build_parser().parse_args(["--wm_type", "HSQR"])
        self.assertEqual(args.hsqr_profile, "legacy_raven")
        self.assertEqual(args.hsqr_seed, 999999)

    def test_deployment_verify_parser_accepts_the_documented_command(self):
        import run_verify_watermark

        args = run_verify_watermark.build_hsqr_parser().parse_args([
            "--wm_type", "HSQR",
            "--mode", "deployment_verify",
            "--hsqr_bundle_dir", "out/hsqr_generation/hsqr_bundle",
            "--suspect_path", "image.png",
            "--out_dir", "out/hsqr_verification",
        ])
        self.assertEqual(args.mode, "deployment_verify")
        self.assertEqual(args.hsqr_bundle_dir, "out/hsqr_generation/hsqr_bundle")
        self.assertEqual(args.suspect_path, "image.png")

    def test_paper_eval_parser_accepts_the_documented_command(self):
        import run_verify_watermark

        args = run_verify_watermark.build_hsqr_parser().parse_args([
            "--wm_type", "HSQR",
            "--mode", "paper_eval",
            "--hsqr_bundle_dir", "bundle",
            "--positive_path", "pos",
            "--negative_path", "neg",
            "--target_fpr", "0.01",
            "--out_dir", "out/hsqr_calibration",
        ])
        self.assertEqual(args.mode, "paper_eval")
        self.assertEqual(args.target_fpr, 0.01)

    def test_threshold_artifact_flag_exists(self):
        import run_verify_watermark

        args = run_verify_watermark.build_hsqr_parser().parse_args([
            "--hsqr_bundle_dir", "b", "--suspect_path", "i.png",
            "--threshold_artifact", "out/hsqr_calibration/threshold.json",
        ])
        self.assertEqual(args.threshold_artifact, "out/hsqr_calibration/threshold.json")

    def test_verifier_requires_a_bundle(self):
        import run_verify_watermark

        with self.assertRaises(SystemExit):
            run_verify_watermark.main([
                "--wm_type", "HSQR", "--suspect_path", "x.png", "--out_dir", "out",
            ])

    def test_verifier_rejects_a_missing_bundle_before_loading_a_model(self):
        import run_verify_watermark

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SfwBundleError):
                run_verify_watermark.main([
                    "--wm_type", "HSQR",
                    "--hsqr_bundle_dir", str(Path(tmp) / "missing"),
                    "--suspect_path", str(write_image(Path(tmp) / "i.png")),
                    "--out_dir", str(Path(tmp) / "out"),
                ])

    def test_verifier_rejects_a_missing_suspect_path_before_loading_a_model(self):
        import run_verify_watermark

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            official_provider().create_bundle(tmp / "bundle")
            with self.assertRaises(FileNotFoundError):
                run_verify_watermark.main([
                    "--wm_type", "HSQR",
                    "--hsqr_bundle_dir", str(tmp / "bundle"),
                    "--suspect_path", str(tmp / "does_not_exist"),
                    "--out_dir", str(tmp / "out"),
                ])

    def test_gm_verification_is_unaffected_by_the_hsqr_dispatch(self):
        import run_verify_watermark

        args = run_verify_watermark.build_parser().parse_args([
            "--gm_bundle_dir", "b", "--suspect_path", "i.png",
        ])
        self.assertEqual(args.wm_type, "GM")
        self.assertEqual(args.mode, "verify")


if __name__ == "__main__":
    unittest.main()
