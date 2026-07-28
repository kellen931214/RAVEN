"""Verification, resume, ROC and profile tests for the GaussMarker runners."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image

from utils.wm import gm_bundle, gm_runtime
from utils.wm.gm_bundle import GmBundle, GmBundleError
from utils.wm.gm_provider import (
    GM_OFFICIAL_SD21_PROFILE,
    GM_SCORE_DEFINITION,
    GmProvider,
    apply_arg_defaults,
)


CPU = torch.device("cpu")


def provider_kwargs(**overrides):
    kwargs = dict(
        latent_shape=(1, 4, 64, 64),
        device=CPU,
        gm_torch_dtype="float32",
        gm_use_gnr=False,
        gm_use_classifier=False,
        gm_watermark_bits_seed=4242,
        modelid_target="stabilityai/stable-diffusion-2-1-base",
        scheduler_target="DPM",
    )
    kwargs.update(overrides)
    return kwargs


class ImageEnumerationTests(unittest.TestCase):
    def test_single_file_and_sorted_directory(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ["b.png", "a.jpg", "c.jpeg", "d.webp", "notes.txt"]
            for name in names:
                (root / name).write_bytes(b"x")
            found = [p.name for p in gm_runtime.enumerate_images(root)]
            self.assertEqual(found, ["a.jpg", "b.png", "c.jpeg", "d.webp"])
            self.assertEqual(
                [p.name for p in gm_runtime.enumerate_images(root / "b.png")], ["b.png"]
            )

    def test_missing_path_fails_closed(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                gm_runtime.enumerate_images(Path(tmp) / "absent")


class SuspectImageErrorTests(unittest.TestCase):
    def test_corrupt_image_produces_status_error_not_a_negative_detection(self):
        with TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            provider = GmProvider(
                **provider_kwargs(gm_bundle_dir=str(bundle_dir), gm_create_bundle=True)
            )
            corrupt = Path(tmp) / "corrupt.png"
            corrupt.write_bytes(b"not really a png")

            threshold_info = {
                "threshold": 0.9,
                "threshold_source": "user_supplied",
                "score_direction": "higher_is_watermarked",
                "comparison_operator": ">=",
            }
            row = gm_runtime.score_image(provider, None, corrupt, threshold_info)
            self.assertEqual(row["status"], "error")
            self.assertIsNone(row["detection_success"])
            self.assertIsNotNone(row["error"])
            self.assertIsNotNone(row["image_sha256"])

    def test_verification_never_modifies_the_bundle(self):
        with TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            GmProvider(**provider_kwargs(gm_bundle_dir=str(bundle_dir), gm_create_bundle=True))
            before = GmBundle.load(bundle_dir).artifact_mtimes()

            verifier = GmProvider(**provider_kwargs(gm_bundle_dir=str(bundle_dir)))
            latent = torch.randn(1, 4, 64, 64, generator=torch.Generator().manual_seed(1))
            detection = verifier.detect_from_latent(latent)
            self.assertIsNotNone(detection["raw_bit_accuracy"])
            self.assertIsNotNone(detection["raw_ring_l1"])

            self.assertEqual(before, GmBundle.load(bundle_dir).artifact_mtimes())

    def test_detection_rejects_non_finite_latents(self):
        with TemporaryDirectory() as tmp:
            provider = GmProvider(
                **provider_kwargs(gm_bundle_dir=str(Path(tmp) / "b"), gm_create_bundle=True)
            )
            latent = torch.zeros(1, 4, 64, 64)
            latent[0, 0, 0, 0] = float("nan")
            with self.assertRaises(ValueError):
                provider.detect_from_latent(latent)

    def test_raw_scores_are_always_emitted(self):
        with TemporaryDirectory() as tmp:
            provider = GmProvider(
                **provider_kwargs(gm_bundle_dir=str(Path(tmp) / "b"), gm_create_bundle=True)
            )
            latent = provider.inject_ring(provider.sample_pre_frequency_latent(1))
            results = provider.get_accuracies(latent)
            self.assertIsNotNone(results["gm_raw_bit_accuracy"])
            self.assertIsNotNone(results["gm_raw_ring_l1"])
            self.assertIsNotNone(results["gm_ring_classifier_feature"])
            self.assertIsNone(results["detection_success"])  # no GNR/classifier -> no decision
            self.assertEqual(results["gm_comparison_operator"], ">=")
            self.assertEqual(
                results["gm_official_reference_commit"], gm_bundle.OFFICIAL_GAUSSMARKER_COMMIT
            )


class ResumeTests(unittest.TestCase):
    EXPECTED = {
        "sample_seed": 5,
        "prompt_sha256": "a" * 64,
        "run_config_sha256": "b" * 64,
        "gm_bundle_config_sha256": "c" * 64,
    }

    def test_identical_sample_resumes(self):
        gm_runtime.assert_resumable("000005", dict(self.EXPECTED), self.EXPECTED)

    def test_changed_seed_rejects_resume(self):
        existing = dict(self.EXPECTED, sample_seed=6)
        with self.assertRaises(RuntimeError):
            gm_runtime.assert_resumable("000005", existing, self.EXPECTED)

    def test_changed_bundle_rejects_resume(self):
        existing = dict(self.EXPECTED, gm_bundle_config_sha256="d" * 64)
        with self.assertRaises(RuntimeError):
            gm_runtime.assert_resumable("000005", existing, self.EXPECTED)

    def test_missing_provenance_rejects_resume(self):
        existing = {k: v for k, v in self.EXPECTED.items() if k != "run_config_sha256"}
        with self.assertRaises(RuntimeError):
            gm_runtime.assert_resumable("000005", existing, self.EXPECTED)

    def test_same_sample_id_and_seed_reproduce_the_same_latent(self):
        with TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            first = GmProvider(
                **provider_kwargs(gm_bundle_dir=str(bundle_dir), gm_create_bundle=True)
            )
            hashes = [first.build_sample_latents(7 + i)["post_injection_latent_sha256"]
                      for i in range(3)]
            # a fresh process would reload the same bundle:
            second = GmProvider(**provider_kwargs(gm_bundle_dir=str(bundle_dir)))
            again = [second.build_sample_latents(7 + i)["post_injection_latent_sha256"]
                     for i in range(3)]
            self.assertEqual(hashes, again)
            self.assertEqual(len(set(hashes)), 3)


class OfficialRocTests(unittest.TestCase):
    def setUp(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is not installed")

    def test_roc_threshold_and_operator(self):
        positives = [0.90 + 0.001 * i for i in range(100)]
        negatives = [0.10 + 0.001 * i for i in range(100)]
        roc = gm_runtime.official_roc(positives, negatives, 0.01)
        self.assertAlmostEqual(roc["roc_auc"], 1.0, places=6)
        self.assertEqual(roc["comparison_operator"], ">=")
        self.assertEqual(roc["score_direction"], "higher_is_watermarked")
        self.assertEqual(roc["positive_count"], 100)
        self.assertEqual(roc["negative_count"], 100)
        self.assertLessEqual(roc["empirical_fpr"], 0.01)
        self.assertEqual(roc["tpr_at_target_fpr"], 1.0)
        # every positive is at or above the threshold, no negative is
        self.assertTrue(all(score >= roc["threshold"] for score in positives))
        self.assertTrue(all(score < roc["threshold"] for score in negatives))

    def test_empty_cohort_fails_closed(self):
        with self.assertRaises(GmBundleError):
            gm_runtime.official_roc([], [0.1], 0.01)

    def test_non_finite_score_fails_closed(self):
        with self.assertRaises(GmBundleError):
            gm_runtime.official_roc([float("nan")], [0.1], 0.01)


class ProfileTests(unittest.TestCase):
    def _parse(self, argv):
        """Parse with the real standalone runner parser (generic defaults included)."""
        import run_verify_watermark

        return run_verify_watermark.build_parser().parse_args(argv)

    def test_official_profile_is_applied(self):
        argv = ["--gm_profile", "official_sd21"]
        args = self._parse(argv)
        # Simulate generic RAVEN-wide defaults that must not win over the profile.
        args.modelid_target = "stabilityai/stable-diffusion-xl-base-1.0"
        args.scheduler_target = "DDIM"
        args.resolution = 1024
        args.num_inference_steps_target = 20
        args.guidance_scale_target = 3.5
        args.model_revision = None
        args.w_radius = 10
        args.w_channel = 0
        info = apply_arg_defaults(args, argv)

        self.assertTrue(info["is_official"])
        self.assertEqual(info["overrides"], {})
        for field, value in GM_OFFICIAL_SD21_PROFILE.items():
            self.assertEqual(getattr(args, field), value, field)
        self.assertEqual(args.modelid_target, "stabilityai/stable-diffusion-2-1-base")
        self.assertEqual(args.scheduler_target, "DPM")
        self.assertEqual(args.resolution, 512)
        self.assertEqual(args.num_inference_steps_target, 50)
        self.assertEqual(args.guidance_scale_target, 7.5)
        self.assertEqual(args.gm_inversion_guidance, 1.0)
        self.assertEqual(args.gm_inversion_prompt, "")
        self.assertEqual(args.gm_target_fpr, 0.01)
        self.assertEqual(args.gm_model_nf, 128)
        self.assertEqual(args.gm_classifier_type, 0)
        self.assertEqual(args.w_seed, 999999)
        self.assertEqual(args.w_channel, 3)
        self.assertEqual(args.w_radius, 4)

    def test_explicit_override_marks_the_run_as_an_ablation(self):
        argv = ["--gm_profile", "official_sd21", "--w_radius", "8"]
        args = self._parse(argv)
        info = apply_arg_defaults(args, argv)

        self.assertFalse(info["is_official"])
        self.assertIn("w_radius", info["overrides"])
        self.assertEqual(args.w_radius, 8)
        self.assertEqual(args.w_channel, 3)  # unrelated official values still applied

    def test_negation_switch_counts_as_an_override(self):
        argv = ["--gm_profile", "official_sd21", "--gm_vae_posterior_mean"]
        args = self._parse(argv)
        info = apply_arg_defaults(args, argv)

        self.assertFalse(info["is_official"])
        self.assertIn("gm_vae_sample", info["overrides"])
        self.assertFalse(args.gm_vae_sample)

    def test_legacy_profile_never_claims_official_parity(self):
        argv = ["--gm_profile", "legacy"]
        args = self._parse(argv)
        info = apply_arg_defaults(args, argv)
        self.assertFalse(info["is_official"])
        self.assertEqual(info["applied"], {})

    def test_ablation_provider_reports_a_legacy_label(self):
        with TemporaryDirectory() as tmp:
            provider = GmProvider(
                **provider_kwargs(
                    gm_bundle_dir=str(Path(tmp) / "b"),
                    gm_create_bundle=True,
                    gm_profile_is_official=False,
                )
            )
            self.assertEqual(
                provider.resolve_threshold()["report_label"], "legacy_or_ablation_mode"
            )
            self.assertFalse(provider.bundle.manifest["profile_is_official"])


class ReportLabelTests(unittest.TestCase):
    def test_all_labels_are_registered(self):
        self.assertEqual(
            set(gm_bundle.REPORT_LABELS),
            {
                "official_paper_evaluation",
                "official_profile_raw_scores",
                "calibrated_deployment_verification",
                "deployment_verification_extension",
                "user_supplied_threshold",
                "legacy_or_ablation_mode",
            },
        )

    def test_unknown_label_is_rejected(self):
        artifact = {
            "schema": gm_bundle.GM_THRESHOLD_SCHEMA,
            "threshold": 0.5,
            "score_definition": GM_SCORE_DEFINITION,
            "score_direction": "higher_is_watermarked",
            "comparison_operator": ">=",
            "threshold_source": "cohort_calibration",
            "report_label": "exact_paper_evaluation_of_one_image",
        }
        with self.assertRaises(GmBundleError):
            gm_bundle.validate_threshold_artifact(artifact)


class VerifyRunnerCliTests(unittest.TestCase):
    def test_verify_requires_a_bundle(self):
        import run_verify_watermark

        with self.assertRaises(SystemExit):
            run_verify_watermark.main(["--suspect_path", "/tmp", "--out_dir", "/tmp/out"])

    def test_paper_eval_requires_matched_cohorts(self):
        import run_verify_watermark

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle_dir = root / "bundle"
            GmProvider(**provider_kwargs(gm_bundle_dir=str(bundle_dir), gm_create_bundle=True))
            pos, neg = root / "pos", root / "neg"
            pos.mkdir()
            neg.mkdir()
            blank = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
            blank.save(pos / "0.png")
            blank.save(pos / "1.png")
            blank.save(neg / "0.png")
            with self.assertRaises(SystemExit):
                run_verify_watermark.main([
                    "--mode", "paper_eval",
                    "--gm_bundle_dir", str(bundle_dir),
                    "--positive_path", str(pos),
                    "--negative_path", str(neg),
                    "--out_dir", str(root / "out"),
                    "--gm_torch_dtype", "float32",
                ])


if __name__ == "__main__":
    unittest.main()
