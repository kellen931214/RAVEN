"""Focused CPU-only HSTR/SFWMark official parity and runner tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image

from utils.wm import sfw_bundle, sfw_runtime
from utils.wm.sfw_bundle import SfwBundle, SfwBundleError
from utils.wm.hstr_provider import (
    HSTR_SCORE_DEFINITION,
    HSTR_TR_ONLY_SCORE_DEFINITION,
    OFFICIAL_BASE_KEY_SEED,
    OFFICIAL_HSTR_PROFILE,
    OFFICIAL_MATH_TR_ONLY_PROFILE,
    HSTRProvider,
    apply_arg_defaults,
)

CPU = torch.device("cpu")


def provider_kwargs(**overrides):
    kwargs = dict(
        latent_shape=(1, 4, 64, 64),
        device=CPU,
        hstr_profile=OFFICIAL_HSTR_PROFILE,
        hstr_key_index=0,
        hstr_rng_device="cpu",
        modelid_target="stabilityai/stable-diffusion-2-1-base",
        scheduler_target="DDIM",
        resolution=512,
    )
    kwargs.update(overrides)
    return kwargs


class OfficialPatternTests(unittest.TestCase):
    def test_key_index_maps_to_frozen_seed_base(self):
        provider = HSTRProvider(**provider_kwargs(hstr_key_index=17, fix_gt=999))
        self.assertEqual(provider.selected_key_seed, OFFICIAL_BASE_KEY_SEED + 17)
        self.assertEqual(provider.key_index, 17)

    def test_selected_patterns_match_official_source_fixtures(self):
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "hstr_official_fixtures.json"
        fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))
        for index in (0, 1, 1024, 2047):
            provider = HSTRProvider(**provider_kwargs(hstr_key_index=index))
            expected_sha = fixtures["keys"][str(index)]["selected_pattern"]["sha256"]
            self.assertEqual(sfw_bundle.sha256_tensor(provider.gt_patch.cpu()), expected_sha)

    def test_injected_latent_scores_as_watermark_with_per_image_scores(self):
        provider = HSTRProvider(**provider_kwargs(hstr_key_index=3))
        clean_a = provider.sample_base_latent(123)
        clean_b = provider.sample_base_latent(124)
        self.assertNotEqual(sfw_bundle.sha256_tensor(clean_a), sfw_bundle.sha256_tensor(clean_b))
        wm_a = provider.get_wm_latents(latents_clean=clean_a)["zT_torch"]
        wm_b = provider.get_wm_latents(latents_clean=clean_b)["zT_torch"]
        scores = provider.get_accuracies(torch.cat([wm_a, wm_b], dim=0))
        self.assertEqual(len(scores["hstr_score"]), 2)
        self.assertEqual(scores["score_definition"], HSTR_SCORE_DEFINITION)
        for value in scores["hstr_channel_min_l1"]:
            self.assertLess(value, 1e-5)
        for score in scores["hstr_score"]:
            self.assertGreater(score, -1e-5)


class BundleTests(unittest.TestCase):
    def test_bundle_round_trip_and_threshold_binding(self):
        with TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            provider = HSTRProvider(**provider_kwargs(hstr_bundle_dir=str(bundle_dir), hstr_create_bundle=True))
            self.assertTrue((bundle_dir / "manifest.json").exists())
            self.assertTrue((bundle_dir / "selected_pattern.pt").exists())
            reloaded = HSTRProvider(**provider_kwargs(hstr_bundle_dir=str(bundle_dir)))
            self.assertTrue(torch.equal(provider.gt_patch.cpu(), reloaded.gt_patch.cpu()))
            artifact = sfw_bundle.build_threshold_artifact(
                threshold=-0.5,
                binding=reloaded.binding_config(),
                score_definition=HSTR_SCORE_DEFINITION,
                report_label="calibrated_deployment_verification",
                method="HSTR",
                target_fpr=0.01,
                empirical_fpr=0.0,
                tpr_at_target_fpr=1.0,
                roc_auc=1.0,
                positive_count=10,
                negative_count=10,
                threshold_source="cohort_calibration",
            )
            reloaded.bundle.save_threshold(artifact)
            threshold = reloaded.resolve_threshold()
            self.assertTrue(threshold["threshold_available"])
            self.assertEqual(threshold["threshold"], -0.5)

    def test_incompatible_bundle_fails_closed(self):
        with TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            HSTRProvider(**provider_kwargs(hstr_bundle_dir=str(bundle_dir), hstr_create_bundle=True))
            with self.assertRaises(SfwBundleError):
                HSTRProvider(**provider_kwargs(hstr_key_index=1, hstr_bundle_dir=str(bundle_dir)))


class RuntimeTests(unittest.TestCase):
    def test_roc_uses_higher_score_and_strict_fpr(self):
        roc = sfw_runtime.hstr_official_roc([0.9, 0.8, 0.7], [0.1, 0.2, 0.3], 0.01)
        self.assertEqual(roc["comparison_operator"], ">=")
        self.assertEqual(roc["score_direction"], "higher_is_watermarked")
        self.assertEqual(roc["tpr_at_target_fpr"], 1.0)
        self.assertLessEqual(roc["empirical_fpr"], 0.01)

    def test_corrupt_image_is_error_not_negative_detection(self):
        with TemporaryDirectory() as tmp:
            provider = HSTRProvider(**provider_kwargs(hstr_bundle_dir=str(Path(tmp) / "bundle"), hstr_create_bundle=True))
            corrupt = Path(tmp) / "corrupt.png"
            corrupt.write_bytes(b"not an image")
            row = sfw_runtime.hstr_score_image(provider, None, corrupt, {"threshold": -1.0, "comparison_operator": ">=", "score_direction": "higher_is_watermarked"})
            self.assertEqual(row["status"], "error")
            self.assertIsNone(row["detection_success"])
            self.assertIsNotNone(row["image_sha256"])

    def test_pairing_from_generation_sidecars(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pos = root / "images" / "watermarked"; neg = root / "images" / "no_watermark"; meta = root / "sample_metadata"
            pos.mkdir(parents=True); neg.mkdir(parents=True); meta.mkdir()
            Image.new("RGB", (8, 8), "red").save(pos / "000000.png")
            Image.new("RGB", (8, 8), "blue").save(neg / "000000.png")
            sidecar = {"sample_id": 0, "prompt_sha256": "a" * 64, "sample_seed": 7, "protocol": "hstr_official_sfwmark_paired_direct_generation"}
            (meta / "000000.json").write_text(json.dumps(sidecar), encoding="utf-8")
            pairing = sfw_runtime.resolve_pairing([pos / "000000.png"], [neg / "000000.png"])
            self.assertTrue(pairing["paired"])
            self.assertEqual(pairing["pair_count"] if "pair_count" in pairing else len(pairing["pairs"]), 1)


class ParserProfileTests(unittest.TestCase):
    def test_official_profile_defaults_are_applied(self):
        import run_watermark

        argv = ["--wm_type", "HSTR", "--hstr_profile", OFFICIAL_HSTR_PROFILE]
        args = run_watermark.build_parser().parse_args(argv)
        args.modelid_target = "stabilityai/stable-diffusion-xl-base-1.0"
        args.scheduler_target = "DPM"
        args.resolution = 1024
        info = apply_arg_defaults(args, argv)
        self.assertEqual(info["profile"], OFFICIAL_HSTR_PROFILE)
        self.assertEqual(args.modelid_target, "stabilityai/stable-diffusion-2-1-base")
        self.assertEqual(args.scheduler_target, "DDIM")
        self.assertEqual(args.resolution, 512)


class TrOnlyAblationTests(unittest.TestCase):
    def tr_provider(self, **overrides):
        return HSTRProvider(**provider_kwargs(
            hstr_profile=OFFICIAL_MATH_TR_ONLY_PROFILE,
            **overrides,
        ))

    def test_official_math_tr_only_injects_only_channel_3_and_scores_only_channel_3(self):
        provider = self.tr_provider(hstr_key_index=3)
        self.assertTrue(provider.uses_official_math)
        self.assertEqual(provider.provider_config()["watermark_channels"], [3])
        self.assertEqual(provider.provider_config()["heterogeneous_channels"], [])
        self.assertEqual(provider.provider_config()["pattern_variant"], "tr_only")
        self.assertEqual(provider.provider_config()["score_mode"], "center_channel_3_complex_l1")
        self.assertFalse(provider.masks[:, 0].any())
        self.assertTrue(provider.masks[:, 3].any())
        self.assertEqual(tuple(provider.watermark_region_mask_hstr.shape), (1, 64, 64))

        clean = provider.sample_base_latent(123)
        watermarked = provider.get_wm_latents(latents_clean=clean)["zT_torch"]
        self.assertTrue(torch.equal(clean[:, 0], watermarked[:, 0]))
        self.assertFalse(torch.equal(clean[:, 3], watermarked[:, 3]))
        scores = provider.get_accuracies(watermarked)
        self.assertEqual(scores["hstr_channel_0_l1"], [None])
        self.assertEqual(scores["hstr_channel_min_l1"], [None])
        self.assertEqual(scores["score_definition"], HSTR_TR_ONLY_SCORE_DEFINITION)
        self.assertEqual(scores["hstr_score"], [-scores["hstr_channel_3_l1"][0]])

    def test_tr_only_bundle_and_threshold_are_not_interchangeable_with_official_hstr(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            official = HSTRProvider(**provider_kwargs(hstr_bundle_dir=str(root / "official"), hstr_create_bundle=True))
            tr_only = self.tr_provider(hstr_bundle_dir=str(root / "tr_only"), hstr_create_bundle=True)
            self.assertNotEqual(official.provider_config_sha256(), tr_only.provider_config_sha256())
            with self.assertRaises(SfwBundleError):
                self.tr_provider(hstr_bundle_dir=str(root / "official"))
            with self.assertRaises(SfwBundleError):
                HSTRProvider(**provider_kwargs(hstr_bundle_dir=str(root / "tr_only")))

            official_threshold = sfw_bundle.build_threshold_artifact(
                threshold=-0.5,
                binding=official.binding_config(),
                score_definition=HSTR_SCORE_DEFINITION,
                report_label="calibrated_deployment_verification",
                method="HSTR",
                threshold_source="cohort_calibration",
            )
            tr_threshold = sfw_bundle.build_threshold_artifact(
                threshold=-0.5,
                binding=tr_only.binding_config(),
                score_definition=HSTR_TR_ONLY_SCORE_DEFINITION,
                report_label="calibrated_deployment_verification",
                method="HSTR",
                threshold_source="cohort_calibration",
            )
            with self.assertRaises(SfwBundleError):
                sfw_bundle.assert_threshold_compatible(official_threshold, tr_only.binding_config())
            with self.assertRaises(SfwBundleError):
                sfw_bundle.assert_threshold_compatible(tr_threshold, official.binding_config())

    def test_tr_only_profile_gets_official_math_defaults_and_ablation_labels(self):
        import run_watermark

        argv = ["--wm_type", "HSTR", "--hstr_profile", OFFICIAL_MATH_TR_ONLY_PROFILE]
        args = run_watermark.build_parser().parse_args(argv)
        args.modelid_target = "stabilityai/stable-diffusion-xl-base-1.0"
        args.scheduler_target = "DPM"
        args.resolution = 1024
        apply_arg_defaults(args, argv)
        self.assertEqual(args.modelid_target, "stabilityai/stable-diffusion-2-1-base")
        self.assertEqual(args.scheduler_target, "DDIM")
        self.assertEqual(args.resolution, 512)
        provider = self.tr_provider()
        self.assertEqual(provider.generation_report_label, "legacy_or_ablation_mode")
        self.assertEqual(provider.generation_protocol, "hstr_tr_only_sfwmark_ablation_paired_direct_generation")


if __name__ == "__main__":
    unittest.main()
