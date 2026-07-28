"""GM bundle persistence, official interchange, fail-closed and threshold tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

import gm_official_reference as official
from utils.wm import gm_bundle
from utils.wm.gm_bundle import GmBundle, GmBundleError
from utils.wm.gm_provider import GM_SCORE_DEFINITION, GmProvider


CPU = torch.device("cpu")
FIXED_KEY = bytes(range(1, 33))
FIXED_NONCE = bytes(range(1, 13))


def provider_kwargs(**overrides):
    kwargs = dict(
        latent_shape=(1, 4, 64, 64),
        device=CPU,
        gm_torch_dtype="float32",
        gm_use_gnr=False,
        gm_use_classifier=False,
        gm_watermark_bits_seed=99,
        modelid_target="stabilityai/stable-diffusion-2-1-base",
        scheduler_target="DPM",
    )
    kwargs.update(overrides)
    return kwargs


def make_bundle(directory) -> GmProvider:
    return GmProvider(**provider_kwargs(gm_bundle_dir=str(directory), gm_create_bundle=True))


class BundleCreationTests(unittest.TestCase):
    def test_create_load_round_trip(self):
        with TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            provider = make_bundle(bundle_dir)
            self.assertEqual(provider.state_source, "bundle_created")
            for name in ("manifest.json", "w1.pth", "w2.pth"):
                self.assertTrue((bundle_dir / name).exists(), name)
            self.assertFalse((bundle_dir / "threshold.json").exists())

            reloaded = GmProvider(**provider_kwargs(gm_bundle_dir=str(bundle_dir)))
            self.assertEqual(reloaded.state_source, "bundle")
            self.assertTrue(torch.equal(provider.watermark, reloaded.watermark))
            np.testing.assert_array_equal(provider.m_flat, reloaded.m_flat)
            self.assertEqual(provider.key, reloaded.key)
            self.assertEqual(provider.nonce, reloaded.nonce)
            self.assertTrue(torch.equal(provider.gt_patch, reloaded.gt_patch))

    def test_manifest_hash_survives_reload(self):
        with TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            provider = make_bundle(bundle_dir)
            manifest = json.loads((bundle_dir / "manifest.json").read_text())
            self.assertEqual(
                GmBundle.config_sha256(manifest), manifest["bundle_config_sha256"]
            )
            self.assertEqual(
                manifest["official_reference_commit"], gm_bundle.OFFICIAL_GAUSSMARKER_COMMIT
            )
            for field in ("model_id", "scheduler", "torch_dtype", "resolution",
                          "inversion_steps", "inversion_guidance_scale", "vae_sample",
                          "vae_scaling_factor", "gnr_sha256", "classifier_sha256",
                          "git_commit", "latent_shape", "w_radius", "w_channel"):
                self.assertIn(field, manifest, field)
            self.assertIsNotNone(provider.bundle)

    def test_manifest_never_contains_the_secret_key_or_nonce(self):
        with TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            provider = make_bundle(bundle_dir)
            text = (bundle_dir / "manifest.json").read_text()
            self.assertNotIn(provider.key.hex(), text)
            self.assertNotIn(provider.nonce.hex(), text)
            self.assertIn("key_sha256", text)
            self.assertIn("nonce_sha256", text)

    def test_existing_bundle_is_never_silently_regenerated(self):
        with TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            make_bundle(bundle_dir)
            before = GmBundle.load(bundle_dir).artifact_mtimes()
            # A second "create" run must reuse, not overwrite.
            reused = make_bundle(bundle_dir)
            self.assertEqual(reused.state_source, "bundle")
            self.assertEqual(before, GmBundle.load(bundle_dir).artifact_mtimes())
            with self.assertRaises(GmBundleError):
                GmBundle.create(bundle_dir, reused.bundle.w1, reused.bundle.w2, {})

    def test_partial_bundle_fails_closed(self):
        with TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            make_bundle(bundle_dir)
            (bundle_dir / "w2.pth").unlink()
            with self.assertRaises(GmBundleError):
                GmBundle.load(bundle_dir)
            with self.assertRaises(GmBundleError):
                make_bundle(bundle_dir)

    def test_corrupt_artifact_fails_closed(self):
        with TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            make_bundle(bundle_dir)
            state = gm_bundle.load_official_w1(bundle_dir / "w1.pth")
            state["w"] = torch.zeros_like(state["w"])
            (bundle_dir / "w1.pth").unlink()
            gm_bundle.save_official_w1(bundle_dir / "w1.pth", state)
            with self.assertRaises(GmBundleError):
                GmBundle.load(bundle_dir)

    def test_edited_manifest_fails_closed(self):
        with TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            make_bundle(bundle_dir)
            manifest = json.loads((bundle_dir / "manifest.json").read_text())
            manifest["w_radius"] = 8
            (bundle_dir / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(GmBundleError):
                GmBundle.load(bundle_dir)

    def test_incompatible_configuration_fails_closed(self):
        with TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            make_bundle(bundle_dir)
            with self.assertRaises(GmBundleError):
                GmProvider(**provider_kwargs(gm_bundle_dir=str(bundle_dir), w_radius=8))

    def test_verification_requires_an_existing_bundle(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(GmBundleError):
                GmProvider(**provider_kwargs(gm_bundle_dir=str(Path(tmp) / "absent")))


class OfficialInterchangeTests(unittest.TestCase):
    def test_raven_created_w1_w2_load_with_official_logic(self):
        with TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            provider = make_bundle(bundle_dir)

            raw = torch.load(bundle_dir / "w1.pth", map_location="cpu", weights_only=False)
            self.assertIsInstance(raw["w"], torch.Tensor)
            self.assertIsInstance(raw["m"], np.ndarray)
            self.assertEqual(raw["m"].shape, (16384,))
            self.assertIsInstance(raw["key"], bytes)
            self.assertEqual(len(raw["key"]), 32)
            self.assertIsInstance(raw["nonce"], bytes)
            self.assertEqual(len(raw["nonce"]), 12)

            reference = official.Gaussian_Shading_chacha(
                1, 8, 8, 1e-6, 1000000,
                watermark=raw["w"], m=raw["m"], key=raw["key"], nonce=raw["nonce"],
            )
            official.set_random_seed(3)
            latent, m_tensor = reference.create_watermark_and_return_w_m()
            self.assertEqual(tuple(latent.shape), (1, 4, 64, 64))
            self.assertTrue(torch.equal(m_tensor, provider.m.cpu()))
            self.assertTrue(torch.equal(latent, provider.sample_pre_frequency_latent(3)))

            w2 = torch.load(bundle_dir / "w2.pth", map_location="cpu", weights_only=False)
            self.assertTrue(w2.is_complex())
            self.assertEqual(tuple(w2.shape), (1, 4, 64, 64))

    def test_official_created_w1_w2_load_in_raven(self):
        with TemporaryDirectory() as tmp:
            reference = official.Gaussian_Shading_chacha(
                1, 8, 8, 1e-6, 1000000, key=FIXED_KEY, nonce=FIXED_NONCE
            )
            official.set_random_seed(0)
            reference.create_watermark_and_return_w_m()
            w1_path = Path(tmp) / "w1.pth"
            torch.save(
                {"w": reference.watermark, "m": reference.m,
                 "key": reference.key, "nonce": reference.nonce},
                w1_path,
            )
            args = official.OfficialArgs()
            w2_path = Path(tmp) / "w2.pth"
            torch.save(official.get_watermarking_pattern(args, CPU, shape=(1, 4, 64, 64)), w2_path)

            provider = GmProvider(
                **provider_kwargs(gm_w1_path=str(w1_path), gm_w2_path=str(w2_path))
            )
            self.assertEqual(provider.state_source, "w1_file")
            self.assertTrue(torch.equal(provider.watermark.cpu(), reference.watermark.to(torch.int64)))
            np.testing.assert_array_equal(provider.m_flat, reference.m)
            self.assertEqual(provider.key, FIXED_KEY)
            self.assertEqual(provider.nonce, FIXED_NONCE)

            # A bundle can be created from imported official artifacts.
            bundle_dir = Path(tmp) / "bundle"
            bundled = GmProvider(
                **provider_kwargs(
                    gm_w1_path=str(w1_path), gm_w2_path=str(w2_path),
                    gm_bundle_dir=str(bundle_dir), gm_create_bundle=True,
                )
            )
            self.assertEqual(bundled.state_source, "bundle_created")
            self.assertEqual(
                gm_bundle.sha256_file(w1_path), gm_bundle.sha256_file(bundle_dir / "w1.pth")
            )

    def test_non_official_w1_is_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "w1_legacy.pth"
            # legacy pre-ChaCha state: repeated watermark tensor, no key/nonce
            watermark = torch.randint(0, 2, (1, 4, 8, 8))
            torch.save({"w": watermark, "m": watermark.repeat(1, 1, 8, 8)}, path)
            with self.assertRaises(GmBundleError):
                gm_bundle.load_official_w1(path)


class ThresholdArtifactTests(unittest.TestCase):
    def _provider(self, bundle_dir):
        return make_bundle(bundle_dir)

    def test_threshold_round_trip_and_binding(self):
        with TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            provider = self._provider(bundle_dir)
            artifact = gm_bundle.build_threshold_artifact(
                threshold=0.83,
                binding=provider.binding_config(),
                score_definition=GM_SCORE_DEFINITION,
                threshold_source="cohort_calibration",
                report_label="calibrated_deployment_verification",
                target_fpr=0.01,
                empirical_fpr=0.008,
                tpr_at_target_fpr=0.97,
                roc_auc=0.999,
                positive_count=10,
                negative_count=10,
            )
            provider.bundle.save_threshold(artifact)

            reloaded = GmProvider(**provider_kwargs(gm_bundle_dir=str(bundle_dir)))
            info = reloaded.resolve_threshold()
            self.assertTrue(info["threshold_available"])
            self.assertEqual(info["threshold"], 0.83)
            self.assertEqual(info["comparison_operator"], ">=")
            self.assertEqual(info["report_label"], "calibrated_deployment_verification")

    def test_threshold_is_not_overwritten_silently(self):
        with TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            provider = self._provider(bundle_dir)
            artifact = gm_bundle.build_threshold_artifact(
                threshold=0.5, binding=provider.binding_config(),
                score_definition=GM_SCORE_DEFINITION,
                threshold_source="cohort_calibration",
                report_label="calibrated_deployment_verification",
            )
            provider.bundle.save_threshold(artifact)
            with self.assertRaises(GmBundleError):
                provider.bundle.save_threshold(artifact)
            provider.bundle.save_threshold(artifact, overwrite=True)

    def test_incompatible_threshold_is_rejected(self):
        with TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            provider = self._provider(bundle_dir)
            binding = provider.binding_config()
            binding["inversion_steps"] = 25  # calibrated with a different inversion
            artifact = gm_bundle.build_threshold_artifact(
                threshold=0.5, binding=binding,
                score_definition=GM_SCORE_DEFINITION,
                threshold_source="cohort_calibration",
                report_label="calibrated_deployment_verification",
            )
            provider.bundle.save_threshold(artifact)
            reloaded = GmProvider(**provider_kwargs(gm_bundle_dir=str(bundle_dir)))
            with self.assertRaises(GmBundleError):
                reloaded.resolve_threshold()

    def test_threshold_without_binding_block_is_rejected(self):
        artifact = {
            "schema": gm_bundle.GM_THRESHOLD_SCHEMA,
            "threshold": 0.5,
            "score_definition": GM_SCORE_DEFINITION,
            "score_direction": "higher_is_watermarked",
            "comparison_operator": ">=",
            "threshold_source": "imported",
            "report_label": "calibrated_deployment_verification",
        }
        with self.assertRaises(GmBundleError):
            gm_bundle.assert_threshold_compatible(artifact, {"inversion_steps": 50})

    def test_strict_greater_than_operator_is_rejected(self):
        artifact = {
            "schema": gm_bundle.GM_THRESHOLD_SCHEMA,
            "threshold": 0.5,
            "score_definition": GM_SCORE_DEFINITION,
            "score_direction": "higher_is_watermarked",
            "comparison_operator": ">",
            "threshold_source": "user_supplied",
            "report_label": "user_supplied_threshold",
        }
        with self.assertRaises(GmBundleError):
            gm_bundle.validate_threshold_artifact(artifact)

    def test_decision_uses_greater_or_equal(self):
        self.assertTrue(GmProvider.decide(0.5, 0.5))
        self.assertTrue(GmProvider.decide(0.6, 0.5))
        self.assertFalse(GmProvider.decide(0.4, 0.5))
        self.assertIsNone(GmProvider.decide(None, 0.5))
        self.assertIsNone(GmProvider.decide(0.5, None))

    def test_no_threshold_yields_raw_scores_without_a_decision(self):
        with TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            provider = self._provider(bundle_dir)
            info = provider.resolve_threshold()
            self.assertFalse(info["threshold_available"])
            self.assertIsNone(info["threshold"])
            self.assertIn(info["report_label"],
                          ("official_profile_raw_scores", "legacy_or_ablation_mode"))

    def test_user_supplied_threshold_is_labelled(self):
        with TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            make_bundle(bundle_dir)
            provider = GmProvider(
                **provider_kwargs(gm_bundle_dir=str(bundle_dir), gm_threshold=0.9)
            )
            info = provider.resolve_threshold()
            self.assertEqual(info["report_label"], "user_supplied_threshold")
            self.assertEqual(info["threshold_source"], "user_supplied")


class CanonicalHashingTests(unittest.TestCase):
    def test_key_order_does_not_change_the_hash(self):
        a = {"b": 1, "a": {"y": [1, 2], "x": True}}
        b = {"a": {"x": True, "y": [1, 2]}, "b": 1}
        self.assertEqual(gm_bundle.canonical_sha256(a), gm_bundle.canonical_sha256(b))

    def test_hash_survives_json_round_trip(self):
        payload = {"seed": 3, "ratio": 0.25, "flag": False, "path": Path("a/b.png"), "none": None}
        before = gm_bundle.canonical_sha256(payload)
        after = gm_bundle.canonical_sha256(json.loads(gm_bundle.canonical_json(payload)))
        self.assertEqual(before, after)

    def test_non_finite_values_are_rejected(self):
        with self.assertRaises(GmBundleError):
            gm_bundle.canonical_sha256({"x": float("nan")})
        with self.assertRaises(GmBundleError):
            gm_bundle.canonical_sha256({"x": float("inf")})

    def test_raw_bytes_never_enter_canonical_metadata(self):
        with self.assertRaises(GmBundleError):
            gm_bundle.canonical_sha256({"key": b"secret"})


if __name__ == "__main__":
    unittest.main()
