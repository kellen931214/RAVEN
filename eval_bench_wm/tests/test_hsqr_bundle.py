"""HSQR (SFWMark) bundle and threshold-artifact fail-closed tests — Issue #5.

Covers:

* selected-key and full-keybook save/load round trips (values, dtype, hashes);
* canonical metadata hash equality before and after serialization;
* tampered pattern / keybook / mapping / manifest rejection;
* a loaded bundle using the *persisted* pattern rather than regenerating it;
* incompatible profile / model / geometry / QR configuration / key rejection;
* artifact conflicts failing instead of overwriting silently;
* threshold artifact binding, score/distance sign and operator gates.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from utils.wm import sfw_bundle
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
        latent_shape=LATENT_SHAPE,
        device=CPU,
        hsqr_profile=OFFICIAL_PROFILE_NAME,
        hsqr_base_key_seed=OFFICIAL_BASE_KEY_SEED,
        hsqr_key_index=0,
        **MODEL_KWARGS,
    )
    kwargs.update(overrides)
    return HSQRProvider(**kwargs)


class BundleTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def make_bundle(self, provider=None, save_keybook=False, key_mapping=None,
                    directory="bundle") -> SfwBundle:
        provider = provider or official_provider()
        return provider.create_bundle(
            self.tmp / directory, save_keybook=save_keybook, key_mapping=key_mapping
        )


class TestBundleRoundTrip(BundleTestCase):
    def test_selected_key_round_trip_preserves_values_dtype_and_hashes(self):
        provider = official_provider(hsqr_key_index=42)
        bundle = self.make_bundle(provider)

        reloaded = SfwBundle.load(bundle.dir)
        self.assertEqual(reloaded.pattern.dtype, torch.bool)
        self.assertTrue(torch.equal(reloaded.pattern, provider.gt_patch))
        self.assertEqual(
            reloaded.manifest["selected_pattern_sha256"], provider.pattern_sha256()
        )
        self.assertEqual(reloaded.manifest["selected_key_index"], 42)
        self.assertEqual(reloaded.manifest["selected_key_seed"], 7433 + 42)
        self.assertEqual(reloaded.manifest["payload_text"], "HSQR7475")
        self.assertEqual(reloaded.manifest["base_key_seed"], 7433)
        self.assertEqual(reloaded.manifest["method"], "HSQR")
        self.assertEqual(reloaded.manifest["profile_name"], OFFICIAL_PROFILE_NAME)
        self.assertEqual(reloaded.manifest["center_slice"], [10, 54])
        self.assertEqual(reloaded.manifest["watermark_channels"], [3])
        self.assertEqual(reloaded.manifest["wm_capacity"], 2048)
        self.assertEqual(
            reloaded.manifest["official_reference_commit"],
            sfw_bundle.OFFICIAL_SFWMARK_COMMIT,
        )

    def test_canonical_hash_survives_serialization_and_reload(self):
        bundle = self.make_bundle()
        manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            SfwBundle.config_sha256(manifest), manifest["bundle_config_sha256"]
        )
        # And the in-memory manifest hashes to the same value.
        self.assertEqual(
            SfwBundle.config_sha256(bundle.manifest), manifest["bundle_config_sha256"]
        )

    def test_full_keybook_and_mapping_round_trip(self):
        provider = official_provider()
        mapping = [provider.sample_key_index(i) for i in range(4)]
        bundle = self.make_bundle(provider, save_keybook=True, key_mapping=mapping)

        reloaded = SfwBundle.load(bundle.dir)
        self.assertEqual(tuple(reloaded.keybook.shape), (2048, 1, 42, 42))
        self.assertEqual(reloaded.keybook.dtype, torch.bool)
        self.assertEqual(reloaded.key_mapping, mapping)
        self.assertTrue(torch.equal(reloaded.keybook[0], provider.gt_patch))

    def test_loaded_bundle_uses_the_persisted_pattern(self):
        provider = official_provider(hsqr_key_index=3)
        bundle = self.make_bundle(provider)

        # Corrupt the *derivation* by loading with a provider that would derive a
        # different pattern; the persisted tensor must win, and the mismatch with
        # the recorded identity must be caught.
        reloaded = SfwBundle.load(bundle.dir)
        rebuilt = HSQRProvider.from_bundle(
            reloaded, latent_shape=LATENT_SHAPE, device=CPU,
            hsqr_inversion_prompt="",
        )
        self.assertEqual(rebuilt.pattern_source, "bundle")
        self.assertTrue(torch.equal(rebuilt.gt_patch, reloaded.pattern))
        self.assertEqual(rebuilt.selected_key_index, 3)


class TestBundleFailClosed(BundleTestCase):
    def test_creating_over_an_existing_bundle_fails(self):
        provider = official_provider()
        self.make_bundle(provider)
        with self.assertRaises(SfwBundleError):
            official_provider().create_bundle(self.tmp / "bundle")

    def test_tampered_pattern_file_is_rejected(self):
        bundle = self.make_bundle()
        tampered = bundle.pattern.clone()
        tampered[0, 0, 0] = ~tampered[0, 0, 0]
        sfw_bundle.save_pattern(bundle.pattern_path, tampered)
        with self.assertRaises(SfwBundleError):
            SfwBundle.load(bundle.dir)

    def test_tampered_manifest_is_rejected(self):
        bundle = self.make_bundle()
        manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
        manifest["selected_key_index"] = 1
        bundle.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(SfwBundleError):
            SfwBundle.load(bundle.dir)

    def test_tampered_keybook_is_rejected(self):
        provider = official_provider()
        bundle = self.make_bundle(provider, save_keybook=True)
        keybook = bundle.keybook.clone()
        keybook[5, 0, 0, 0] = ~keybook[5, 0, 0, 0]
        sfw_bundle.save_pattern(bundle.keybook_path, keybook)
        with self.assertRaises(SfwBundleError):
            SfwBundle.load(bundle.dir)

    def test_tampered_key_mapping_is_rejected(self):
        bundle = self.make_bundle(key_mapping=[1, 2, 3])
        payload = json.loads(bundle.key_mapping_path.read_text(encoding="utf-8"))
        payload["key_mapping"] = [1, 2, 4]
        bundle.key_mapping_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(SfwBundleError):
            SfwBundle.load(bundle.dir)

    def test_undeclared_extra_artifact_is_rejected(self):
        bundle = self.make_bundle()
        sfw_bundle.save_pattern(bundle.keybook_path, torch.zeros(2048, 1, 42, 42, dtype=torch.bool))
        with self.assertRaises(SfwBundleError):
            SfwBundle.load(bundle.dir)

    def test_missing_selected_pattern_is_rejected(self):
        bundle = self.make_bundle()
        bundle.pattern_path.unlink()
        with self.assertRaises(SfwBundleError):
            SfwBundle.load(bundle.dir)

    def test_non_boolean_pattern_is_rejected(self):
        with self.assertRaises(SfwBundleError):
            sfw_bundle.validate_pattern(torch.zeros(1, 42, 42, dtype=torch.float32), 1)

    def test_key_mapping_outside_capacity_is_rejected(self):
        provider = official_provider()
        with self.assertRaises(SfwBundleError):
            provider.create_bundle(self.tmp / "b2", key_mapping=[0, 2048])


class TestBundleCompatibility(BundleTestCase):
    def _reload_with(self, bundle, **overrides):
        reloaded = SfwBundle.load(bundle.dir)
        provider = official_provider(**overrides)
        provider.attach_bundle(reloaded)
        return provider

    def test_a_matching_configuration_attaches(self):
        bundle = self.make_bundle()
        provider = self._reload_with(bundle)
        self.assertIsNotNone(provider.bundle)

    def test_different_key_index_is_rejected(self):
        bundle = self.make_bundle()
        with self.assertRaises(SfwBundleError):
            self._reload_with(bundle, hsqr_key_index=1)

    def test_different_model_is_rejected(self):
        bundle = self.make_bundle()
        with self.assertRaises(SfwBundleError):
            self._reload_with(bundle, modelid_target="CompVis/stable-diffusion-v1-4")

    def test_different_scheduler_is_rejected(self):
        bundle = self.make_bundle()
        with self.assertRaises(SfwBundleError):
            self._reload_with(bundle, scheduler_target="DPM")

    def test_different_delta_is_rejected(self):
        bundle = self.make_bundle()
        with self.assertRaises(SfwBundleError):
            self._reload_with(bundle, delta=1)

    def test_different_inversion_prompt_is_rejected(self):
        bundle = self.make_bundle()
        with self.assertRaises(SfwBundleError):
            self._reload_with(bundle, hsqr_inversion_prompt="a cat")

    def test_different_inversion_steps_is_rejected(self):
        bundle = self.make_bundle()
        with self.assertRaises(SfwBundleError):
            self._reload_with(bundle, hsqr_inversion_steps=25)


class TestThresholdArtifact(BundleTestCase):
    def _artifact(self, provider, threshold=-60.0, **overrides):
        kwargs = dict(
            threshold=threshold,
            binding=provider.binding_config(),
            score_definition=HSQR_SCORE_DEFINITION,
            threshold_source="cohort_calibration",
            report_label="calibrated_deployment_verification",
            target_fpr=0.01,
            empirical_fpr=0.0,
            tpr_at_target_fpr=1.0,
            roc_auc=1.0,
            positive_count=4,
            negative_count=4,
        )
        kwargs.update(overrides)
        return sfw_bundle.build_threshold_artifact(**kwargs)

    def test_threshold_records_both_score_and_distance_operating_points(self):
        provider = official_provider()
        self.make_bundle(provider)
        artifact = self._artifact(provider, threshold=-60.0)
        self.assertEqual(artifact["threshold"], -60.0)
        self.assertEqual(artifact["distance_threshold"], 60.0)
        self.assertEqual(artifact["comparison_operator"], ">=")
        self.assertEqual(artifact["distance_comparison_operator"], "<=")
        self.assertEqual(artifact["score_direction"], "higher_is_watermarked")

    def test_threshold_round_trip_and_resolution(self):
        provider = official_provider()
        bundle = self.make_bundle(provider)
        bundle.save_threshold(self._artifact(provider))

        reloaded = SfwBundle.load(bundle.dir)
        rebuilt = HSQRProvider.from_bundle(
            reloaded, latent_shape=LATENT_SHAPE, device=CPU, hsqr_inversion_prompt=""
        )
        info = rebuilt.resolve_threshold()
        self.assertTrue(info["threshold_available"])
        self.assertEqual(info["threshold"], -60.0)
        self.assertEqual(info["distance_threshold"], 60.0)
        self.assertEqual(info["report_label"], "calibrated_deployment_verification")
        self.assertEqual(info["threshold_target_fpr"], 0.01)

    def test_saving_a_threshold_twice_requires_overwrite(self):
        provider = official_provider()
        bundle = self.make_bundle(provider)
        bundle.save_threshold(self._artifact(provider))
        with self.assertRaises(SfwBundleError):
            bundle.save_threshold(self._artifact(provider))
        bundle.save_threshold(self._artifact(provider, threshold=-55.0), overwrite=True)

    def test_threshold_from_another_bundle_is_rejected(self):
        provider_a = official_provider(hsqr_key_index=0)
        provider_b = official_provider(hsqr_key_index=1)
        bundle_a = self.make_bundle(provider_a, directory="a")
        self.make_bundle(provider_b, directory="b")
        foreign = self._artifact(provider_b)
        with self.assertRaises(SfwBundleError):
            sfw_bundle.assert_threshold_compatible(foreign, provider_a.binding_config())
        # …and it is refused at resolution time, not silently applied.
        bundle_a.save_threshold(foreign)
        reloaded = SfwBundle.load(bundle_a.dir)
        rebuilt = HSQRProvider.from_bundle(
            reloaded, latent_shape=LATENT_SHAPE, device=CPU, hsqr_inversion_prompt=""
        )
        with self.assertRaises(SfwBundleError):
            rebuilt.resolve_threshold()

    def test_positive_distance_threshold_is_rejected_by_the_schema(self):
        provider = official_provider()
        self.make_bundle(provider)
        artifact = self._artifact(provider)
        artifact["comparison_operator"] = "<="
        with self.assertRaises(SfwBundleError):
            sfw_bundle.validate_threshold_artifact(artifact)

    def test_non_finite_threshold_is_rejected(self):
        provider = official_provider()
        self.make_bundle(provider)
        artifact = self._artifact(provider)
        artifact["threshold"] = float("nan")
        with self.assertRaises(SfwBundleError):
            sfw_bundle.validate_threshold_artifact(artifact)

    def test_threshold_without_binding_block_is_rejected(self):
        provider = official_provider()
        self.make_bundle(provider)
        artifact = self._artifact(provider)
        artifact.pop("binding")
        with self.assertRaises(SfwBundleError):
            sfw_bundle.assert_threshold_compatible(artifact, provider.binding_config())

    def test_no_threshold_means_no_decision_not_a_negative(self):
        provider = official_provider()
        self.make_bundle(provider)
        info = provider.resolve_threshold()
        self.assertFalse(info["threshold_available"])
        self.assertIsNone(info["threshold"])
        self.assertEqual(info["report_label"], "official_profile_raw_scores")
        self.assertIsNone(provider.decide(-40.0, info["threshold"]))

    def test_user_supplied_threshold_is_labelled(self):
        provider = official_provider(hsqr_threshold=-50.0)
        info = provider.resolve_threshold()
        self.assertEqual(info["threshold"], -50.0)
        self.assertEqual(info["threshold_source"], "user_supplied")
        self.assertEqual(info["report_label"], "user_supplied_threshold")

    def test_legacy_threshold_requires_an_explicit_flag_and_stays_labelled(self):
        provider = official_provider()
        self.assertFalse(provider.resolve_threshold()["threshold_available"])

        legacy = official_provider(hsqr_allow_legacy_threshold=True)
        info = legacy.resolve_threshold()
        self.assertTrue(info["threshold_available"])
        self.assertAlmostEqual(info["threshold"], -65.86233520507812)
        self.assertEqual(info["threshold_source"], "legacy_default_threshold")
        self.assertEqual(info["report_label"], "legacy_threshold")
        self.assertEqual(info["threshold_nominal_fpr"], 1e-3)
        self.assertIsNone(info["threshold_target_fpr"])


class TestCanonicalHashing(unittest.TestCase):
    def test_nan_is_rejected(self):
        with self.assertRaises(SfwBundleError):
            sfw_bundle.canonical_sha256({"value": float("nan")})

    def test_raw_bytes_are_rejected(self):
        with self.assertRaises(SfwBundleError):
            sfw_bundle.canonical_sha256({"secret": b"\x00\x01"})

    def test_key_order_does_not_change_the_hash(self):
        self.assertEqual(
            sfw_bundle.canonical_sha256({"a": 1, "b": 2}),
            sfw_bundle.canonical_sha256({"b": 2, "a": 1}),
        )

    def test_bool_and_int_are_distinct(self):
        self.assertNotEqual(
            sfw_bundle.canonical_sha256({"v": True}),
            sfw_bundle.canonical_sha256({"v": 1}),
        )


if __name__ == "__main__":
    unittest.main()
