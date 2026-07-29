"""HSQR (SFWMark) official-parity and regression tests — Issue #5.

Covers the CPU/static half of the required test matrix:

* official QR fixtures for key indices 0, 1, a middle key and 2047;
* exact payload strings and boolean pattern equality against the frozen spec;
* element-wise injection and detector parity against ``hsqr_official_reference``;
* the ``distance <-> score`` sign and the threshold comparison operator;
* batch scoring returning one result per item (the pre-fix code scored item 0
  only and silently reused it for every later image);
* explicit key index replacing the ``fix_gt`` overload in the official profile;
* deterministic per-sample seeds/keys independent of loop order;
* incompatible latent shape / profile / key-index rejection.
"""

from __future__ import annotations

import unittest

import torch

import hsqr_official_reference as ref
from utils.wm.hsqr_provider import (
    HSQR_COMPARISON_OPERATOR,
    HSQR_SCORE_DEFINITION,
    HSQR_SCORE_DIRECTION,
    LEGACY_PROFILE_NAME,
    OFFICIAL_BASE_KEY_SEED,
    OFFICIAL_PROFILE_NAME,
    HSQRProvider,
    apply_arg_defaults,
)
from utils.wm.sfw_bundle import sha256_tensor


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


class TestOfficialKeybook(unittest.TestCase):
    def test_frozen_key_fixtures(self):
        provider = official_provider()
        for key_index, fixture in ref.OFFICIAL_KEY_FIXTURES.items():
            with self.subTest(key_index=key_index):
                self.assertEqual(provider.key_seed(key_index), fixture["key_seed"])
                self.assertEqual(provider.payload_text(key_index), fixture["payload"])
                pattern = provider.make_pattern(key_index)
                self.assertEqual(sha256_tensor(pattern), fixture["pattern_sha256"])

    def test_patterns_match_official_reference_elementwise(self):
        provider = official_provider()
        for key_index in ref.OFFICIAL_KEY_FIXTURES:
            with self.subTest(key_index=key_index):
                expected = ref.official_pattern(key_index)
                actual = provider.make_pattern(key_index)
                self.assertEqual(actual.dtype, torch.bool)
                self.assertEqual(tuple(actual.shape), (1, 42, 42))
                self.assertTrue(torch.equal(actual, expected))

    def test_official_base_key_seed_is_7433(self):
        provider = official_provider()
        self.assertEqual(provider.base_key_seed, 7433)
        self.assertEqual(provider.wm_capacity, 2048)
        self.assertEqual(provider.key_seed(0), 7433)
        self.assertEqual(provider.key_seed(2047), 9480)

    def test_official_profile_rejects_non_official_base_seed(self):
        with self.assertRaises(ValueError):
            official_provider(hsqr_base_key_seed=999999)

    def test_keybook_is_the_full_capacity_and_matches_make_pattern(self):
        provider = official_provider()
        keybook = provider.keybook()
        self.assertEqual(tuple(keybook.shape), (2048, 1, 42, 42))
        self.assertEqual(keybook.dtype, torch.bool)
        for key_index in (0, 1, 1024, 2047):
            self.assertTrue(torch.equal(keybook[key_index], provider.make_pattern(key_index)))


class TestKeySelection(unittest.TestCase):
    def test_official_profile_uses_explicit_index_not_fix_gt(self):
        # fix_gt used to silently pick a random key through the global NumPy RNG.
        for fix_gt in (0, 1, 7):
            provider = official_provider(hsqr_key_index=3, fix_gt=fix_gt)
            self.assertEqual(provider.selected_key_index, 3)
            self.assertEqual(provider.key_selection, "explicit_index")

    def test_official_profile_forbids_legacy_fix_gt_selection(self):
        with self.assertRaises(ValueError):
            official_provider(hsqr_key_selection="legacy_fix_gt")

    def test_key_index_is_range_checked(self):
        for bad in (-1, 2048, 99999):
            with self.subTest(key_index=bad), self.assertRaises(ValueError):
                official_provider(hsqr_key_index=bad)

    def test_key_identity_is_independent_of_global_rng(self):
        import numpy as np

        np.random.seed(1)
        first = official_provider(hsqr_key_index=17).key_identity()
        np.random.seed(999)
        [np.random.random() for _ in range(100)]
        second = official_provider(hsqr_key_index=17).key_identity()
        self.assertEqual(first, second)

    def test_legacy_profile_still_reproduces_the_fix_gt_overload(self):
        provider = HSQRProvider(
            latent_shape=LATENT_SHAPE, device=CPU, hsqr_profile=LEGACY_PROFILE_NAME,
            hsqr_seed=999999, fix_gt=1,
        )
        self.assertEqual(provider.key_selection, "legacy_fix_gt")
        self.assertFalse(provider.profile_is_official)
        # Frozen legacy value: seed 999999, fix_gt=1 -> keybook index 1993.
        self.assertEqual(provider.selected_key_index, 1993)

    def test_per_sample_key_policy_is_deterministic_and_order_independent(self):
        provider = official_provider(hsqr_key_policy="per_sample")
        forward = [provider.sample_key_index(i) for i in range(16)]
        backward = [provider.sample_key_index(i) for i in reversed(range(16))][::-1]
        self.assertEqual(forward, backward)
        self.assertTrue(all(0 <= index < 2048 for index in forward))
        # A different provider instance derives exactly the same mapping.
        self.assertEqual(forward, [official_provider(hsqr_key_policy="per_sample")
                                   .sample_key_index(i) for i in range(16)])


class TestInjectionParity(unittest.TestCase):
    def test_injection_matches_official_reference_elementwise(self):
        provider = official_provider(hsqr_key_index=5)
        latent = torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(7))
        expected = ref.official_inject(latent.clone(), ref.official_pattern(5))
        actual = provider.inject(latent.clone())
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=0))

    def test_injection_only_touches_the_center_region_of_the_watermark_channel(self):
        provider = official_provider()
        latent = torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(3))
        injected = provider.inject(latent)
        outside = torch.ones(64, 64, dtype=torch.bool)
        outside[10:54, 10:54] = False
        self.assertTrue(torch.equal(injected[:, :, outside], latent[:, :, outside]))

    def test_paired_latents_share_the_pre_injection_base_latent(self):
        provider = official_provider()
        sample = provider.build_sample_latents(sample_seed=1234)
        self.assertEqual(
            sample["clean_base_latent_sha256"],
            sample["watermark_pre_injection_base_latent_sha256"],
        )
        self.assertNotEqual(
            sample["base_latent_sha256"], sample["watermarked_latent_sha256"]
        )

    def test_different_samples_do_not_share_a_complete_base_latent(self):
        provider = official_provider()
        hashes = {provider.build_sample_latents(100 + i)["base_latent_sha256"] for i in range(8)}
        self.assertEqual(len(hashes), 8)

    def test_base_latent_is_reproducible_from_the_seed_alone(self):
        first = official_provider().build_sample_latents(555)
        # A different provider, after unrelated RNG traffic, must reproduce it.
        torch.randn(1000)
        second = official_provider(hsqr_key_index=9).build_sample_latents(555)
        self.assertEqual(first["base_latent_sha256"], second["base_latent_sha256"])


class TestDetectorParity(unittest.TestCase):
    def test_detector_target_matches_official_reference(self):
        provider = official_provider(hsqr_key_index=11)
        expected = ref.official_target(ref.official_pattern(11))
        actual = provider.detector_target(provider.gt_patch)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(tuple(actual.shape), (42, 21))
        self.assertEqual(set(actual.real.unique().tolist()), {45.0, -45.0})

    def test_l1_distance_matches_official_reference(self):
        provider = official_provider(hsqr_key_index=2)
        latent = torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(21))
        watermarked = provider.inject(latent)
        expected = ref.official_l1_distance(watermarked, ref.official_pattern(2))
        actual = provider.l1_distances(watermarked)[0]
        self.assertAlmostEqual(actual, expected, places=3)

    def test_watermarked_scores_higher_than_clean(self):
        provider = official_provider()
        sample = provider.build_sample_latents(77)
        wm_distance = provider.l1_distances(sample["watermarked_latent"])[0]
        clean_distance = provider.l1_distances(sample["clean_latent"])[0]
        self.assertLess(wm_distance, clean_distance)
        self.assertGreater(
            provider.score_from_distance(wm_distance),
            provider.score_from_distance(clean_distance),
        )

    def test_score_is_the_negated_distance(self):
        provider = official_provider()
        sample = provider.build_sample_latents(5)
        record = provider.detect_from_latent(sample["watermarked_latent"])[0]
        self.assertEqual(record["hsqr_score"], -record["hsqr_l1_distance"])
        self.assertGreater(record["hsqr_l1_distance"], 0.0)
        self.assertEqual(record["score_definition"], HSQR_SCORE_DEFINITION)
        self.assertEqual(record["score_direction"], HSQR_SCORE_DIRECTION)

    def test_threshold_comparison_uses_score_not_distance(self):
        provider = official_provider()
        self.assertEqual(HSQR_COMPARISON_OPERATOR, ">=")
        self.assertTrue(provider.decide(score=-40.0, threshold=-65.86))
        self.assertFalse(provider.decide(score=-70.0, threshold=-65.86))
        # A missing threshold is undecided, never a negative detection.
        self.assertIsNone(provider.decide(score=-40.0, threshold=None))
        self.assertIsNone(provider.decide(score=None, threshold=-65.86))

    def test_is_detection_successful_consumes_a_raw_distance(self):
        provider = official_provider(hsqr_threshold=-65.86233520507812)
        self.assertTrue(provider.is_detection_successful(40.0))
        self.assertFalse(provider.is_detection_successful(70.0))


class TestBatchScoring(unittest.TestCase):
    def _batch_provider(self, batch: int) -> HSQRProvider:
        return official_provider(latent_shape=(batch, 4, 64, 64))

    def test_one_result_per_batch_item_in_input_order(self):
        provider = self._batch_provider(3)
        single = official_provider()
        latents = []
        expected = []
        for seed in (11, 22, 33):
            sample = single.build_sample_latents(seed)
            latents.append(sample["watermarked_latent"])
            expected.append(single.l1_distances(sample["watermarked_latent"])[0])
        batch = torch.cat(latents, dim=0)

        distances = provider.l1_distances(batch)
        self.assertEqual(len(distances), 3)
        for got, want in zip(distances, expected):
            self.assertAlmostEqual(got, want, places=4)

    def test_later_batch_elements_affect_their_own_output(self):
        """Regression: the pre-fix detector indexed the target at batch item 0."""
        provider = self._batch_provider(2)
        single = official_provider()
        watermarked = single.build_sample_latents(41)["watermarked_latent"]
        clean = single.build_sample_latents(42)["clean_latent"]

        first = provider.l1_distances(torch.cat([watermarked, clean], dim=0))
        second = provider.l1_distances(torch.cat([watermarked, clean * 2.0], dim=0))
        self.assertAlmostEqual(first[0], second[0], places=5)
        self.assertNotAlmostEqual(first[1], second[1], places=3)
        # And the watermarked item is clearly separated from the clean one.
        self.assertLess(first[0], first[1])

    def test_get_accuracies_returns_one_distance_per_item(self):
        provider = self._batch_provider(4)
        latents = torch.randn((4, 4, 64, 64), generator=torch.Generator().manual_seed(2))
        self.assertEqual(len(provider.get_accuracies(latents)["l1_dist"]), 4)

    def test_detect_from_latent_returns_one_record_per_item(self):
        provider = self._batch_provider(3)
        latents = torch.randn((3, 4, 64, 64), generator=torch.Generator().manual_seed(4))
        records = provider.detect_from_latent(latents)
        self.assertEqual(len(records), 3)
        self.assertEqual({record["selected_key_index"] for record in records}, {0})


class TestIdentification(unittest.TestCase):
    def test_identification_recovers_the_injected_key(self):
        provider = official_provider(hsqr_key_index=123)
        sample = provider.build_sample_latents(9)
        results = provider.identify(
            sample["watermarked_latent"], candidate_indices=[0, 5, 123, 500]
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["identified_key_index"], 123)
        self.assertEqual(results[0]["identified_payload_text"], "HSQR7556")
        self.assertEqual(results[0]["candidate_count"], 4)

    def test_identification_does_not_change_single_key_verification(self):
        provider = official_provider(hsqr_key_index=7)
        sample = provider.build_sample_latents(3)
        before = provider.detect_from_latent(sample["watermarked_latent"])[0]
        provider.identify(sample["watermarked_latent"], candidate_indices=[0, 1, 2])
        after = provider.detect_from_latent(sample["watermarked_latent"])[0]
        self.assertEqual(before, after)


class TestGeometryAndProfileGates(unittest.TestCase):
    def test_official_profile_rejects_incompatible_latent_shape(self):
        with self.assertRaises(ValueError):
            official_provider(latent_shape=(1, 4, 128, 128))

    def test_official_profile_rejects_a_non_official_center_slice(self):
        with self.assertRaises(ValueError):
            official_provider(hsqr_center_start=8, hsqr_center_end=56)

    def test_legacy_profile_warns_but_continues_on_other_geometry(self):
        with self.assertWarns(RuntimeWarning):
            provider = HSQRProvider(
                latent_shape=(1, 4, 128, 128), device=CPU,
                hsqr_profile=LEGACY_PROFILE_NAME, hsqr_seed=999999, fix_gt=0,
            )
        self.assertFalse(provider.profile_is_official)

    def test_capacity_is_fixed_at_2048(self):
        with self.assertRaises(ValueError):
            official_provider(hsqr_wm_capacity=1024)

    def test_payload_that_does_not_fit_version_1_is_rejected(self):
        from utils.wm.hsqr_provider import QRCodeGenerator

        generator = QRCodeGenerator(box_size=2, border=0, qr_version=1, error_correction="H")
        with self.assertRaises(ValueError):
            generator.make_qr_tensor("HSQR-this-payload-is-far-too-long-for-version-1")


class TestProfileApplication(unittest.TestCase):
    class Args:
        pass

    def _args(self, **values):
        args = self.Args()
        for key, value in values.items():
            setattr(args, key, value)
        return args

    def test_official_profile_sets_the_immutable_sd21_defaults(self):
        args = self._args(hsqr_profile=OFFICIAL_PROFILE_NAME)
        info = apply_arg_defaults(args, [])
        self.assertTrue(info["is_official"])
        self.assertEqual(args.modelid_target, "stabilityai/stable-diffusion-2-1-base")
        self.assertEqual(args.scheduler_target, "DDIM")
        self.assertEqual(args.hsqr_torch_dtype, "float32")
        self.assertEqual(args.resolution, 512)
        self.assertEqual(args.num_inference_steps_target, 50)
        self.assertEqual(args.guidance_scale_target, 7.5)
        self.assertEqual(args.hsqr_base_key_seed, 7433)
        self.assertEqual(args.delta, 0)
        self.assertEqual((args.hsqr_center_start, args.hsqr_center_end), (10, 54))
        self.assertEqual(args.hsqr_inversion_guidance, 0.0)

    def test_an_explicit_override_stops_the_run_being_official(self):
        args = self._args(hsqr_profile=OFFICIAL_PROFILE_NAME, scheduler_target="DPM")
        info = apply_arg_defaults(args, ["--scheduler_target", "DPM"])
        self.assertFalse(info["is_official"])
        self.assertIn("scheduler_target", info["overrides"])
        self.assertEqual(args.scheduler_target, "DPM")

    def test_legacy_profile_applies_nothing_and_is_never_official(self):
        args = self._args(hsqr_profile=LEGACY_PROFILE_NAME, modelid_target="sdxl")
        info = apply_arg_defaults(args, [])
        self.assertFalse(info["is_official"])
        self.assertEqual(info["applied"], {})
        self.assertEqual(args.modelid_target, "sdxl")


if __name__ == "__main__":
    unittest.main()
