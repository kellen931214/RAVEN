"""RingID official-parity tests (Issue #3), CPU-only and network-free.

Every check compares ``utils/wm/ringid_provider.py`` against the literal
transcription of https://github.com/showlab/RingID at commit
``45631a59aecd7d63ccdb640aaaf3e616fdb89fb9`` in ``tests/rid_official_reference.py``.
No model is loaded and nothing is downloaded.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rid_official_reference as official  # noqa: E402

from utils.wm import ringid_provider as rid  # noqa: E402
from utils.wm.rid_bundle import RidBundleError  # noqa: E402


DEVICE = torch.device("cpu")
LATENT_SHAPE = (1, 4, 64, 64)

# The frozen fixture: one profile, one RNG seed, one candidate ordering, one key
# index and one clean latent, all network-free.
FIXTURE_SEED = 42
FIXTURE_KEY_INDEX = 628
FIXTURE_MAX_INDEX = 640  # enough to cover key 628 without building all 2048


def make_provider(**overrides):
    kwargs = dict(latent_shape=LATENT_SHAPE, device=DEVICE)
    kwargs.update(overrides)
    return rid.RingIDProvider(**kwargs)


class MaskParityTests(unittest.TestCase):
    def test_rounder_ring_mask_matches_official_element_by_element(self):
        for r_out, r_in in ((14, 3), (14, 13), (10, 9), (4, 3)):
            with self.subTest(r_out=r_out, r_in=r_in):
                np.testing.assert_array_equal(
                    rid.ring_mask(size=64, r_out=r_out, r_in=r_in),
                    official.ring_mask(size=64, r_out=r_out, r_in=r_in),
                )

    def test_effective_mask_is_the_rounder_ring_not_the_plain_circle(self):
        """Regression: the provider previously used the plain circle-difference mask.

        ``USE_ROUNDER_RING = True`` rebinds the official ``ring_mask``; the two
        masks select a different number of coefficients, so the old provider
        scored a different frequency region than official RingID.
        """
        rounder = rid.ring_mask(size=64, r_out=14, r_in=3)
        plain = rid.plain_ring_mask(size=64, r_out=14, r_in=3)
        self.assertTrue(rid.USE_ROUNDER_RING)
        self.assertNotEqual(int(rounder.sum()), int(plain.sum()))
        np.testing.assert_array_equal(rounder, official.ring_mask(size=64, r_out=14, r_in=3))

    def test_circle_mask_and_channel_masks_match_official(self):
        np.testing.assert_array_equal(rid.circle_mask(size=64, r=14), official.circle_mask(size=64, r=14))
        provider = make_provider()
        region, heter = official.official_watermark_region_mask(size=64, device="cpu")
        self.assertTrue(torch.equal(provider.watermarking_mask, region))
        self.assertTrue(torch.equal(provider.heter_watermark_region_mask, heter))
        self.assertEqual(rid.WATERMARK_CHANNEL, [0, 3])
        self.assertEqual(rid.RADIUS, 14)
        self.assertEqual(rid.RADIUS_CUTOFF, 3)


class KeybookParityTests(unittest.TestCase):
    def test_candidate_ordering_matches_official_itertools_product(self):
        provider = make_provider()
        self.assertEqual(provider.key_value_combinations(), official.official_key_value_combinations())

    def test_default_capacity_is_two_to_the_eleven(self):
        provider = make_provider()
        self.assertEqual(provider.candidate_count, 2 ** (rid.RADIUS - rid.RADIUS_CUTOFF))
        self.assertEqual(provider.candidate_count, 2048)
        self.assertEqual(len(provider.key_value_combinations()), 2048)

    def test_quantized_values_are_the_official_two_level_profile(self):
        self.assertEqual(make_provider().quantization_values, [-64.0, 64.0])

    def test_key_628_matches_official_reference_element_by_element(self):
        provider = make_provider(rid_key_index=FIXTURE_KEY_INDEX, rid_key_seed=FIXTURE_SEED)
        expected = official.official_keybook(
            general_seed=FIXTURE_SEED, latent_shape=LATENT_SHAPE, device="cpu",
            max_index=FIXTURE_MAX_INDEX,
        )[FIXTURE_KEY_INDEX]
        self.assertTrue(torch.equal(provider.gt_patch, expected))

    def test_heterogeneous_channel_matches_under_the_frozen_rng_fixture(self):
        """Channel 0 is a per-candidate Gaussian draw; the RNG stream must match."""
        provider = make_provider(rid_key_seed=FIXTURE_SEED)
        expected = official.official_keybook(
            general_seed=FIXTURE_SEED, latent_shape=LATENT_SHAPE, device="cpu",
            max_index=FIXTURE_MAX_INDEX,
        )
        for index in (0, 1, 5, FIXTURE_KEY_INDEX, FIXTURE_MAX_INDEX - 1):
            with self.subTest(index=index):
                actual = provider.build_key_pattern(index)
                self.assertTrue(torch.equal(actual[:, 0], expected[index][:, 0]))
                self.assertTrue(torch.equal(actual[:, 3], expected[index][:, 3]))

    def test_fix_gt_and_spatial_shift_match_the_released_code(self):
        provider = make_provider(rid_key_seed=FIXTURE_SEED)
        raw = official.official_keybook(
            general_seed=FIXTURE_SEED, latent_shape=LATENT_SHAPE, device="cpu",
            max_index=8, fix_gt=0, time_shift=0,
        )
        fixed = official.official_keybook(
            general_seed=FIXTURE_SEED, latent_shape=LATENT_SHAPE, device="cpu",
            max_index=8, fix_gt=1, time_shift=0,
        )
        shifted = official.official_keybook(
            general_seed=FIXTURE_SEED, latent_shape=LATENT_SHAPE, device="cpu",
            max_index=8, fix_gt=1, time_shift=1,
        )
        self.assertFalse(torch.equal(raw[3], fixed[3]))
        self.assertFalse(torch.equal(fixed[3], shifted[3]))

        no_fix = make_provider(rid_key_seed=FIXTURE_SEED, rid_profile="legacy",
                               fix_gt=0, time_shift=0)
        self.assertTrue(torch.equal(no_fix.build_key_pattern(3), raw[3]))
        fix_only = make_provider(rid_key_seed=FIXTURE_SEED, rid_profile="legacy",
                                 fix_gt=1, time_shift=0)
        self.assertTrue(torch.equal(fix_only.build_key_pattern(3), fixed[3]))
        self.assertTrue(torch.equal(make_provider(rid_key_seed=FIXTURE_SEED).build_key_pattern(3),
                                    shifted[3]))

    def test_candidate_order_hash_is_stable_and_independent_of_the_key(self):
        a = make_provider(rid_key_index=0).candidate_order_sha256()
        b = make_provider(rid_key_index=628).candidate_order_sha256()
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)


class InjectionParityTests(unittest.TestCase):
    def test_injected_fourier_tensor_and_real_latent_match_official(self):
        provider = make_provider(rid_key_seed=FIXTURE_SEED)
        pattern = official.official_keybook(
            general_seed=FIXTURE_SEED, latent_shape=LATENT_SHAPE, device="cpu",
            max_index=FIXTURE_MAX_INDEX,
        )[FIXTURE_KEY_INDEX]

        clean = provider.sample_clean_latent(1234).to(torch.float64)
        region, _ = official.official_watermark_region_mask(size=64, device="cpu")
        expected = official.generate_Fourier_watermark_latents(
            device="cpu", radius=rid.RADIUS, radius_cutoff=rid.RADIUS_CUTOFF,
            original_latents=clean.clone(), watermark_pattern=pattern,
            watermark_channel=rid.WATERMARK_CHANNEL, watermark_region_mask=region,
        )
        actual = provider.inject(clean.clone())
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(actual.is_complex())


class DistanceParityTests(unittest.TestCase):
    def setUp(self):
        self.provider = make_provider(rid_key_seed=FIXTURE_SEED)
        self.pattern = self.provider.gt_patch
        self.region, _ = official.official_watermark_region_mask(size=64, device="cpu")
        generator = torch.Generator().manual_seed(11)
        self.recovered = torch.randn(*LATENT_SHAPE, generator=generator, dtype=torch.float64)

    def test_per_channel_and_channel_min_match_official_get_distance(self):
        record = self.provider.channel_distances(self.recovered)[0]
        expected_min = official.get_distance(
            self.pattern, official.fft(self.recovered), self.region,
            p=1, mode="complex", channel_min=True, channel=rid.WATERMARK_CHANNEL,
        )
        self.assertAlmostEqual(record["rid_channel_min_l1"], expected_min, places=12)
        self.assertAlmostEqual(record["rid_score"], -expected_min, places=12)
        self.assertEqual(record["score_direction"], "higher_is_watermarked")
        self.assertEqual(record["score_definition"], rid.RID_SCORE_DEFINITION)

        per_channel = rid.official_channel_distances(
            self.pattern, official.fft(self.recovered), self.region
        )
        self.assertAlmostEqual(record["rid_channel_0_l1"], per_channel[0], places=12)
        self.assertAlmostEqual(record["rid_channel_3_l1"], per_channel[1], places=12)
        self.assertAlmostEqual(record["rid_channel_min_l1"], min(per_channel), places=12)

    def test_official_mode_never_averages_the_two_channels(self):
        """Regression for the previous ``__get_l1_distance`` single-mean over [0, 3]."""
        record = self.provider.channel_distances(self.recovered)[0]
        averaged = official.get_distance(
            self.pattern, official.fft(self.recovered), self.region,
            p=1, mode="complex", channel_min=False, channel=rid.WATERMARK_CHANNEL,
        )
        self.assertNotAlmostEqual(record["rid_channel_min_l1"], averaged, places=6)
        self.assertLess(record["rid_channel_min_l1"], averaged)

        legacy = make_provider(rid_key_seed=FIXTURE_SEED, rid_profile="legacy", channel_min=0)
        self.assertAlmostEqual(
            legacy.channel_distances(self.recovered)[0]["rid_channel_min_l1"], averaged, places=12
        )

    def test_one_score_per_batch_item(self):
        batch = torch.cat([self.recovered, self.recovered * 2, self.recovered * 3], dim=0)
        records = self.provider.channel_distances(batch)
        self.assertEqual(len(records), 3)
        self.assertEqual(len({record["rid_channel_min_l1"] for record in records}), 3)
        singles = [self.provider.channel_distances(batch[i][None, ...])[0]["rid_channel_min_l1"]
                   for i in range(3)]
        for record, single in zip(records, singles):
            self.assertAlmostEqual(record["rid_channel_min_l1"], single, places=12)

    def test_watermarked_latent_scores_far_better_than_a_clean_one(self):
        sample = self.provider.build_sample_latents(7)
        wm = self.provider.channel_distances(sample["watermarked_latent"])[0]
        clean = self.provider.channel_distances(sample["clean_latent"].to(torch.float64))[0]
        self.assertLess(wm["rid_channel_min_l1"], clean["rid_channel_min_l1"])
        self.assertGreater(wm["rid_score"], clean["rid_score"])


class IdentificationTests(unittest.TestCase):
    INDICES = [0, 1, 7, 628, 629, 1000, 2047]

    def setUp(self):
        self.provider = make_provider(rid_key_seed=FIXTURE_SEED, rid_top_k=3)

    def test_argmin_and_top_k_match_the_official_traversal(self):
        sample = self.provider.build_sample_latents(21)
        result = self.provider.identify_key(
            sample["watermarked_latent"], candidate_indices=self.INDICES, true_key_index=628
        )[0]

        # Official identify.py: distance per candidate, then np.argmin.
        region, _ = official.official_watermark_region_mask(size=64, device="cpu")
        recovered_fft = official.fft(sample["watermarked_latent"])
        distances = []
        for index in sorted(self.INDICES):
            pattern = self.provider.build_key_pattern(index)
            distances.append(official.get_distance(
                pattern, recovered_fft, region, p=1, mode="complex",
                channel_min=True, channel=rid.WATERMARK_CHANNEL,
            ))
        expected_best = sorted(self.INDICES)[int(np.argmin(np.array(distances)))]

        self.assertEqual(result["predicted_key_index"], expected_best)
        self.assertEqual(result["predicted_key_index"], 628)
        self.assertEqual(result["predicted_key_id"], "rid-key-000628")
        self.assertTrue(result["identification_correct"])
        self.assertEqual(result["candidate_count"], len(self.INDICES))
        self.assertEqual(len(result["top_k_key_indices"]), 3)
        order = [i for _, i in sorted(zip(distances, sorted(self.INDICES)))]
        self.assertEqual(result["top_k_key_indices"], order[:3])
        self.assertAlmostEqual(result["best_distance"], min(distances), places=12)
        self.assertAlmostEqual(
            result["identification_margin"], sorted(distances)[1] - min(distances), places=12
        )

    def test_identification_needs_no_true_key(self):
        sample = make_provider(rid_key_seed=FIXTURE_SEED, rid_key_index=1000).build_sample_latents(3)
        result = self.provider.identify_key(
            sample["watermarked_latent"], candidate_indices=self.INDICES
        )[0]
        self.assertEqual(result["predicted_key_index"], 1000)
        self.assertIsNone(result["true_key_index"])
        self.assertIsNone(result["identification_correct"])

    def test_one_identification_record_per_batch_item(self):
        a = self.provider.build_sample_latents(5)["watermarked_latent"]
        b = make_provider(rid_key_seed=FIXTURE_SEED, rid_key_index=2047).build_sample_latents(6)["watermarked_latent"]
        results = self.provider.identify_key(
            torch.cat([a, b], dim=0), candidate_indices=self.INDICES
        )
        self.assertEqual([r["predicted_key_index"] for r in results], [628, 2047])

    def test_ties_resolve_deterministically_to_the_lowest_candidate_index(self):
        """Documented tie rule: stable sort, i.e. upstream ``np.argmin`` semantics."""
        info = self.provider.build_keybook(self.INDICES)
        tied = dict(info)
        # Force an exact tie by making every candidate identical.
        tied["masked"] = info["masked"][0][None, ...].repeat(len(self.INDICES), 1, 1)
        sample = self.provider.build_sample_latents(9)
        result = self.provider.identify_key(sample["watermarked_latent"], keybook=tied)[0]
        self.assertEqual(result["predicted_key_index"], min(self.INDICES))
        self.assertAlmostEqual(result["identification_margin"], 0.0, places=12)

    def test_declared_keybook_hash_covers_the_declared_candidates_only(self):
        full_subset = self.provider.build_keybook(self.INDICES)["keybook_sha256"]
        other = make_provider(rid_key_seed=FIXTURE_SEED).build_keybook([0, 1, 7])["keybook_sha256"]
        self.assertNotEqual(full_subset, other)


class ProfileAndFlagTests(unittest.TestCase):
    def _parse(self, argv):
        import run_verify_watermark

        args = run_verify_watermark.build_rid_parser().parse_args(argv)
        info = rid.apply_arg_defaults(args, argv)
        return args, info

    def test_official_profile_pins_the_sd21_configuration(self):
        args, info = self._parse(["--rid_profile", "official_sd21"])
        self.assertTrue(info["is_official"])
        self.assertEqual(args.modelid_target, "stabilityai/stable-diffusion-2-1-base")
        self.assertEqual(args.model_revision, "fp16")
        self.assertEqual(args.rid_torch_dtype, "float16")
        self.assertEqual(args.scheduler_target, "DPM")
        self.assertEqual(args.resolution, 512)
        self.assertEqual(args.num_inference_steps_target, 50)
        self.assertEqual(args.guidance_scale_target, 7.5)
        self.assertEqual(args.rid_inversion_steps, 50)
        self.assertEqual(args.rid_inversion_prompt, "")
        self.assertEqual(args.rid_inversion_guidance, 1.0)
        self.assertFalse(args.rid_vae_sample)
        self.assertEqual(args.rid_vae_scaling_factor, 0.18215)
        self.assertEqual(args.rid_shift_semantics, "official_code_exact")

    def test_explicit_override_downgrades_the_run_to_an_ablation(self):
        args, info = self._parse(["--rid_profile", "official_sd21", "--resolution", "768"])
        self.assertFalse(info["is_official"])
        self.assertIn("resolution", info["overrides"])
        self.assertEqual(args.resolution, 768)

    def test_ring_width_other_than_one_fails_in_official_mode(self):
        with self.assertRaises(RidBundleError):
            make_provider(ring_width=2)
        with self.assertRaises(NotImplementedError):
            rid.make_Fourier_ringid_pattern(
                "cpu", [[0.0]] * 11, torch.zeros(*LATENT_SHAPE, dtype=torch.float64),
                radius=14, radius_cutoff=3, ring_watermark_channel=[3],
                heter_watermark_channel=[], ring_width=2,
            )

    def test_time_shift_factor_cannot_be_presented_as_effective_in_official_mode(self):
        with self.assertRaises(RidBundleError):
            make_provider(time_shift_factor=0.85)
        provider = make_provider(rid_shift_semantics="paper_described_shift",
                                 time_shift_factor=0.85, rid_key_seed=FIXTURE_SEED)
        self.assertEqual(provider.shift_semantics, "paper_described_shift")
        official_pattern = make_provider(rid_key_seed=FIXTURE_SEED).gt_patch
        self.assertFalse(torch.equal(provider.gt_patch, official_pattern))
        self.assertIn("watermark_seed", rid.OFFICIAL_UNUSED_ARGS)

    def test_paper_shift_profile_is_a_separate_labelled_profile(self):
        self.assertEqual(
            rid.RID_PROFILES["paper_shift_ablation"]["rid_shift_semantics"],
            "paper_described_shift",
        )
        args, info = self._parse(["--rid_profile", "paper_shift_ablation"])
        self.assertFalse(info["is_official"])

    def test_legacy_rid_seed_is_never_remapped_to_an_official_key(self):
        with self.assertRaises(RidBundleError):
            make_provider(rid_seed=999999)
        legacy = make_provider(rid_profile="legacy", rid_seed=999999)
        self.assertFalse(legacy.profile_is_official)
        self.assertEqual(legacy.key_seed, 999999)
        self.assertFalse(torch.equal(legacy.gt_patch, make_provider().gt_patch))


class PerSampleLatentTests(unittest.TestCase):
    def test_samples_do_not_share_a_complete_initial_latent(self):
        provider = make_provider(rid_key_seed=FIXTURE_SEED)
        samples = [provider.build_sample_latents(seed) for seed in range(4)]
        clean_hashes = {sample["clean_latent_sha256"] for sample in samples}
        wm_hashes = {sample["post_injection_latent_sha256"] for sample in samples}
        self.assertEqual(len(clean_hashes), 4)
        self.assertEqual(len(wm_hashes), 4)
        self.assertEqual(len({sample["selected_pattern_sha256"] for sample in samples}), 1)

    def test_clean_and_pre_injection_latents_are_the_same_tensor(self):
        sample = make_provider().build_sample_latents(3)
        self.assertEqual(sample["clean_latent_sha256"], sample["pre_injection_latent_sha256"])

    def test_same_seed_reproduces_the_same_latent_across_providers(self):
        a = make_provider(rid_key_seed=FIXTURE_SEED).build_sample_latents(17)
        b = make_provider(rid_key_seed=FIXTURE_SEED).build_sample_latents(17)
        self.assertEqual(a["clean_latent_sha256"], b["clean_latent_sha256"])
        self.assertEqual(a["post_injection_latent_sha256"], b["post_injection_latent_sha256"])

    def test_sample_latents_do_not_depend_on_iteration_order(self):
        provider = make_provider(rid_key_seed=FIXTURE_SEED)
        forward = [provider.build_sample_latents(s)["clean_latent_sha256"] for s in (0, 1, 2)]
        backward = [provider.build_sample_latents(s)["clean_latent_sha256"] for s in (2, 1, 0)]
        self.assertEqual(forward, list(reversed(backward)))


class ScoreDirectionTests(unittest.TestCase):
    def test_decision_rule_is_score_ge_threshold(self):
        provider = make_provider()
        self.assertTrue(provider.decide(-10.0, -20.0))
        self.assertTrue(provider.decide(-20.0, -20.0))
        self.assertFalse(provider.decide(-30.0, -20.0))
        self.assertIsNone(provider.decide(None, -20.0))
        self.assertIsNone(provider.decide(-10.0, None))

    def test_no_threshold_means_raw_scores_only(self):
        info = make_provider().resolve_threshold()
        self.assertFalse(info["threshold_available"])
        self.assertIsNone(info["threshold"])
        self.assertEqual(info["comparison_operator"], ">=")
        self.assertEqual(info["score_direction"], "higher_is_watermarked")


if __name__ == "__main__":
    unittest.main()
