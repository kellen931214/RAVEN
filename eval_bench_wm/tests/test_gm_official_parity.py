"""Official GaussMarker parity fixture.

Every test compares the RAVEN provider (``utils/wm/gm_provider.py``) against the
literal official reference transcribed in ``tests/gm_official_reference.py``
from https://github.com/SunnierLee/GaussMarker commit
``4ac9bfd4e152a56bd93c2a06a809ef6ff8e73155``.

The fixtures are fixed deterministic tensors and artifacts. No test downloads
anything or requires a GPU.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

import gm_official_reference as official
from utils.wm import gm_bundle
from utils.wm.ddim_inversion import official_forward_diffusion
from utils.wm.gm_bundle import OFFICIAL_GAUSSMARKER_COMMIT
from utils.wm.gm_provider import GmProvider
from utils.wm.gm_unet import GmUNet


CPU = torch.device("cpu")
FIXED_KEY = bytes(range(32))
FIXED_NONCE = bytes(range(12))
BITS_SEED = 20260728


def build_provider(tmpdir: str, **overrides) -> GmProvider:
    """Provider with a fixed official w1/w2 on CPU and no detector artifacts."""
    state = GmProvider.create_official_state(
        channel_copy=1, w_copy=8, h_copy=8, bits_seed=BITS_SEED, key=FIXED_KEY, nonce=FIXED_NONCE
    )
    w1_path = Path(tmpdir) / "w1_fixture.pth"
    gm_bundle.save_official_w1(w1_path, state)
    kwargs = dict(
        latent_shape=(1, 4, 64, 64),
        device=CPU,
        gm_torch_dtype="float32",
        gm_w1_path=str(w1_path),
        gm_use_gnr=False,
        gm_use_classifier=False,
        modelid_target="stabilityai/stable-diffusion-2-1-base",
        scheduler_target="DPM",
    )
    kwargs.update(overrides)
    return GmProvider(**kwargs)


def official_watermark(provider: GmProvider) -> official.Gaussian_Shading_chacha:
    return official.Gaussian_Shading_chacha(
        provider.ch, provider.w, provider.h, provider.fpr, provider.user_number,
        watermark=provider.watermark.cpu(), m=provider.m_flat, key=provider.key, nonce=provider.nonce,
    )


class OfficialReferenceIdentityTests(unittest.TestCase):
    def test_frozen_official_commit(self):
        self.assertEqual(OFFICIAL_GAUSSMARKER_COMMIT, official.OFFICIAL_COMMIT)
        self.assertEqual(
            OFFICIAL_GAUSSMARKER_COMMIT, "4ac9bfd4e152a56bd93c2a06a809ef6ff8e73155"
        )


class ChaChaStateTests(unittest.TestCase):
    def test_chacha_encrypt_decrypt_round_trip(self):
        with TemporaryDirectory() as tmp:
            provider = build_provider(tmp)
            bits = np.random.RandomState(0).randint(0, 2, 4 * 64 * 64).astype(np.uint8)
            encrypted = provider.stream_key_encrypt(bits)
            decrypted = provider.stream_key_decrypt(encrypted).flatten().numpy()
            np.testing.assert_array_equal(bits, decrypted)

    def test_fixed_key_nonce_message_matches_official(self):
        """watermark bits -> repeat -> packbits -> ChaCha20 -> unpackbits."""
        state = GmProvider.create_official_state(
            channel_copy=1, w_copy=8, h_copy=8, bits_seed=BITS_SEED,
            key=FIXED_KEY, nonce=FIXED_NONCE,
        )
        reference = official.Gaussian_Shading_chacha(
            1, 8, 8, 1e-6, 1000000, key=FIXED_KEY, nonce=FIXED_NONCE
        )
        sd = state["w"].repeat(1, 1, 8, 8)
        reference_m = reference.stream_key_encrypt(sd.flatten().numpy())
        np.testing.assert_array_equal(state["m"], reference_m)

    def test_repeat_copy_and_shape_parity(self):
        with TemporaryDirectory() as tmp:
            provider = build_provider(tmp)
            self.assertEqual(tuple(provider.watermark.shape), (1, 4, 8, 8))
            self.assertEqual(provider.m_flat.shape, (4 * 64 * 64,))
            self.assertEqual(tuple(provider.m.shape), (1, 4, 64, 64))
            self.assertEqual(provider.marklength, 256)
            sd = provider.watermark.cpu().repeat(1, provider.ch, provider.w, provider.h)
            self.assertEqual(tuple(sd.shape), (1, 4, 64, 64))
            # decrypting the stored message must return the repeated identity bits
            decrypted = provider.stream_key_decrypt(provider.m_flat)
            torch.testing.assert_close(decrypted.to(torch.int64), sd.to(torch.int64))


class LatentSamplingTests(unittest.TestCase):
    def test_pre_frequency_latent_matches_official_element_by_element(self):
        with TemporaryDirectory() as tmp:
            provider = build_provider(tmp)
            reference = official_watermark(provider)

            for sample_seed in (0, 1, 5):
                official.set_random_seed(sample_seed)  # official: np.random.seed(seed + 3)
                expected = reference.truncSampling(provider.m_flat)
                actual = provider.sample_pre_frequency_latent(sample_seed)
                self.assertEqual(actual.dtype, torch.float16)
                self.assertEqual(tuple(actual.shape), (1, 4, 64, 64))
                self.assertTrue(
                    torch.equal(expected, actual),
                    f"pre-frequency latent diverged from official for seed {sample_seed}",
                )

    def test_local_rng_does_not_mutate_global_numpy_state(self):
        with TemporaryDirectory() as tmp:
            provider = build_provider(tmp)
            np.random.seed(12345)
            before = np.random.random()
            np.random.seed(12345)
            provider.sample_pre_frequency_latent(3)
            after = np.random.random()
            self.assertEqual(before, after)

    def test_same_seed_reproduces_and_different_seeds_diverge(self):
        with TemporaryDirectory() as tmp:
            provider = build_provider(tmp)
            first = provider.build_sample_latents(11)
            again = provider.build_sample_latents(11)
            other = provider.build_sample_latents(12)

            self.assertEqual(
                first["pre_injection_latent_sha256"], again["pre_injection_latent_sha256"]
            )
            self.assertEqual(
                first["post_injection_latent_sha256"], again["post_injection_latent_sha256"]
            )
            self.assertNotEqual(
                first["pre_injection_latent_sha256"], other["pre_injection_latent_sha256"]
            )
            self.assertNotEqual(
                first["post_injection_latent_sha256"], other["post_injection_latent_sha256"]
            )

    def test_watermark_identity_is_constant_across_samples(self):
        with TemporaryDirectory() as tmp:
            provider = build_provider(tmp)
            identity = gm_bundle.sha256_tensor(provider.watermark)
            message = gm_bundle.sha256_array(provider.m_flat)
            for seed in range(3):
                provider.build_sample_latents(seed)
            self.assertEqual(identity, gm_bundle.sha256_tensor(provider.watermark))
            self.assertEqual(message, gm_bundle.sha256_array(provider.m_flat))


class RingParityTests(unittest.TestCase):
    def test_mask_and_pattern_match_official(self):
        with TemporaryDirectory() as tmp:
            provider = build_provider(tmp)
            args = official.OfficialArgs()
            expected_patch = official.get_watermarking_pattern(args, CPU, shape=(1, 4, 64, 64))
            actual_patch = provider.build_watermarking_pattern()
            self.assertTrue(torch.equal(expected_patch, actual_patch))

            expected_mask = official.get_watermarking_mask(expected_patch.real, args, CPU)
            self.assertTrue(torch.equal(expected_mask, provider.watermarking_mask))

    def test_injection_matches_official(self):
        with TemporaryDirectory() as tmp:
            provider = build_provider(tmp)
            args = official.OfficialArgs()
            latent = provider.sample_pre_frequency_latent(4)

            expected = official.inject_watermark(
                latent.float().clone(), provider.watermarking_mask, provider.gt_patch, args
            )
            actual = provider.inject_ring(latent)
            self.assertTrue(torch.equal(expected, actual))

    def test_ring_l1_and_classifier_feature_sign_and_scale(self):
        with TemporaryDirectory() as tmp:
            provider = build_provider(tmp)
            args = official.OfficialArgs()
            recovered = torch.randn(1, 4, 64, 64, generator=torch.Generator().manual_seed(3))

            official_metric = official.eval_watermark(
                recovered, provider.watermarking_mask, provider.gt_patch, args
            )
            raw_l1 = provider.ring_l1(recovered)
            feature = provider.ring_classifier_feature(raw_l1)

            # official eval_watermark returns l1 * 0.01, the detector appends -metric
            self.assertAlmostEqual(raw_l1 * 0.01, official_metric, places=6)
            self.assertAlmostEqual(feature, -official_metric, places=9)
            self.assertLess(feature, 0.0)

    def test_ring_feature_uses_the_continuous_latent_not_the_sign_map(self):
        with TemporaryDirectory() as tmp:
            provider = build_provider(tmp)
            recovered = torch.randn(1, 4, 64, 64, generator=torch.Generator().manual_seed(5))
            sign_map = (recovered > 0).float()
            self.assertNotAlmostEqual(
                provider.ring_l1(recovered), provider.ring_l1(sign_map), places=3
            )


class BitRecoveryTests(unittest.TestCase):
    def test_voting_and_recovery_match_official(self):
        with TemporaryDirectory() as tmp:
            provider = build_provider(tmp)
            reference = official_watermark(provider)

            latent = provider.sample_pre_frequency_latent(2).float()
            expected = reference.pred_w_from_latent(latent)
            raw_m = (latent > 0).to(torch.int64)
            actual = provider.pred_w_from_m(raw_m)
            self.assertTrue(torch.equal(expected.to(torch.int64), actual.to(torch.int64)))

    def test_clean_pre_frequency_latent_recovers_the_identity(self):
        with TemporaryDirectory() as tmp:
            provider = build_provider(tmp)
            latent = provider.sample_pre_frequency_latent(9).float()
            raw_m = (latent > 0).to(torch.int64)
            recovered = provider.pred_w_from_m(raw_m)
            self.assertEqual(provider.bit_accuracy(recovered), 1.0)

    def test_raw_and_restored_bit_accuracy_are_reported_separately(self):
        with TemporaryDirectory() as tmp:
            provider = build_provider(tmp)
            latent = provider.inject_ring(provider.sample_pre_frequency_latent(6))
            detection = provider.detect_from_latent(latent)
            self.assertIsNotNone(detection["raw_bit_accuracy"])
            self.assertIsNone(detection["restored_bit_accuracy"])  # no GNR checkpoint
            self.assertIsNone(detection["classifier_probability"])
            self.assertFalse(detection["gnr_used"])


class GnrTests(unittest.TestCase):
    def test_gnr_architecture_matches_official_layer_names(self):
        raven = GmUNet(4, 4, nf=8)
        expected_prefixes = {"inc", "down1", "down2", "down3", "up2", "up3", "up4", "outc"}
        actual_prefixes = {name.split(".")[0] for name in raven.state_dict()}
        self.assertEqual(expected_prefixes, actual_prefixes)

    def test_gnr_checkpoint_loads_strictly(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "model_final.pth"
            torch.save(GmUNet(4, 4, nf=8).state_dict(), path)
            provider = build_provider(tmp, gm_use_gnr=True, gm_gnr_path=str(path), gm_model_nf=8)
            self.assertTrue(provider.gnr_available())
            model = provider.gnr
            self.assertIsInstance(model, GmUNet)
            restored = provider.restore_with_gnr(torch.zeros(1, 4, 64, 64, dtype=torch.int64))
            self.assertEqual(tuple(restored.shape), (1, 4, 64, 64))

    def test_gnr_checkpoint_with_wrong_nf_fails_closed(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "model_final.pth"
            torch.save(GmUNet(4, 4, nf=8).state_dict(), path)
            provider = build_provider(tmp, gm_use_gnr=True, gm_gnr_path=str(path), gm_model_nf=16)
            with self.assertRaises(RuntimeError):
                _ = provider.gnr

    def test_missing_gnr_checkpoint_fails_closed(self):
        with TemporaryDirectory() as tmp:
            provider = build_provider(
                tmp, gm_use_gnr=True, gm_gnr_path=str(Path(tmp) / "absent.pth")
            )
            self.assertFalse(provider.gnr_available())
            with self.assertRaises(FileNotFoundError):
                _ = provider.gnr


class ClassifierTests(unittest.TestCase):
    CLASSIFIER = Path(__file__).resolve().parents[1] / "GM_utils" / "sd21_cls2.pkl"

    def setUp(self):
        if not self.CLASSIFIER.exists():
            self.skipTest("official sd21_cls2.pkl is not present")
        try:
            import joblib  # noqa: F401
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn/joblib are not installed")

    def test_feature_order_and_probability_match_official(self):
        import joblib

        with TemporaryDirectory() as tmp:
            provider = build_provider(
                tmp, gm_use_classifier=True, gm_classifier_path=str(self.CLASSIFIER)
            )
            clf = joblib.load(self.CLASSIFIER)
            for bit_acc, ring_l1 in ((0.98, 35.0), (0.51, 120.0), (0.75, 70.0)):
                # official: x.append([correct, w_metrics_affine[i]]) with
                # w_metrics_affine = -eval_watermark(...) = -0.01 * l1
                expected = float(clf.predict_proba(np.array([[bit_acc, -0.01 * ring_l1]]))[:, 1][0])
                actual = provider.classifier_probability(bit_acc, ring_l1)
                self.assertAlmostEqual(expected, actual, places=12)

    def test_classifier_is_monotone_in_bit_accuracy(self):
        with TemporaryDirectory() as tmp:
            provider = build_provider(
                tmp, gm_use_classifier=True, gm_classifier_path=str(self.CLASSIFIER)
            )
            low = provider.classifier_probability(0.51, 120.0)
            high = provider.classifier_probability(0.99, 30.0)
            self.assertLess(low, high)


class _ConstantUNet:
    """Deterministic stand-in for the SD UNet in the inversion parity test."""

    class _Out:
        def __init__(self, sample):
            self.sample = sample

    def __call__(self, latent_model_input, t, encoder_hidden_states=None):
        return self._Out(0.01 * latent_model_input + 0.001 * float(t))


class InversionParityTests(unittest.TestCase):
    def _scheduler(self):
        from diffusers import DPMSolverMultistepScheduler

        # SD 2.1 base scheduler configuration, constructed locally (no network).
        return DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
        )

    def test_shared_helper_matches_official_forward_diffusion(self):
        latents = torch.randn(1, 4, 64, 64, generator=torch.Generator().manual_seed(1))
        embeddings = torch.zeros(1, 77, 1024)

        expected = official.forward_diffusion(
            _ConstantUNet(), self._scheduler(), latents.clone(), embeddings,
            guidance_scale=1.0, num_inference_steps=10, device=CPU,
        )
        actual = official_forward_diffusion(
            unet=_ConstantUNet(), scheduler=self._scheduler(), latents=latents.clone(),
            text_embeddings=embeddings, guidance_scale=1.0, num_inference_steps=10, device=CPU,
        )
        self.assertTrue(torch.equal(expected, actual))

    def test_generic_dpm_inverse_scheduler_is_not_equivalent(self):
        """Documents why the generic inverse scheduler must not be reused for GM."""
        from diffusers import DPMSolverMultistepInverseScheduler

        latents = torch.randn(1, 4, 64, 64, generator=torch.Generator().manual_seed(2))
        embeddings = torch.zeros(1, 77, 1024)
        unet = _ConstantUNet()

        official_result = official.forward_diffusion(
            unet, self._scheduler(), latents.clone(), embeddings,
            guidance_scale=1.0, num_inference_steps=10, device=CPU,
        )

        inverse = DPMSolverMultistepInverseScheduler(
            num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
            beta_schedule="scaled_linear", prediction_type="epsilon",
        )
        inverse.set_timesteps(10)
        generic = latents.clone() * inverse.init_noise_sigma
        for t in inverse.timesteps:
            model_out = unet(inverse.scale_model_input(generic, t), t).sample
            generic = inverse.step(model_out, t, generic).prev_sample

        self.assertFalse(
            torch.allclose(official_result, generic, atol=1e-3),
            "DPMSolverMultistepInverseScheduler unexpectedly matched the official DDIM "
            "inversion; re-check before reusing generic inversion for GM",
        )

    def test_inversion_configuration_defaults_are_official(self):
        with TemporaryDirectory() as tmp:
            provider = build_provider(tmp)
            self.assertEqual(provider.inversion_prompt, "")
            self.assertEqual(provider.inversion_guidance, 1.0)
            self.assertEqual(provider.inversion_steps, 50)
            self.assertTrue(provider.vae_sample)
            self.assertEqual(provider.vae_scaling_factor, 0.18215)

    def test_image_preprocessing_matches_official_transform(self):
        from PIL import Image

        array = (np.arange(600 * 400 * 3, dtype=np.uint8) % 251).reshape(600, 400, 3)
        image = Image.fromarray(array)
        expected = official.transform_img(image, target_size=512)
        actual = GmProvider.transform_img(image, target_size=512)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(tuple(actual.shape), (3, 512, 512))
        self.assertGreaterEqual(float(actual.min()), -1.0)
        self.assertLessEqual(float(actual.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
