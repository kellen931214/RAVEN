"""RingID key-artifact, fail-closed and generation-runner tests (Issue #3).

CPU-only and network-free: image synthesis is stubbed so the generation runner's
IO, resume and provenance behaviour can be tested without loading a model.
"""

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.wm import rid_bundle, rid_runtime  # noqa: E402
from utils.wm import ringid_provider as rid  # noqa: E402
from utils.wm.rid_bundle import RidBundle, RidBundleError  # noqa: E402


LATENT_SHAPE = (1, 4, 64, 64)
FIXTURE_SEED = 42
EVAL_BENCH_ROOT = Path(__file__).resolve().parents[1]


def make_provider(bundle_dir=None, create=False, **overrides):
    kwargs = dict(
        latent_shape=LATENT_SHAPE,
        device=torch.device("cpu"),
        rid_key_seed=FIXTURE_SEED,
        rid_bundle_dir=None if bundle_dir is None else str(bundle_dir),
        rid_create_bundle=create,
        modelid_target="stabilityai/stable-diffusion-2-1-base",
        model_revision="fp16",
        scheduler_target="DPM",
    )
    kwargs.update(overrides)
    return rid.RingIDProvider(**kwargs)


class BundleRoundTripTests(unittest.TestCase):
    def test_save_and_load_preserve_the_key_and_mask_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            provider = make_provider(bundle_dir, create=True)
            self.assertEqual(provider.state_source, "created")
            pattern_sha = rid_bundle.sha256_tensor(provider.gt_patch)

            reloaded = RidBundle.load(bundle_dir)
            self.assertEqual(reloaded.manifest["selected_pattern_sha256"], pattern_sha)
            self.assertEqual(reloaded.manifest["mask_sha256"],
                             rid_bundle.sha256_tensor(provider.watermarking_mask))
            self.assertEqual(reloaded.manifest["selected_key_index"], 628)
            self.assertEqual(reloaded.manifest["candidate_count"], 2048)
            self.assertEqual(reloaded.manifest["official_reference_commit"],
                             rid_bundle.OFFICIAL_RINGID_COMMIT)
            self.assertEqual(reloaded.manifest["spatial_shift_factor_semantics"],
                             "official_code_exact")
            self.assertTrue(torch.equal(reloaded.pattern, provider.gt_patch))

            # A second provider reloads the same identity instead of rebuilding it.
            second = make_provider(bundle_dir)
            self.assertEqual(second.state_source, "bundle")
            self.assertTrue(torch.equal(second.gt_patch, provider.gt_patch))

    def test_manifest_records_every_required_artifact_field(self):
        required = (
            "schema", "method", "official_reference_commit", "profile_name", "model_id",
            "model_revision", "latent_shape", "radius", "radius_cutoff", "ring_width",
            "rounder_ring", "heterogeneous_channels", "ring_channels",
            "quantization_values", "candidate_count", "candidate_order_sha256",
            "selected_key_index", "selected_key_id", "selected_pattern_sha256",
            "mask_sha256", "fix_gt", "spatial_shift", "spatial_shift_factor",
            "spatial_shift_factor_semantics", "rng_algorithm", "rng_seed", "rng_device",
            "rng_dtype", "bundle_config_sha256", "git_commit",
        )
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            provider = make_provider(bundle_dir, create=True)
            for field in required:
                self.assertIn(field, provider.bundle.manifest, field)

    def test_creating_a_bundle_never_overwrites_an_existing_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            bundle_dir.mkdir()
            (bundle_dir / "selected_pattern.pt").write_bytes(b"stale")
            provider = make_provider()
            with self.assertRaises(RidBundleError):
                RidBundle.create(bundle_dir, provider.gt_patch, provider.watermarking_mask,
                                 provider.bundle_manifest_config())

    def test_explicit_key_index_disagreeing_with_the_bundle_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            make_provider(bundle_dir, create=True)
            with self.assertRaises(RidBundleError):
                make_provider(bundle_dir, rid_key_index=1)
            # No explicit index: the bundle's own key is adopted.
            self.assertEqual(make_provider(bundle_dir).key_index, 628)

    def test_verification_may_not_create_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RidBundleError):
                make_provider(Path(tmp) / "missing", create=False)


class BundleFailClosedTests(unittest.TestCase):
    def _bundle(self, tmp, **overrides):
        bundle_dir = Path(tmp) / "bundle"
        make_provider(bundle_dir, create=True, **overrides)
        return bundle_dir

    def test_edited_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = self._bundle(tmp)
            manifest = json.loads((bundle_dir / "manifest.json").read_text())
            manifest["selected_key_index"] = 1
            (bundle_dir / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(RidBundleError):
                RidBundle.load(bundle_dir)

    def test_tampered_pattern_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = self._bundle(tmp)
            other = make_provider(rid_key_index=1).gt_patch
            rid_bundle.save_pattern(bundle_dir / "selected_pattern.pt", other)
            with self.assertRaises(RidBundleError):
                RidBundle.load(bundle_dir)

    def test_incompatible_model_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = self._bundle(tmp)
            with self.assertRaises(RidBundleError):
                make_provider(bundle_dir, modelid_target="stabilityai/stable-diffusion-xl-base-1.0")

    def test_incompatible_key_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = self._bundle(tmp)
            with self.assertRaises(RidBundleError):
                make_provider(bundle_dir, rid_key_seed=7)
            with self.assertRaises(RidBundleError):
                make_provider(bundle_dir, rid_profile="paper_shift_ablation",
                              time_shift_factor=0.85)

    def test_incompatible_detector_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = self._bundle(tmp)
            with self.assertRaises(RidBundleError):
                make_provider(bundle_dir, rid_inversion_steps=20)

    def test_unknown_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = self._bundle(tmp)
            manifest = json.loads((bundle_dir / "manifest.json").read_text())
            manifest["schema"] = "rid_bundle_v0"
            (bundle_dir / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(RidBundleError):
                RidBundle.load(bundle_dir)

    def test_threshold_artifact_is_bound_to_the_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = self._bundle(tmp)
            provider = make_provider(bundle_dir)
            artifact = rid_bundle.build_threshold_artifact(
                threshold=-70.0,
                binding=provider.binding_config(),
                score_definition=rid.RID_SCORE_DEFINITION,
                threshold_source="cohort_calibration",
                report_label="calibrated_deployment_verification",
            )
            provider.bundle.save_threshold(artifact)

            info = make_provider(bundle_dir).resolve_threshold()
            self.assertTrue(info["threshold_available"])
            self.assertEqual(info["threshold"], -70.0)
            self.assertEqual(info["comparison_operator"], ">=")

            # A threshold from a different key must not be reusable.
            foreign = dict(artifact)
            foreign["binding"] = dict(artifact["binding"], selected_key_index=1)
            with self.assertRaises(RidBundleError):
                rid_bundle.assert_threshold_compatible(foreign, provider.binding_config())

    def test_wrong_score_direction_is_rejected(self):
        with self.assertRaises(RidBundleError):
            rid_bundle.validate_threshold_artifact({
                "schema": rid_bundle.RID_THRESHOLD_SCHEMA,
                "threshold": 1.0,
                "score_definition": rid.RID_SCORE_DEFINITION,
                "score_direction": "lower_is_watermarked",
                "comparison_operator": ">=",
                "threshold_source": "x",
                "report_label": "legacy_or_ablation_mode",
            })


class FreshProcessReloadTests(unittest.TestCase):
    def test_key_identity_survives_a_fresh_python_process(self):
        """A key id must mean the same tensor in a process that never generated it."""
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            provider = make_provider(bundle_dir, create=True)
            expected = rid_bundle.sha256_tensor(provider.gt_patch)
            sample = provider.build_sample_latents(5)
            latent_path = Path(tmp) / "latent.pt"
            torch.save(sample["watermarked_latent"], latent_path)
            del provider

            script = textwrap.dedent(f"""
                import sys, json, torch
                sys.path.insert(0, {str(EVAL_BENCH_ROOT)!r})
                from utils.wm import rid_bundle
                from utils.wm.ringid_provider import RingIDProvider

                provider = RingIDProvider(
                    latent_shape=(1, 4, 64, 64), device=torch.device("cpu"),
                    rid_key_seed={FIXTURE_SEED}, rid_bundle_dir={str(bundle_dir)!r},
                    rid_create_bundle=False,
                    modelid_target="stabilityai/stable-diffusion-2-1-base",
                    model_revision="fp16", scheduler_target="DPM",
                )
                latent = torch.load({str(latent_path)!r}, weights_only=False)
                identified = provider.identify_key(
                    latent, candidate_indices=[0, 1, 628, 1000]
                )[0]
                print(json.dumps({{
                    "state_source": provider.state_source,
                    "pattern_sha256": rid_bundle.sha256_tensor(provider.gt_patch),
                    "score": provider.channel_distances(latent)[0]["rid_score"],
                    "predicted_key_index": identified["predicted_key_index"],
                }}))
            """)
            completed = subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True, timeout=900
            )
            self.assertEqual(completed.returncode, 0, completed.stderr[-4000:])
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            self.assertEqual(payload["state_source"], "bundle")
            self.assertEqual(payload["pattern_sha256"], expected)
            self.assertEqual(payload["predicted_key_index"], 628)
            self.assertGreater(payload["score"], -1e-6)


class _StubPipeProvider:
    """Deterministic stand-in for a diffusion pipeline (no model, no network)."""

    pipe = None

    def __init__(self):
        self.device = torch.device("cpu")
        self.calls = 0

    def get_latent_shape(self):
        return torch.Size(LATENT_SHAPE)

    def generate(self, prompts, latents, num_inference_steps, guidance_scale):
        self.calls += 1
        flat = latents.detach().cpu().to(torch.float32).flatten()[: 64 * 64]
        pixels = ((flat - flat.min()) / (flat.max() - flat.min() + 1e-9) * 255)
        image = Image.fromarray(
            pixels.reshape(64, 64).to(torch.uint8).cpu().numpy(), mode="L"
        ).convert("RGB")
        return {"images_PIL": [image]}


class GenerationRunnerTests(unittest.TestCase):
    """``run_watermark.py --wm_type RID`` IO, resume and pairing behaviour."""

    def _run(self, tmp, extra_argv=(), out_name="gen"):
        import run_watermark

        out_dir = Path(tmp) / out_name
        bundle_dir = Path(tmp) / "bundle"
        argv = [
            "--wm_type", "RID",
            "--rid_profile", "official_sd21",
            "--num", "3",
            "--seed", "0",
            "--rid_bundle_dir", str(bundle_dir),
            "--out_dir", str(out_dir),
            *extra_argv,
        ]
        args = run_watermark.build_parser().parse_known_args(argv)[0]

        stub = _StubPipeProvider()
        original_pipe = rid_runtime.build_pipe_provider
        original_prompts = run_watermark.get_text_prompts
        original_device = run_watermark.DEVICE
        run_watermark.DEVICE = torch.device("cpu")  # no model, no GPU needed
        rid_runtime.build_pipe_provider = lambda a, d: stub
        run_watermark.get_text_prompts = lambda num_prompts, dataset_id: [
            f"a photo of sample {i}" for i in range(num_prompts)
        ]
        try:
            rows = run_watermark.run_rid_generation(args, argv)
        finally:
            rid_runtime.build_pipe_provider = original_pipe
            run_watermark.get_text_prompts = original_prompts
            run_watermark.DEVICE = original_device
        return rows, out_dir, bundle_dir, stub

    def test_multi_sample_generation_never_overwrites_and_pairs_latents(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, out_dir, bundle_dir, stub = self._run(tmp)

            self.assertEqual(len(rows), 3)
            images = sorted((out_dir / "images" / "watermarked").glob("*.png"))
            cleans = sorted((out_dir / "images" / "no_watermark").glob("*.png"))
            self.assertEqual([p.name for p in images], ["000000.png", "000001.png", "000002.png"])
            self.assertEqual([p.name for p in cleans], ["000000.png", "000001.png", "000002.png"])
            self.assertEqual(len({rid_bundle.sha256_file(p) for p in images}), 3)

            # Independent complete initial latents, one constant key identity.
            self.assertEqual(len({row["post_injection_latent_sha256"] for row in rows}), 3)
            self.assertEqual(len({row["clean_base_latent_sha256"] for row in rows}), 3)
            self.assertEqual(len({row["selected_pattern_sha256"] for row in rows}), 1)
            for row in rows:
                self.assertEqual(row["clean_base_latent_sha256"],
                                 row["pre_injection_latent_sha256"])
                self.assertEqual(row["sample_seed"], row["sample_id"])

            manifest = json.loads((out_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["report_label"], "official_profile_raw_scores")
            self.assertTrue(manifest["rid_profile_is_official"])
            self.assertEqual(manifest["entrypoint"],
                             "eval_bench_wm/run_watermark.py:run_rid_generation")
            self.assertTrue((bundle_dir / "manifest.json").exists())
            self.assertEqual(len(rid_bundle.read_jsonl(out_dir / "results.jsonl")), 3)

    def test_resume_reproduces_the_same_samples_and_never_regenerates(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, out_dir, _, first_stub = self._run(tmp)
            hashes = [row["image_sha256"] for row in rows]

            rows2, out_dir2, _, second_stub = self._run(tmp)
            self.assertEqual(out_dir, out_dir2)
            self.assertEqual([row["image_sha256"] for row in rows2], hashes)
            self.assertEqual(second_stub.calls, 0)

    def test_resume_with_a_different_seed_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp)
            with self.assertRaises(RuntimeError):
                self._run(tmp, extra_argv=["--seed", "5"])

    def test_generation_requires_a_bundle_directory(self):
        import run_watermark

        with tempfile.TemporaryDirectory() as tmp:
            argv = ["--wm_type", "RID", "--num", "1", "--out_dir", str(Path(tmp) / "o")]
            args = run_watermark.build_parser().parse_known_args(argv)[0]
            stub = _StubPipeProvider()
            original = rid_runtime.build_pipe_provider
            original_device = run_watermark.DEVICE
            run_watermark.DEVICE = torch.device("cpu")
            rid_runtime.build_pipe_provider = lambda a, d: stub
            try:
                with self.assertRaises(RuntimeError):
                    run_watermark.run_rid_generation(args, argv)
            finally:
                rid_runtime.build_pipe_provider = original
                run_watermark.DEVICE = original_device


class SuspectScoringTests(unittest.TestCase):
    def test_corrupt_image_is_an_error_record_not_a_negative_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            provider = make_provider(bundle_dir, create=True)
            broken = Path(tmp) / "broken.png"
            broken.write_bytes(b"not an image")
            row = rid_runtime.score_image(
                provider, _StubPipeProvider(), broken, image_index=0,
                threshold_info={"threshold": -70.0, "threshold_source": "user_supplied",
                                "comparison_operator": ">="},
            )
            self.assertEqual(row["status"], "error")
            self.assertIsNone(row["detection_success"])
            self.assertIsNone(row["rid_score"])
            self.assertIsNotNone(row["error"])
            self.assertEqual(row["image_index"], 0)

    def test_result_rows_carry_every_required_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider = make_provider(Path(tmp) / "bundle", create=True)
            row = rid_runtime.score_image(
                provider, _StubPipeProvider(), Path(tmp) / "missing.png", image_index=3,
                threshold_info={"threshold": None, "threshold_source": None,
                                "comparison_operator": ">="},
            )
            for field in rid_runtime.RESULT_FIELDS:
                if field in ("model_id", "model_revision", "official_reference_commit",
                             "git_branch", "git_commit"):
                    continue  # merged from run provenance by the runner
                self.assertIn(field, row, field)


class RocTests(unittest.TestCase):
    def test_cohort_roc_uses_the_canonical_score_and_reports_empirical_fpr(self):
        positives = [-10.0, -11.0, -12.0, -9.0]
        negatives = [-70.0, -71.0, -69.0, -72.0]
        roc = rid_runtime.official_roc(
            positives, negatives, 0.01, score_definition=rid.RID_SCORE_DEFINITION
        )
        self.assertEqual(roc["score_definition"], rid.RID_SCORE_DEFINITION)
        self.assertEqual(roc["score_direction"], "higher_is_watermarked")
        self.assertEqual(roc["comparison_operator"], ">=")
        self.assertEqual(roc["roc_auc"], 1.0)
        self.assertEqual(roc["tpr_at_target_fpr"], 1.0)
        self.assertEqual(roc["empirical_fpr"], 0.0)
        self.assertEqual(roc["positive_count"], 4)
        self.assertEqual(roc["negative_count"], 4)


if __name__ == "__main__":
    unittest.main()
