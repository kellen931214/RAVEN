"""Three focused fail-closed regressions for the GaussMarker path.

1. an existing bundle is rejected when the detector configuration differs;
2. an unpaired positive/negative cohort is never labelled official paper evaluation;
3. a changed-seed resume leaves ``run_manifest.json`` byte-identical.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from utils.wm import gm_bundle, gm_runtime
from utils.wm.gm_bundle import GmBundleError
from utils.wm.gm_provider import GmProvider


CPU = torch.device("cpu")


def provider_kwargs(**overrides):
    kwargs = dict(
        latent_shape=(1, 4, 64, 64),
        device=CPU,
        gm_torch_dtype="float32",
        gm_use_gnr=False,
        gm_use_classifier=False,
        gm_watermark_bits_seed=31337,
        modelid_target="stabilityai/stable-diffusion-2-1-base",
        scheduler_target="DPM",
    )
    kwargs.update(overrides)
    return kwargs


class IncompatibleBundleDetectorConfigTests(unittest.TestCase):
    def test_bundle_rejects_a_different_detector_configuration(self):
        with TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            GmProvider(**provider_kwargs(gm_bundle_dir=str(bundle_dir), gm_create_bundle=True))

            # Same watermark identity, different detector configuration.
            for field, value in (
                ("gm_inversion_steps", 25),
                ("gm_inversion_guidance", 7.5),
                ("gm_vae_sample", False),
                ("gm_classifier_type", 1),
                ("gm_model_nf", 64),
                ("modelid_target", "CompVis/stable-diffusion-v1-4"),
            ):
                with self.subTest(field=field):
                    with self.assertRaises(GmBundleError):
                        GmProvider(**provider_kwargs(gm_bundle_dir=str(bundle_dir), **{field: value}))

    def test_manifest_missing_a_required_field_is_rejected(self):
        with TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            GmProvider(**provider_kwargs(gm_bundle_dir=str(bundle_dir), gm_create_bundle=True))

            manifest_path = bundle_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            del manifest["inversion_steps"]
            manifest["bundle_config_sha256"] = gm_bundle.GmBundle.config_sha256(manifest)
            manifest_path.write_text(gm_bundle.canonical_json(manifest) + "\n")

            with self.assertRaises(GmBundleError) as ctx:
                GmProvider(**provider_kwargs(gm_bundle_dir=str(bundle_dir)))
            self.assertIn("inversion_steps", str(ctx.exception))

    def test_non_official_bundle_is_never_relabelled_as_official(self):
        with TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / "bundle"
            GmProvider(
                **provider_kwargs(
                    gm_bundle_dir=str(bundle_dir),
                    gm_create_bundle=True,
                    gm_profile_is_official=False,
                )
            )
            # The CLI now claims the official profile; the bundle must win.
            reloaded = GmProvider(
                **provider_kwargs(gm_bundle_dir=str(bundle_dir), gm_profile_is_official=True)
            )
            self.assertFalse(reloaded.profile_is_official)
            self.assertFalse(reloaded.bundle_profile_is_official)
            self.assertEqual(
                reloaded.resolve_threshold()["report_label"], "legacy_or_ablation_mode"
            )
            self.assertEqual(
                gm_runtime.cohort_report_label("paper_eval", reloaded.profile_is_official, True),
                "legacy_or_ablation_mode",
            )


class UnpairedCohortTests(unittest.TestCase):
    def _cohorts(self, root: Path, with_metadata: bool, negative_seed_offset: int = 0):
        positives, negatives = root / "pos" / "images", root / "neg" / "images"
        for directory in (positives, negatives):
            directory.mkdir(parents=True)
        for index in range(2):
            name = f"{index:06d}"
            (positives / f"{name}.png").write_bytes(b"pos")
            (negatives / f"{name}.png").write_bytes(b"neg")
            if with_metadata:
                for role, directory, offset in (
                    ("pos", positives, 0),
                    ("neg", negatives, negative_seed_offset),
                ):
                    meta_dir = directory.parent / "sample_metadata"
                    meta_dir.mkdir(exist_ok=True)
                    (meta_dir / f"{name}.json").write_text(json.dumps({
                        "sample_id": index,
                        "prompt_sha256": gm_bundle.sha256_text(f"prompt {index}"),
                        "sample_seed": index + offset,
                    }))
        return sorted(positives.iterdir()), sorted(negatives.iterdir())

    def test_cohorts_without_pairing_provenance_are_not_official(self):
        with TemporaryDirectory() as tmp:
            positives, negatives = self._cohorts(Path(tmp), with_metadata=False)
            pairing = gm_runtime.resolve_pairing(positives, negatives)

            # Equal cohort sizes alone must not be accepted as a paired cohort.
            self.assertEqual(len(positives), len(negatives))
            self.assertFalse(pairing["paired"])
            self.assertIn("sample metadata", pairing["reason"])
            self.assertEqual(
                gm_runtime.cohort_report_label("paper_eval", True, pairing["paired"]),
                "legacy_or_ablation_mode",
            )

    def test_mismatched_generation_seed_breaks_the_pairing(self):
        with TemporaryDirectory() as tmp:
            positives, negatives = self._cohorts(
                Path(tmp), with_metadata=True, negative_seed_offset=100
            )
            pairing = gm_runtime.resolve_pairing(positives, negatives)
            self.assertFalse(pairing["paired"])
            self.assertIn("sample_seed", pairing["reason"])

    def test_paired_cohort_with_full_provenance_is_official(self):
        with TemporaryDirectory() as tmp:
            positives, negatives = self._cohorts(Path(tmp), with_metadata=True)
            pairing = gm_runtime.resolve_pairing(positives, negatives)
            self.assertTrue(pairing["paired"], pairing["reason"])
            self.assertEqual(len(pairing["pairs"]), 2)
            self.assertIsNotNone(pairing["pairing_sha256"])
            self.assertEqual(
                gm_runtime.cohort_report_label("paper_eval", True, pairing["paired"]),
                "official_paper_evaluation",
            )

    def test_pair_manifest_alone_establishes_the_pairing(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            positives, negatives = self._cohorts(root, with_metadata=False)
            manifest_path = root / "pairs.json"
            manifest_path.write_text(json.dumps({
                "protocol": "official_paper_eval",
                "pairs": [
                    {
                        "positive": positive.name,
                        "negative": negative.name,
                        "sample_id": index,
                        "prompt_sha256": gm_bundle.sha256_text(f"prompt {index}"),
                        "positive_sample_seed": index,
                        "negative_sample_seed": 1000 + index,
                        "distortion_config_sha256": "d" * 64,
                        "distortion_seed": 7,
                    }
                    for index, (positive, negative) in enumerate(zip(positives, negatives))
                ],
            }))

            # No sidecar metadata exists anywhere in either cohort.
            self.assertIsNone(gm_runtime.load_pair_metadata(positives[0]))

            pairing = gm_runtime.resolve_pairing(positives, negatives, manifest_path)
            self.assertTrue(pairing["paired"], pairing["reason"])
            self.assertEqual(pairing["protocol"], "official_paper_eval")
            self.assertEqual([p["pairing_source"] for p in pairing["pairs"]], ["pair_manifest"] * 2)
            self.assertEqual(pairing["pairs"][0]["negative_sample_seed"], 1000)
            self.assertEqual(
                gm_runtime.cohort_report_label("paper_eval", True, pairing["paired"]),
                "official_paper_evaluation",
            )

    def test_paper_eval_fails_closed_without_pairing(self):
        import run_verify_watermark

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._cohorts(root, with_metadata=False)
            args = run_verify_watermark.build_parser().parse_args([
                "--mode", "paper_eval",
                "--positive_path", str(root / "pos" / "images"),
                "--negative_path", str(root / "neg" / "images"),
            ])
            with self.assertRaises(SystemExit):
                run_verify_watermark._resolve_inputs(args)

            args.allow_unmatched_cohorts = True
            inputs = run_verify_watermark._resolve_inputs(args)
            self.assertFalse(inputs["pairing"]["paired"])


class RunManifestResumeTests(unittest.TestCase):
    def _manifest(self, path: Path, sha: str) -> str:
        payload = {
            "run_config_sha256": sha,
            "created_utc": "2026-07-28T00:00:00Z",
            "base_seed": 0,
            "entrypoint": "eval_bench_wm/run_watermark.py:run_gm_generation",
        }
        path.write_text(gm_bundle.canonical_json(payload) + "\n", encoding="utf-8")
        return gm_bundle.sha256_file(path)

    def test_changed_seed_leaves_the_run_manifest_untouched(self):
        with TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "run_manifest.json"
            before = self._manifest(manifest_path, "a" * 64)

            with self.assertRaises(RuntimeError) as ctx:
                gm_runtime.assert_run_manifest_compatible(manifest_path, "b" * 64)
            self.assertIn("nothing was modified", str(ctx.exception))

            self.assertEqual(before, gm_bundle.sha256_file(manifest_path))

    def test_compatible_manifest_is_returned_verbatim(self):
        with TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "run_manifest.json"
            before = self._manifest(manifest_path, "a" * 64)

            manifest = gm_runtime.assert_run_manifest_compatible(manifest_path, "a" * 64)
            self.assertEqual(manifest["created_utc"], "2026-07-28T00:00:00Z")
            self.assertEqual(before, gm_bundle.sha256_file(manifest_path))

    def test_incompatible_resume_never_creates_a_new_bundle(self):
        import run_watermark

        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run"
            out_dir.mkdir()
            manifest_path = out_dir / "run_manifest.json"
            before = self._manifest(manifest_path, "a" * 64)
            bundle_dir = Path(tmp) / "new_bundle"  # does not exist

            argv = [
                "--wm_type", "GM",
                "--out_dir", str(out_dir),
                "--gm_bundle_dir", str(bundle_dir),
                "--seed", "999",
                "--num", "1",
            ]
            args = run_watermark.build_parser().parse_args(argv)

            with self.assertRaises(RuntimeError) as ctx:
                run_watermark.run_gm_generation(args, argv)
            self.assertIn("Nothing was modified", str(ctx.exception))

            self.assertFalse(bundle_dir.exists())
            self.assertEqual([p.name for p in out_dir.iterdir()], ["run_manifest.json"])
            self.assertEqual(before, gm_bundle.sha256_file(manifest_path))

    def test_absent_manifest_may_be_created(self):
        with TemporaryDirectory() as tmp:
            self.assertIsNone(
                gm_runtime.assert_run_manifest_compatible(Path(tmp) / "run_manifest.json", "a" * 64)
            )


if __name__ == "__main__":
    unittest.main()
