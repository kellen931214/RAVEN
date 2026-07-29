"""Guard the frozen official inversion parity evidence.

The evidence itself is produced on a GPU by
``eval_bench_wm/tools/hsqr_inversion_parity.py``, which runs the frozen official
``ddim_invert`` and ``utils/wm/sfw_inversion`` against the same loaded pipeline.
These tests are the cheap, CPU-only contract around that artifact: they assert
that the recorded comparison actually supports the parity label the code
advertises, and that the label never silently overstates what was measured.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _TESTS_DIR.parent
for _path in (str(_PKG_ROOT), str(_TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import official_sfwmark_source as official_src
from utils.wm import sfw_inversion

EVIDENCE_PATH = _TESTS_DIR / "fixtures" / "hsqr_inversion_parity_evidence.json"

#: Every artifact Issue #5 requires the parity evidence to cover.
REQUIRED_ARTIFACTS = (
    "preprocessed_input_tensor",
    "vae_latent",
    "final_recovered_latent",
)


def load_evidence() -> dict:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


class TestInversionParityEvidence(unittest.TestCase):
    def setUp(self):
        self.evidence = load_evidence()

    def test_evidence_exists_and_is_pinned_to_the_official_commit(self):
        self.assertEqual(self.evidence["schema"], "hsqr_inversion_parity_v1")
        self.assertEqual(
            self.evidence["provenance"]["official_commit"],
            official_src.OFFICIAL_SFWMARK_COMMIT,
        )
        self.assertEqual(
            self.evidence["provenance"]["official_utils_sha256"],
            official_src.OFFICIAL_UTILS_SHA256,
        )

    def test_required_artifacts_are_all_compared(self):
        compared = {entry["artifact"] for entry in self.evidence["comparisons"]}
        for artifact in REQUIRED_ARTIFACTS:
            with self.subTest(artifact=artifact):
                self.assertIn(artifact, compared)

    def test_intermediate_latents_are_sampled(self):
        self.assertGreaterEqual(len(self.evidence["intermediate_latents"]), 3)

    def test_inverse_scheduler_timesteps_match(self):
        timesteps = self.evidence["timesteps"]
        self.assertTrue(timesteps["match"])
        self.assertEqual(timesteps["official"], timesteps["raven"])
        self.assertEqual(timesteps["count_official"], 50)

    def test_every_compared_tensor_is_bitwise_identical(self):
        entries = self.evidence["comparisons"] + self.evidence["intermediate_latents"]
        self.assertTrue(entries)
        for entry in entries:
            with self.subTest(artifact=entry["artifact"]):
                self.assertTrue(entry["shape_match"])
                self.assertEqual(entry["max_abs_diff"], 0.0)
                self.assertTrue(entry["bitwise_identical"])
                self.assertGreater(entry["elements_compared"], 0)

    def test_hsqr_distance_and_score_agree(self):
        detection = self.evidence["hsqr_detection"]
        self.assertEqual(detection["l1_abs_diff"], 0.0)
        self.assertEqual(detection["score_abs_diff"], 0.0)
        self.assertEqual(
            detection["official_score"], -detection["official_l1_distance"]
        )
        self.assertEqual(detection["raven_score"], -detection["raven_l1_distance"])

    def test_no_stubbed_dependency_influenced_the_official_path(self):
        self.assertEqual(self.evidence["official_stubs_touched"], [])

    def test_parity_label_matches_the_measured_evidence(self):
        """The advertised label must be earned by the artifact, in both directions."""
        entries = self.evidence["comparisons"] + self.evidence["intermediate_latents"]
        all_bitwise = all(entry.get("bitwise_identical") for entry in entries)
        self.assertEqual(
            sfw_inversion.SFW_INVERSION_PARITY_STATUS
            == "official_code_parity_verified_bitwise",
            all_bitwise,
            "SFW_INVERSION_PARITY_STATUS disagrees with the recorded comparison",
        )

    def test_weights_parity_is_reported_separately_and_honestly(self):
        """A mirror run must never be labelled as official-weight parity."""
        used_official = self.evidence["model"]["official_model_used"]
        self.assertEqual(
            self.evidence["model"]["official_model_id"],
            "stabilityai/stable-diffusion-2-1-base",
        )
        if not used_official:
            self.assertEqual(
                sfw_inversion.SFW_INVERSION_WEIGHTS_PARITY,
                "official_weights_unavailable_not_verified",
            )
            self.assertNotEqual(
                self.evidence["model"]["model_id"],
                self.evidence["model"]["official_model_id"],
            )

    def test_evidence_states_what_is_not_claimed(self):
        self.assertIn("not_claimed", self.evidence)
        self.assertIn("official", self.evidence["not_claimed"])

    def test_evidence_path_constant_points_at_this_file(self):
        referenced = _PKG_ROOT.parent / sfw_inversion.SFW_INVERSION_PARITY_EVIDENCE
        self.assertEqual(referenced.resolve(), EVIDENCE_PATH.resolve())

    def test_gpu_preflight_was_recorded(self):
        preflight = self.evidence["gpu_preflight"]
        self.assertTrue(preflight["cuda_available"])
        self.assertIn("device_name", preflight)


if __name__ == "__main__":
    unittest.main()
