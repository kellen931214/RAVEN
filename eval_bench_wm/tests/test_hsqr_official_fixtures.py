"""Element-by-element HSQR parity against the frozen official SFWMark code.

Two layers:

* :class:`TestCommittedOfficialFixtures` always runs. It compares
  ``HSQRProvider`` element by element against ``fixtures/hsqr_official_fixtures.json``,
  which was produced by executing the frozen official ``src/utils.py`` (see
  ``tools/generate_hsqr_official_fixtures.py``). This is what makes the parity
  claim checkable in a normal clone, without network access.

* :class:`TestLiveOfficialSource` re-derives everything from the official source
  itself and is skipped when that checkout is absent.

Keys 0, 1, 1024 and 2047 (first, second, a middle key, the last) are covered, as
Issue #5 requires.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import torch

_TESTS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _TESTS_DIR.parent
for _path in (str(_PKG_ROOT), str(_TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import official_sfwmark_source as official_src
from utils.wm.hsqr_provider import (
    HSQRProvider,
    OFFICIAL_BASE_KEY_SEED,
    OFFICIAL_PROFILE_NAME,
    QR_TARGET_MAGNITUDE,
)

sys.path.insert(0, str(_PKG_ROOT / "tools"))
from generate_hsqr_official_fixtures import (  # noqa: E402
    REQUIRED_KEYS,
    build_base_latent,
    sha256_tensor,
    unpack_bool_tensor,
)

FIXTURE_PATH = _TESTS_DIR / "fixtures" / "hsqr_official_fixtures.json"
LATENT_SHAPE = (1, 4, 64, 64)


def load_fixtures() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def official_provider(key_index: int) -> HSQRProvider:
    return HSQRProvider(
        latent_shape=LATENT_SHAPE,
        device="cpu",
        hsqr_profile=OFFICIAL_PROFILE_NAME,
        hsqr_base_key_seed=OFFICIAL_BASE_KEY_SEED,
        hsqr_key_index=key_index,
        modelid_target="stabilityai/stable-diffusion-2-1-base",
        model_revision=None,
        scheduler_target="DDIM",
        resolution=512,
    )


class TestFixtureProvenance(unittest.TestCase):
    """The fixture file must stay pinned to the frozen official commit."""

    def setUp(self):
        self.fixtures = load_fixtures()

    def test_schema_and_commit_are_pinned(self):
        self.assertEqual(self.fixtures["schema"], "hsqr_official_fixtures_v1")
        provenance = self.fixtures["provenance"]
        self.assertEqual(provenance["official_commit"],
                         official_src.OFFICIAL_SFWMARK_COMMIT)
        self.assertEqual(provenance["official_utils_sha256"],
                         official_src.OFFICIAL_UTILS_SHA256)

    def test_fixtures_declare_official_derivation(self):
        self.assertIn("executed", self.fixtures["derivation"])
        self.assertIn("not transcribed", self.fixtures["derivation"])

    def test_required_keys_are_present(self):
        self.assertEqual(
            sorted(int(k) for k in self.fixtures["keys"]), sorted(REQUIRED_KEYS)
        )

    def test_official_constants_match_the_provider(self):
        constants = self.fixtures["official_constants"]
        self.assertEqual(constants["base_key_seed"], OFFICIAL_BASE_KEY_SEED)
        self.assertEqual(constants["wm_capacity"], 2048)
        self.assertEqual(constants["delta"], 0)
        self.assertEqual(constants["center_start"], 10)
        self.assertEqual(constants["center_end"], 54)
        self.assertEqual(constants["watermark_channel"], [3])
        self.assertEqual(constants["target_magnitude"], QR_TARGET_MAGNITUDE)

    def test_base_latent_is_reproducible(self):
        """A change in torch's RNG must surface here, not as a silent drift."""
        self.assertEqual(
            sha256_tensor(build_base_latent()), self.fixtures["base_latent"]["sha256"]
        )


class TestCommittedOfficialFixtures(unittest.TestCase):
    """Provider vs official fixtures, element by element, at zero tolerance."""

    def setUp(self):
        self.fixtures = load_fixtures()
        self.base_latent = build_base_latent()

    def test_key_seed_and_payload(self):
        for key_index in REQUIRED_KEYS:
            entry = self.fixtures["keys"][str(key_index)]
            with self.subTest(key=key_index):
                provider = official_provider(key_index)
                self.assertEqual(provider.key_seed(key_index), entry["key_seed"])
                self.assertEqual(provider.payload_text(key_index), entry["payload"])

    def test_qr_pattern_matches_every_element(self):
        for key_index in REQUIRED_KEYS:
            entry = self.fixtures["keys"][str(key_index)]
            with self.subTest(key=key_index):
                expected = unpack_bool_tensor(
                    entry["pattern_bits_b64"], tuple(entry["pattern_shape"])
                )
                actual = official_provider(key_index).make_pattern(key_index)
                self.assertEqual(tuple(actual.shape), tuple(entry["pattern_shape"]))
                self.assertEqual(actual.dtype, torch.bool)
                differing = int((expected != actual).sum())
                self.assertEqual(
                    differing, 0,
                    f"key {key_index}: {differing} of {expected.numel()} QR elements "
                    "differ from the official pattern",
                )
                self.assertEqual(int(actual.sum()), entry["pattern_ones"])
                self.assertEqual(sha256_tensor(actual), entry["pattern_sha256"])

    def test_watermarked_latent_is_bitwise_identical(self):
        for key_index in REQUIRED_KEYS:
            entry = self.fixtures["keys"][str(key_index)]
            with self.subTest(key=key_index):
                provider = official_provider(key_index)
                pattern = provider.make_pattern(key_index)
                watermarked = provider.inject(self.base_latent.clone(), pattern=pattern)
                watermarked = watermarked.detach().cpu().to(torch.float32)
                self.assertEqual(
                    sha256_tensor(watermarked), entry["watermarked_latent_sha256"],
                    f"key {key_index}: injected latent differs from the official one",
                )

    def test_distances_and_scores_match(self):
        for key_index in REQUIRED_KEYS:
            entry = self.fixtures["keys"][str(key_index)]
            with self.subTest(key=key_index):
                provider = official_provider(key_index)
                pattern = provider.make_pattern(key_index)
                watermarked = provider.inject(self.base_latent.clone(), pattern=pattern)
                watermarked = watermarked.detach().cpu().to(torch.float32)

                wm = provider.l1_distances(watermarked, pattern=pattern)
                clean = provider.l1_distances(self.base_latent, pattern=pattern)
                self.assertEqual(wm, entry["watermarked_l1_distances"])
                self.assertEqual(clean, entry["clean_l1_distances"])

                self.assertEqual(
                    [provider.score_from_distance(d) for d in wm],
                    entry["watermarked_scores"],
                )
                self.assertEqual(
                    [provider.score_from_distance(d) for d in clean],
                    entry["clean_scores"],
                )

    def test_watermarked_scores_beat_clean_scores(self):
        """Sign convention: higher score means watermarked."""
        for key_index in REQUIRED_KEYS:
            entry = self.fixtures["keys"][str(key_index)]
            with self.subTest(key=key_index):
                self.assertTrue(
                    min(entry["watermarked_scores"]) > max(entry["clean_scores"]),
                    "official fixtures do not separate watermarked from clean",
                )

    def test_distances_are_per_batch_item(self):
        """Regression: the detector once scored batch index 0 for every item."""
        entry = self.fixtures["keys"]["0"]
        self.assertEqual(len(entry["clean_l1_distances"]), self.base_latent.shape[0])
        self.assertEqual(
            len(set(entry["clean_l1_distances"])), self.base_latent.shape[0],
            "distinct base latents must produce distinct distances",
        )


@unittest.skipUnless(
    official_src.official_available(),
    f"frozen official SFWMark source unavailable; set ${official_src.OFFICIAL_SRC_ENV}",
)
class TestLiveOfficialSource(unittest.TestCase):
    """Re-derive parity from the official source when it is present."""

    @classmethod
    def setUpClass(cls):
        cls.utils = official_src.load_official_utils()
        cls.base_latent = build_base_latent()

    def test_official_constants_are_what_we_pinned(self):
        self.assertEqual(int(self.utils.w_seed), OFFICIAL_BASE_KEY_SEED)
        self.assertEqual(int(self.utils.wm_capacity), 2048)
        self.assertEqual(int(self.utils.delta), 0)
        self.assertEqual(int(self.utils.start), 10)
        self.assertEqual(int(self.utils.end), 54)
        self.assertEqual(list(self.utils.HSQR_WATERMARK_CHANNEL), [3])

    def test_committed_fixtures_reproduce_from_official_source(self):
        from generate_hsqr_official_fixtures import build_fixtures

        regenerated = build_fixtures()
        committed = load_fixtures()
        volatile = {"generated_utc"}
        self.assertEqual(
            {k: v for k, v in regenerated.items() if k not in volatile},
            {k: v for k, v in committed.items() if k not in volatile},
        )

    def test_provider_matches_official_pattern_live(self):
        from generate_hsqr_official_fixtures import official_pattern

        for key_index in REQUIRED_KEYS:
            with self.subTest(key=key_index):
                expected = official_pattern(self.utils, key_index)
                actual = official_provider(key_index).make_pattern(key_index)
                self.assertEqual(int((expected != actual).sum()), 0)

    def test_no_stub_is_reachable_from_the_official_hsqr_path(self):
        """Stubbed dependencies cannot influence any fixture value."""
        from generate_hsqr_official_fixtures import (
            official_distances, official_inject, official_pattern,
        )

        official_src.reset_stub_usage()
        pattern = official_pattern(self.utils, 0)
        latent = official_inject(self.utils, self.base_latent.clone(), pattern)
        official_distances(self.utils, latent, pattern)
        self.assertEqual(official_src.stub_usage(), [])


if __name__ == "__main__":
    unittest.main()
