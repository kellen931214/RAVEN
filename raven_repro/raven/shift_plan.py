"""Attack shift planning with deterministic seed arithmetic.

Every attack seed is ``base_seed + numeric_run_id`` so re-running the same
run_id always produces the same shift.  Non-numeric run_ids are hashed
deterministically.
"""

from __future__ import annotations

import random
from typing import Any, Mapping

from .eval_protocol import canonical_json_hash


def compute_attack_seed(base_seed: int, run_id: str) -> int:
    """Compute the per-sample attack seed from the base seed and run ID.

    ``attack_seed = base_seed + numeric_run_id``

    For non-numeric run IDs the first 8 hex digits of the canonical JSON hash
    are used as the numeric component.  This is the same arithmetic the legacy
    ``planned_shift()`` in ``run_raven_formal_eval.py`` used.
    """
    try:
        numeric_id = int(run_id)
    except (ValueError, TypeError):
        numeric_id = int(canonical_json_hash({"run_id": str(run_id)})[:8], 16)
    return int(base_seed) + numeric_id


def plan_shift(
    run_id: str,
    config: Mapping[str, Any],
    *,
    index: int = 0,
) -> tuple[float, float, int]:
    """Return (dx, dy, attack_seed) for one sample.

    Parameters
    ----------
    run_id : str
        Per-sample identifier used to derive the attack seed.
    config : Mapping
        Normalized config carrying at least ``base_seed``, ``shift_mode``,
        ``shift_magnitude_min``, ``shift_magnitude_max``, and optionally
        ``shift_x`` / ``shift_y``.
    index : int
        Enumeration index (used only for logging provenance; the shift is
        derived from the seed, not the index).

    Returns
    -------
    (dx, dy, attack_seed) : tuple[float, float, int]
        Planned image-pixel displacement and the per-sample attack seed.
    """
    base_seed = int(config["base_seed"])
    attack_seed = compute_attack_seed(base_seed, str(run_id))
    mode = str(config.get("shift_mode", "random"))

    if mode == "none":
        return 0.0, 0.0, attack_seed

    if mode == "fixed":
        dx = float(config["shift_x"])
        dy = float(config["shift_y"])
        return dx, dy, attack_seed

    if mode != "random":
        raise ValueError(f"Unsupported shift_mode: {mode!r}")

    shift_min = int(config.get("shift_magnitude_min", 24))
    shift_max = int(config.get("shift_magnitude_max", 32))
    if shift_min > shift_max:
        raise ValueError(
            f"shift_magnitude_min ({shift_min}) > shift_magnitude_max ({shift_max})"
        )

    magnitudes = tuple(range(shift_min, shift_max + 1))

    # Seed the per-sample RNG deterministically from the run_id and attack_seed.
    # This matches the legacy ``paper_random_independent_axes`` protocol.
    rng_seed = int(
        canonical_json_hash(
            {"protocol": "random_independent_axes", "run_id": str(run_id), "attack_seed": attack_seed}
        )[:16],
        16,
    )
    rng = random.Random(rng_seed)

    # Magnitude and sign are independently sampled for each axis.
    dx = rng.choice(magnitudes) * rng.choice((-1, 1))
    dy = rng.choice(magnitudes) * rng.choice((-1, 1))
    return float(dx), float(dy), attack_seed
