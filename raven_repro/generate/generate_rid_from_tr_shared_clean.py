#!/usr/bin/env python3
"""Generate RingID watermarked images from canonical TR clean artifacts."""

from __future__ import annotations

from shared_tr_clean_fourier import MethodSpec, main_for_spec
from utils.wm.ringid_provider import RID_SHARED_TR_CLEAN_MODE

SPEC = MethodSpec(
        method='RID',
        wm_name='RingID',
        protocol='ringid_shared_tr_clean_v2',
        protocol_mode=RID_SHARED_TR_CLEAN_MODE,
        provider_module='ringid_provider',
        provider_class='RingIDProvider',
        provider_entrypoint_field='rid_provider_entrypoint_sha256',
        bundle_arg='rid_bundle_dir',
        create_bundle_arg='rid_create_bundle',
        shared_profile='raven_shared_tr_clean_v2_not_official_ringid_generation',
        pairing_relation='shared_tr_clean_latent_ringid_fourier_replacement',
        official_source_repository='showlab/RingID',
        official_source_commit='45631a59aecd7d63ccdb640aaaf3e616fdb89fb9',
        official_math_claim='official RingID pattern and Fourier-domain replacement applied to the supplied canonical Tree-Ring latent',
        not_claimed='NOT end-to-end official RingID generation parity: this cohort supplies the canonical Tree-Ring clean latent and uses the Tree-Ring float32 DDIM configuration.',
)


def main() -> int:
    return main_for_spec(SPEC)


if __name__ == "__main__":
    raise SystemExit(main())
