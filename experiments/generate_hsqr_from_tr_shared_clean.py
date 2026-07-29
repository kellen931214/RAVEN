#!/usr/bin/env python3
"""Generate HSQR watermarked images from canonical TR clean artifacts."""

from __future__ import annotations

from shared_tr_clean_fourier import MethodSpec, main_for_spec
from utils.wm.hsqr_provider import HSQR_SHARED_TR_CLEAN_MODE

SPEC = MethodSpec(
        method='HSQR',
        wm_name='HSQR',
        protocol='hsqr_shared_tr_clean_v2',
        protocol_mode=HSQR_SHARED_TR_CLEAN_MODE,
        provider_module='hsqr_provider',
        provider_class='HSQRProvider',
        provider_entrypoint_field='hsqr_provider_entrypoint_sha256',
        bundle_arg='hsqr_bundle_dir',
        create_bundle_arg='hsqr_create_bundle',
        shared_profile='raven_shared_tr_clean_v2_not_official_hsqr_generation',
        pairing_relation='shared_tr_clean_latent_hsqr_center_rfft_sign_injection',
        official_source_repository='thomas11809/SFWMark',
        official_source_commit='78666128b44614a0cc471993649e3132d5dddfcb',
        official_math_claim='official HSQR QR pattern and center RFFT sign injection applied to the supplied canonical Tree-Ring latent',
        not_claimed='NOT end-to-end official HSQR generation parity: this cohort supplies the canonical Tree-Ring clean latent and uses the Tree-Ring float32 DDIM configuration.',
)


def main() -> int:
    return main_for_spec(SPEC)


if __name__ == "__main__":
    raise SystemExit(main())
