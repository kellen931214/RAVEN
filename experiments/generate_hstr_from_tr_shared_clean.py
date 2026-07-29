#!/usr/bin/env python3
"""Generate HSTR watermarked images from canonical TR clean artifacts."""

from __future__ import annotations

from shared_tr_clean_fourier import MethodSpec, main_for_spec
from utils.wm.hstr_provider import HSTR_SHARED_TR_CLEAN_MODE

SPEC = MethodSpec(
        method='HSTR',
        wm_name='HSTR',
        protocol='hstr_shared_tr_clean_v2',
        protocol_mode=HSTR_SHARED_TR_CLEAN_MODE,
        provider_module='hstr_provider',
        provider_class='HSTRProvider',
        provider_entrypoint_field='hstr_provider_entrypoint_sha256',
        bundle_arg='hstr_bundle_dir',
        create_bundle_arg='hstr_create_bundle',
        shared_profile='raven_shared_tr_clean_v2_not_official_hstr_generation',
        pairing_relation='shared_tr_clean_latent_hstr_center_fourier_replacement',
        official_source_repository='thomas11809/SFWMark',
        official_source_commit='78666128b44614a0cc471993649e3132d5dddfcb',
        official_math_claim='official HSTR SFWMark center Fourier pattern injection applied to the supplied canonical Tree-Ring latent',
        not_claimed='NOT end-to-end official HSTR generation parity: this cohort supplies the canonical Tree-Ring clean latent and uses the Tree-Ring float32 DDIM configuration.',
)


def main() -> int:
    return main_for_spec(SPEC)


if __name__ == "__main__":
    raise SystemExit(main())
