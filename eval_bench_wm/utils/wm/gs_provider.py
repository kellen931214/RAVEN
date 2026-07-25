"""
Original by https://github.com/lthero-big/A-watermark-for-Diffusion-Models and heavily modified for debugging purposes and getting access to internals.
Please give them credit and adhere to their license agreement.
"""

import hashlib
import json
import typing

import argparse

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.backends import default_backend

import numpy as np
from scipy.special import betainc
from scipy.stats import norm

import torch

from PIL import Image

from .wm_provider import WmProvider

from utils.image_utils import torch_to_PIL



parser = argparse.ArgumentParser(add_help=False)

parser.add_argument('--l', default=1, type=int, help="The size of slide windows for m")
parser.add_argument('--num_replications', default=64, type=int, help="The number of replications of the message bits to get barcode image")
parser.add_argument('--message_width_in_bytes', default=32, type=int, help="Message width in bytes")
parser.add_argument(
    '--gs_protocol_mode',
    default='official_compatible',
    choices=('legacy', 'official_compatible'),
    help='Gaussian Shading implementation path. Default is official_compatible '
         '(bsmhmmlf/Gaussian-Shading); the legacy path is retained but must be '
         'requested explicitly with --gs_protocol_mode legacy.',
)
parser.add_argument(
    '--gs_detection_mode',
    default='official_onebit',
    choices=('official_onebit', 'official_traceability', 'legacy_default'),
    help='Detection-success threshold policy for Gaussian Shading. official_onebit '
         '(default) and official_traceability use the official beta-tail thresholds '
         '(tau_onebit / tau_bits) from official_thresholds() compared with ">="; '
         'legacy_default uses the legacy fixed GS threshold compared with ">" and '
         'must be requested explicitly. These are distinct threshold families and '
         'are never mixed with a clean-negative-calibrated 1%-FPR threshold.',
)
parser.add_argument('--gs_channel_copy', default=1, type=int)
parser.add_argument('--gs_hw_copy', default=8, type=int)
parser.add_argument('--gs_fpr', default=1e-6, type=float)
parser.add_argument('--gs_user_number', default=1000000, type=int)
parser.add_argument('--gs_sampling_seed', default=None, type=int)
parser.add_argument('--gs_secret_index', default=None, type=int)



class GsProvider(WmProvider):
    """
    Original by https://github.com/lthero-big/A-watermark-for-Diffusion-Models and heavily modified for debugging and getting access to internals.
    """

    def __init__(self,
                 message_width_in_bytes: int = 32,  # (channels/f_c) * f_h * f_w // 8
                 num_replications: int = 64,
                 l: int = 1,
                 offset: int = 0,
                 message: typing.Optional[str] = None,
                 key: str = None,
                 nonce: str = None,
                 gs_protocol_mode: str = "official_compatible",
                 gs_detection_mode: str = "official_onebit",
                 gs_channel_copy: int = 1,
                 gs_hw_copy: int = 8,
                 gs_fpr: float = 1e-6,
                 gs_user_number: int = 1000000,
                 gs_sampling_seed: typing.Optional[int] = None,
                 gs_secret_index: typing.Optional[int] = None,
                 **kwargs):
        """
        This provider uses a fixed list of keys, messages, and nonces for the watermarking process to eliminate every possibliy of false negatives due to wrong seeds.
        Generate more with "generate_secrets.py" as needed.

        These are the params for the the watermark:
        - message_width_in_bytes = Num users to distinguish
        - num_replications = strength of error correction
        - l = window size, aka the leeway given for errors in reconstructing original noise vector and it counteracts the error correction

        The "barcode" refers to the barcode image (zeros and ones, arranged in columns) that we have right before encrypting (during the watermark-generating stage) / right after decrypting (verifying a watermark stage).
        During verification, this gives a good vidual represenation of error introduced by the inversion process. See Section A in our Appendix, where we explain Gaussian Shading in detail.

        The barcode dimensions flattened divided by l must match the latent dimension flattened:
        
        <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        -> message_width_in_bytes * 8 * num_replications / l == num_channels * latent_resolution * latent_resolution
        <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        Example:
        -> 32 (message_width_in_bytes) * 8 * 64 (num_replication) / 1 (l) = 16384 = 4 (num_channels) * 64 (latent_resolution) * 64 (latent_resolution)

        @param message_width_in_bytes: width of the message in bytes
        @param num_replications: num_replications
        @param l: l
        @param offset: offset in the list of messages, keys, and nonces
        @param message: message to use all batch_size. This is useful if you want to simulate a user generating mutiple images.
        @param key: key to use all batch_size
        @param nonce: nonce to use for all batch_size
        """
        super().__init__(**kwargs)

        self.message_width_in_bytes = message_width_in_bytes
        self.message_width_in_bits = int(message_width_in_bytes * 8)
        self.barcode_width = self.message_width_in_bits

        self.num_replications = num_replications
        self.l = l
        self.offset = offset  # the exact messages, keys, nonces used when calling get_wm_latents are decided by batch_size starting from offset
        self.gs_protocol_mode = str(gs_protocol_mode)
        if self.gs_protocol_mode not in {"legacy", "official_compatible"}:
            raise ValueError(f"unsupported gs_protocol_mode: {self.gs_protocol_mode!r}")
        self.gs_detection_mode = str(gs_detection_mode)
        if self.gs_detection_mode not in {
            "official_onebit",
            "official_traceability",
            "legacy_default",
        }:
            raise ValueError(f"unsupported gs_detection_mode: {self.gs_detection_mode!r}")
        self.gs_channel_copy = int(gs_channel_copy)
        self.gs_hw_copy = int(gs_hw_copy)
        self.gs_fpr = float(gs_fpr)
        self.gs_user_number = int(gs_user_number)
        self.gs_sampling_seed = None if gs_sampling_seed is None else int(gs_sampling_seed)
        self.gs_secret_index = int(offset if gs_secret_index is None else gs_secret_index)

        if self.gs_protocol_mode == "legacy":
            assert self.message_width_in_bits * self.num_replications // self.l == self.num_channels * self.latent_resolution * self.latent_resolution
        else:
            if self.l != 1:
                raise ValueError("official_compatible Gaussian Shading requires l=1")
            if self.num_channels % self.gs_channel_copy:
                raise ValueError("latent channels must be divisible by gs_channel_copy")
            if self.latent_resolution % self.gs_hw_copy:
                raise ValueError("latent resolution must be divisible by gs_hw_copy")
            official_bits = (
                self.num_channels
                * self.latent_resolution
                * self.latent_resolution
                // (self.gs_channel_copy * self.gs_hw_copy * self.gs_hw_copy)
            )
            if official_bits != self.message_width_in_bits:
                raise ValueError(
                    "official payload size mismatch: "
                    f"layout requires {official_bits} bits but message_width_in_bytes="
                    f"{self.message_width_in_bytes} provides {self.message_width_in_bits}"
                )

        # use a predefined list of crypto params
        if message is None:
            from .messages import MESSAGES
            MESSAGES = [m[:self.message_width_in_bytes] for m in MESSAGES]  # trim to message_width_in_bytes to fit our required message length
            from .keys import KEYS
            from .nonces import NONCES
            self.messages = MESSAGES
            self.keys = KEYS
            self.nonces = NONCES

        # if only one param is given, just reuse it every time
        else:
            assert key is not None and nonce is not None
            self.messages = [message for _ in range(self.batch_size)]
            self.keys = [key for _ in range(self.batch_size)]
            self.nonces = [nonce for _ in range(self.batch_size)]

        if self.gs_protocol_mode == "official_compatible":
            for index in range(self.offset, self.offset + self.batch_size):
                message_bytes, key_bytes, nonce_bytes = self._secret_bytes(index)
                if len(message_bytes) != self.message_width_in_bytes:
                    raise ValueError("official message length does not match configured payload")
                if len(key_bytes) != 32:
                    raise ValueError("official-compatible ChaCha20 requires a 32-byte key")
                if len(nonce_bytes) != 12:
                    raise ValueError("official-compatible ChaCha20 requires a 12-byte nonce")


    def get_wm_type(self) -> str:
        return "GS"


    @staticmethod
    def _sha256_bytes(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()


    def _secret_bytes(self, index: int) -> typing.Tuple[bytes, bytes, bytes]:
        message = self.messages[index]
        key = self.keys[index]
        nonce = self.nonces[index]
        if isinstance(message, str):
            message = self.__character_str_to_bytes(message)
        if isinstance(key, str):
            key = bytes.fromhex(key)
        if isinstance(nonce, str):
            nonce = bytes.fromhex(nonce)
        message = bytes(message[:self.message_width_in_bytes])
        key = bytes(key)
        nonce = bytes(nonce[:12] if self.gs_protocol_mode == "official_compatible" else nonce)
        return message, key, nonce


    def secret_provenance(self, index: typing.Optional[int] = None) -> typing.Dict[str, typing.Any]:
        """Return auditable identifiers without exposing raw secret material."""
        secret_index = self.gs_secret_index if index is None else int(index)
        message, key, nonce = self._secret_bytes(secret_index)
        hashes = {
            "message_sha256": self._sha256_bytes(message),
            "key_sha256": self._sha256_bytes(key),
            "nonce_sha256": self._sha256_bytes(nonce),
        }
        combined = hashlib.sha256(
            json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "secret_index": secret_index,
            **hashes,
            "secret_bundle_sha256": combined,
        }


    def _official_payload(self, message_bytes: bytes) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Match official `watermark.repeat(1, ch, hw, hw)` layout exactly."""
        payload_bits = np.unpackbits(np.frombuffer(message_bytes, dtype=np.uint8))
        payload = torch.from_numpy(payload_bits.copy()).reshape(
            1,
            self.num_channels // self.gs_channel_copy,
            self.latent_resolution // self.gs_hw_copy,
            self.latent_resolution // self.gs_hw_copy,
        ).to(torch.uint8)
        diffused = payload.repeat(
            1, self.gs_channel_copy, self.gs_hw_copy, self.gs_hw_copy
        )
        return payload, diffused


    def watermark_target_tensor(self, index: typing.Optional[int] = None) -> torch.Tensor:
        secret_index = self.gs_secret_index if index is None else int(index)
        message, _, _ = self._secret_bytes(secret_index)
        if self.gs_protocol_mode == "official_compatible":
            payload, _ = self._official_payload(message)
            return payload.to(device=self.device)
        bits = np.unpackbits(np.frombuffer(message, dtype=np.uint8))
        return torch.from_numpy(bits.copy()).reshape(1, -1).to(
            device=self.device, dtype=torch.uint8
        )


    def _official_encrypt_bits(self, diffused: torch.Tensor, key: bytes, nonce: bytes) -> np.ndarray:
        from Crypto.Cipher import ChaCha20

        cipher = ChaCha20.new(key=key, nonce=nonce)
        encrypted = cipher.encrypt(np.packbits(diffused.flatten().cpu().numpy()).tobytes())
        return np.unpackbits(np.frombuffer(encrypted, dtype=np.uint8))


    def _official_latent_from_bits(self, bits: np.ndarray, uniforms: np.ndarray) -> torch.Tensor:
        # Official truncnorm sampling for one bit is mathematically
        # norm.ppf((u + bit) / 2), with u ~ Uniform[0, 1).
        sampled = norm.ppf((uniforms + bits.astype(np.float64)) / 2.0)
        return torch.tensor(
            sampled.reshape(self.latent_shape[1:]), dtype=self.dtype, device=self.device
        )


    def _get_official_wm_latents(self) -> typing.Dict[str, typing.Any]:
        latents = []
        clean_latents = []
        targets = []
        target_strings = []
        uniform_hashes = []
        secret_records = []
        for batch_index, secret_index in enumerate(range(self.offset, self.offset + self.batch_size)):
            message, key, nonce = self._secret_bytes(secret_index)
            payload, diffused = self._official_payload(message)
            encrypted_bits = self._official_encrypt_bits(diffused, key, nonce)
            sampling_seed = None if self.gs_sampling_seed is None else self.gs_sampling_seed + batch_index
            if sampling_seed is None:
                uniforms = np.random.uniform(0.0, 1.0, size=encrypted_bits.size)
            else:
                uniforms = np.random.default_rng(sampling_seed).uniform(
                    0.0, 1.0, size=encrypted_bits.size
                )
            latents.append(self._official_latent_from_bits(encrypted_bits, uniforms))
            clean_latents.append(
                torch.tensor(
                    norm.ppf(uniforms).reshape(self.latent_shape[1:]),
                    dtype=self.dtype,
                    device=self.device,
                )
            )
            targets.append(payload.squeeze(0).to(device=self.device))
            target_strings.append("".join(f"{byte:08b}" for byte in message))
            uniform_hashes.append(self._sha256_bytes(uniforms.astype(np.float64).tobytes()))
            secret_records.append(self.secret_provenance(secret_index))
        latents_torch = torch.stack(latents, dim=0)
        clean_latents_torch = torch.stack(clean_latents, dim=0)
        targets_torch = torch.stack(targets, dim=0)
        return {
            "zT_torch": latents_torch,
            "zT_clean_torch": clean_latents_torch,
            "barcodes_torch": targets_torch,
            "message_bits_str_list": target_strings,
            "sampling_uniform_sha256_list": uniform_hashes,
            "secret_provenance_list": secret_records,
            "gs_protocol_mode": self.gs_protocol_mode,
        }


    def official_thresholds(self) -> typing.Dict[str, typing.Any]:
        """Reproduce the official beta-tail detection/traceability thresholds."""
        mark_length = self.message_width_in_bits
        tau_onebit = None
        tau_bits = None
        for errors in range(mark_length):
            single_user_fpr = float(betainc(errors + 1, mark_length - errors, 0.5))
            if tau_onebit is None and single_user_fpr <= self.gs_fpr:
                tau_onebit = errors / mark_length
            if tau_bits is None and single_user_fpr * self.gs_user_number <= self.gs_fpr:
                tau_bits = errors / mark_length
        return {
            "fpr": self.gs_fpr,
            "user_number": self.gs_user_number,
            "tau_onebit": tau_onebit,
            "tau_bits": tau_bits,
            "comparison_operator": ">=",
            "source": "bsmhmmlf/Gaussian-Shading watermark.py@09c678f",
        }


    def active_detection_threshold(self) -> typing.Dict[str, typing.Any]:
        """Resolve the detection-success threshold actually used, honoring
        ``gs_detection_mode``.

        Three distinct threshold families are kept explicitly separate:

        - ``official_onebit`` / ``official_traceability`` -> official beta-tail
          thresholds ``tau_onebit`` / ``tau_bits`` from :meth:`official_thresholds`,
          compared with ``>=`` (matches bsmhmmlf/Gaussian-Shading).
        - ``legacy_default`` -> the legacy fixed GS threshold, compared with ``>``
          (retained only for explicit legacy/debug parity).

        Neither is a clean-negative-calibrated 1%-FPR threshold; callers must not
        relabel the resulting detection rate as TPR@1%FPR.
        """
        official = self.official_thresholds()
        if self.gs_detection_mode == "legacy_default":
            from utils.utils import GS_THRESHOLDS, LEGACY_THRESHOLD_NOMINAL_FPR

            return {
                "detection_mode": "legacy_default",
                "threshold": float(GS_THRESHOLDS),
                "threshold_type": "legacy_default_threshold",
                "comparison_operator": ">",
                "nominal_fpr": LEGACY_THRESHOLD_NOMINAL_FPR.get("GS"),
                "calibrated_from_current_clean_negatives": False,
                "official_tau_onebit": official["tau_onebit"],
                "official_tau_bits": official["tau_bits"],
            }
        if self.gs_detection_mode == "official_traceability":
            threshold = official["tau_bits"]
            threshold_type = "official_beta_tail_tau_bits"
        else:
            threshold = official["tau_onebit"]
            threshold_type = "official_beta_tail_tau_onebit"
        return {
            "detection_mode": self.gs_detection_mode,
            "threshold": None if threshold is None else float(threshold),
            "threshold_type": threshold_type,
            "comparison_operator": ">=",
            "nominal_fpr": official["fpr"],
            "calibrated_from_current_clean_negatives": False,
            "official_tau_onebit": official["tau_onebit"],
            "official_tau_bits": official["tau_bits"],
        }


    def is_detection_successful(self, value: float) -> bool:
        """Detection-success decision for GS honoring ``gs_detection_mode``.

        Official modes compare bit-accuracy with ``>=`` against the beta-tail
        threshold; legacy_default reproduces the legacy strict ``>`` comparison.
        """
        info = self.active_detection_threshold()
        threshold = info["threshold"]
        if threshold is None:
            return False
        if info["comparison_operator"] == ">=":
            return float(value) >= float(threshold)
        return float(value) > float(threshold)


    def __character_str_to_bytes(self,
                                 message: str) -> bytes:
        """
        Convert a string to a byte string

        @param message: message to convert
        @param padded_message_length: length of the byte string

        @return: byte string
        """
        # Convert the message to a byte string
        message_bytes = message.encode()

        # Ensure the encoded message is 256 bits (32 bytes)
        if len(message_bytes) < self.message_width_in_bytes:
            padded_message = message_bytes + b'\x00' * (self.message_width_in_bytes - len(message_bytes))
        else:
            padded_message = message_bytes[:self.message_width_in_bytes]
        return padded_message

    def get_wm_latents(self, **kwargs) -> typing.Dict[str, any]:
        """
        Get Watermarked latents and barcodes
                        
        @return: dict
        """
        if self.gs_protocol_mode == "official_compatible":
            return self._get_official_wm_latents()

        # Legacy implementation intentionally retained byte-for-byte in behavior.
        # iterate message, keys, nonces
        latents_torch = []
        barcodes_torch = []
        message_bits_str_list = []
        for i in range(self.offset,
                       self.batch_size + self.offset):
            # get the message, key and nonce
            message_bytes = self.messages[i]  # message_bytes = k
            # remember the message bits as string
            message_bits_str_list.append(''.join(format(byte, '08b') for byte in self.messages[i]))

            key_bytes = self.keys[i]
            nonce_bytes = self.nonces[i]

            # ----------------------------------------------------- STEP 1: Replicate the message to get a barcode -----------------------------------------------------
            replicated_message_bytes = message_bytes * self.num_replications  # replicated_message_bytes = s_d, 2048 bytes in old setup
            replicated_message_bits_str = ''.join(format(byte, '08b') for byte in replicated_message_bytes)  # replicated_message_bits_flat = s_d in binary, 16384 bits in old setup
            # Make barcode as np from s_d
            barcode_ints_2d_np = np.array([int(b, 2) for b in replicated_message_bits_str]).reshape(self.num_replications, self.message_width_in_bits)
            barcode_ints_2d_torch = torch.tensor(barcode_ints_2d_np, dtype=torch.uint8, device=self.device)
            barcodes_torch.append(barcode_ints_2d_torch)

            # ------------------------------------------------------------- STEP 2: Encrypt the barcode -------------------------------------------------------------
            # set up cipher
            cipher = Cipher(algorithms.ChaCha20(key_bytes, nonce_bytes), mode=None, backend=default_backend())
            encryptor = cipher.encryptor()
            encrypted_bytes = encryptor.update(replicated_message_bytes) + encryptor.finalize()  # encrypted_bytes = m
            encrypted_bits_str = ''.join(format(byte, '08b') for byte in encrypted_bytes)  # encrypted_bits_str = m_bits
    
            # ------------------------------------- STEP 3: Embed the encrypted message into the latent space via Gaussian sampling -------------------------------------
            # sampling from uniform into gaussian
            # Traverse the binary representation of m, cutting according to window size l
            pixel_values = []
            for i in range(0, len(encrypted_bits_str), self.l):
                window = encrypted_bits_str[i:i + self.l]
                y = int(window, 2)  # Convert the binary sequence (base 2) inside the window into integer y
                u = np.random.uniform(0, 1)  # Generate random u
                # Calculate z^s_T
                pixel_value = norm.ppf((u + y) / 2**self.l)
                pixel_values.append(pixel_value)
            pixel_values = torch.tensor(pixel_values, dtype=self.dtype, device=self.device)
            latent_torch = pixel_values.reshape(self.latent_shape[1:])  # latent shape without batch dim
            latents_torch.append(latent_torch)
        
        # finalize
        latents_torch = torch.stack(latents_torch, dim=0)
        barcodes_torch = torch.stack(barcodes_torch, axis=0)

        latents_PIL = torch_to_PIL(latents_torch)

        barcodes_PIL = torch_to_PIL(barcodes_torch)

        results_dict = {"zT_torch": latents_torch,
                        "zT_PIL": latents_PIL,
                        "zT": latents_PIL,
                        "barcodes_torch": barcodes_torch,
                        "barcodes_PIL": barcodes_PIL,
                        "barcodes": barcodes_PIL,
                        "message_bits_str_list": message_bits_str_list
                        }
    
        return results_dict
    

    def __recover_messages_from_latents(self,
                                        latents: torch.Tensor) -> typing.List[typing.Union[str, Image.Image]]:
        """
        Recover messages from latents

        @param latents: latent either tensor with batch dim
        @return: dict
        """
        # lets work on numpies only
        latents = latents.cpu().numpy()

        # iterate keys, nonces
        recovered_messages_bits_str = []
        recovered_barcodes_torch = []
        for i in range(self.offset,
                       self.batch_size + self.offset):
            # get the message, key and nonce
            key_bytes = self.keys[i]
            nonce_bytes = self.nonces[i]
            latent = latents[i - self.offset]  # This is ugly
        
            # ------------------------- STEP 3 reverse: Extract the encrypted message from the latent space via reverse Gaussian Sampling -------------------------
            # Reconstruct m from reversed_latents
            encrypted_numbers = []
            for zsT_pixel_value in np.nditer(latent):
                # Use the inverse operation of norm.ppf to recover the original y value
                y_reconstructed = norm.cdf(zsT_pixel_value) * 2**self.l
                y_reconstructed = int(y_reconstructed)
                # in some very rare and sporadic cases, y_reconstructed can be a 2 for l=1.
                if y_reconstructed == 2**self.l:
                    y_reconstructed -= 1
                encrypted_numbers.append(y_reconstructed)
    
            # For l > 1 we need another step that unfold number up to 2**l -1 to binary strings
            encrypted_bits = []
            for n in encrypted_numbers:
                # Convert the integer to binary and pad with zeros
                encrypted_bits.append(bin(n)[2:].zfill(self.l))
            encrypted_bits_str = ''.join(encrypted_bits)  # encrypted_bits_str = reconstructed_m_bits

            # Convert binary bits to bytes
            # This is hell for my smooth brain
            encrypted_bytes = bytes(
            int(
                    ''.join(
                        str(bit) for bit in encrypted_bits_str[j:j+8]
                        ),
                        2
                    ) for j in range(0, len(encrypted_bits_str), 8)
            )
    
            # ---------------------------------------- STEP 2 reverse: Decrypt the message to retrieve barcode ----------------------------------------
            # Decrypt m to recover the data s_d before diffusion
            # initiate the Cipher
            cipher = Cipher(algorithms.ChaCha20(key_bytes, nonce_bytes), mode=None, backend=default_backend())
            decryptor = cipher.decryptor()
            replicated_message_bytes = decryptor.update(encrypted_bytes) + decryptor.finalize()  # replicated_message_bytes = s_d_reconstructed
            
            # Convert the decrypted byte string into a binary representation
            replicated_message_bits_str = ['{:08b}'.format(byte) for byte in replicated_message_bytes]  # replicated_message_bits_str = bits_list
            # Merge the binary strings into a long string
            all_bits = ''.join(replicated_message_bits_str)
            # list of strings with len self.message_width_in_bits
            barcode_string_1d = [all_bits[i:i+self.message_width_in_bits] for i in range(0,
                                                                                         len(all_bits),
                                                                                         self.message_width_in_bits)]  # barcode_string_2d = segments
        
            # make torch
            # 1st dim: number of elements in barcode_string_1d
            # 2nd dim: number of character ("0" or "1") in each element, these are self.message_width_in_bits
            barcode_int_2d_torch = torch.tensor([list(map(int, list(row))) for row in barcode_string_1d], dtype=torch.uint8, device=self.device)
    
            # ---------------------------------------- STEP 1 reverse: Recover the original message from the barcode -----------------------------------
            # Voting mechanism to determine each bit
            message_bits_str = ''
            for i in range(self.message_width_in_bits):
                # Calculate the count of '1's for each bit across all lines
                count_1 = sum(column[i] == '1' for column in barcode_string_1d)
                # for l>1, we might get sporadic errors becasue of numerical
                message_bits_str += '1' if count_1 > len(barcode_string_1d) / 2 else '0'

            # append
            recovered_messages_bits_str.append(message_bits_str)
            recovered_barcodes_torch.append(barcode_int_2d_torch)

        # finalize
        recovered_barcodes_torch = torch.stack(recovered_barcodes_torch, dim=0)
        recovered_barcodes_PIL = torch_to_PIL(recovered_barcodes_torch)

        return {"messages_bits_str": recovered_messages_bits_str,
                "barcodes_torch": recovered_barcodes_torch,
                "barcodes_PIL": recovered_barcodes_PIL}


    def _official_decrypt_diffused(
        self, reversed_latent: torch.Tensor, key: bytes, nonce: bytes
    ) -> torch.Tensor:
        from Crypto.Cipher import ChaCha20

        encrypted_bits = (reversed_latent > 0).to(torch.uint8).flatten().cpu().numpy()
        cipher = ChaCha20.new(key=key, nonce=nonce)
        decrypted = cipher.decrypt(np.packbits(encrypted_bits).tobytes())
        bits = np.unpackbits(np.frombuffer(decrypted, dtype=np.uint8))
        return torch.from_numpy(bits.copy()).reshape(
            1, self.num_channels, self.latent_resolution, self.latent_resolution
        ).to(device=self.device, dtype=torch.uint8)


    def _official_majority_vote(self, diffused: torch.Tensor) -> torch.Tensor:
        ch_stride = self.num_channels // self.gs_channel_copy
        hw_stride = self.latent_resolution // self.gs_hw_copy
        split_ch = torch.cat(
            torch.split(diffused, tuple([ch_stride] * self.gs_channel_copy), dim=1),
            dim=0,
        )
        split_h = torch.cat(
            torch.split(split_ch, tuple([hw_stride] * self.gs_hw_copy), dim=2),
            dim=0,
        )
        split_w = torch.cat(
            torch.split(split_h, tuple([hw_stride] * self.gs_hw_copy), dim=3),
            dim=0,
        )
        votes = torch.sum(split_w, dim=0)
        threshold = self.gs_channel_copy * self.gs_hw_copy * self.gs_hw_copy // 2
        # Match official code: ties (vote <= threshold) decode to zero.
        return (votes > threshold).to(torch.uint8)


    def _get_official_accuracies(self, latents: torch.Tensor) -> typing.Dict[str, typing.Any]:
        bit_accuracies = []
        recovered_strings = []
        recovered_targets = []
        for batch_index, secret_index in enumerate(range(self.offset, self.offset + self.batch_size)):
            message, key, nonce = self._secret_bytes(secret_index)
            diffused = self._official_decrypt_diffused(latents[batch_index], key, nonce)
            recovered = self._official_majority_vote(diffused)
            target, _ = self._official_payload(message)
            target = target.squeeze(0).to(device=self.device)
            bit_accuracies.append(float((recovered == target).float().mean().item()))
            recovered_strings.append("".join(str(int(bit)) for bit in recovered.flatten().cpu()))
            recovered_targets.append(recovered)
        return {
            "accuracies": bit_accuracies,
            "bit_accuracies": bit_accuracies,
            "barcodes_torch": torch.stack(recovered_targets, dim=0),
            "message_bits_str_list": recovered_strings,
            "gs_protocol_mode": self.gs_protocol_mode,
        }


    def __calculate_bit_accuracy(self,
                                 original_message_hex: any,  # no idea what datatype
                                 extracted_message_bits_str: str) -> float:
        """
        Gett bit accuracy between extracted bits and the original message hex

        @param original_message_hex: original message in hex
        @param extracted_message_bits_str: extracted message in bits

        @return: bit accuracy
        """
        # Convert the original hex message to binary
        original_message_bin = ''.join(format(byte, '08b') for byte in original_message_hex)

        # Ensure both binary strings are of the same length
        min_length = min(len(original_message_bin), len(extracted_message_bits_str))
        original_message_bin = original_message_bin[:min_length]
        extracted_message_bits_str = extracted_message_bits_str[:min_length]
        
        # Calculate bit accuracy
        matching_bits = sum(1 for x, y in zip(original_message_bin, extracted_message_bits_str) if x == y)
        bit_accuracy = matching_bits / min_length
        
        return bit_accuracy
    

    def get_accuracies(self, latents: typing.Union[torch.Tensor, np.array]) -> typing.Dict[str, any]:
        """
        Get bit accuracy between original and extracted messages

        @param latents: latent either tensor with batch dim or numpy with batch dim
        @return: dict
        """
        if self.gs_protocol_mode == "official_compatible":
            if isinstance(latents, np.ndarray):
                latents = torch.from_numpy(latents).to(device=self.device, dtype=self.dtype)
            return self._get_official_accuracies(latents)

        # get the extracted message
        recovered = self.__recover_messages_from_latents(latents)
        recovered_messages_bits_str = recovered["messages_bits_str"]
        recovered_barcodes_torch = recovered["barcodes_torch"]
        recovered_barcodes_PIL = recovered["barcodes_PIL"]

        # iterate and calulate bit accuracies
        bit_accuracies = []
        recovered_message_bits_str_list = []
        for i in range(self.offset, self.batch_size + self.offset):
            original_message_hex = self.messages[i]
            recovered_message_bits_str = recovered_messages_bits_str[i - self.offset]

            # get the bit accuracy
            bit_accuracy = self.__calculate_bit_accuracy(original_message_hex, recovered_message_bits_str)
            bit_accuracies.append(bit_accuracy)
            recovered_message_bits_str_list.append(recovered_message_bits_str)

        return {
            "accuracies": bit_accuracies,
            "bit_accuracies": bit_accuracies,
            "barcodes_torch": recovered_barcodes_torch,
            "barcodes_PIL": recovered_barcodes_PIL,
            "barcodes": recovered_barcodes_PIL,
            "message_bits_str_list": recovered_message_bits_str_list
        }
    

    def wiggle_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Resample latents

        @param latents: latent tensor with batch dim
        @return: torch.Tensor with batch dim 
        """

        # reverse sampling back to barcode pixels in [0, 2**self.l - 1]
        latents = latents.detach().cpu().numpy()
        latents = norm.cdf(latents) * 2**self.l
        latents = latents.astype(np.int32)
        # fix bug where we sometimes get 2**l
        latents[latents == 2**self.l] = 2**self.l - 1
        # latents is now integers in [0, 2**self.l - 1]

        # forward sampling with randomnes
        # y we already have
        y = latents
        # u we draw
        u = np.random.uniform(low=0, high=1, size=y.shape).astype(np.float32)
        # sampling a gaussian
        new_latent = norm.ppf((u + y) / 2**self.l)

        return torch.tensor(new_latent, dtype=self.dtype, device=self.device)


# Official upstream generation settings from bsmhmmlf/Gaussian-Shading
# run_gaussian_shading.py (stabilityai SD2.1-base, DPMSolverMultistepScheduler, fp16).
OFFICIAL_REPRODUCTION_MODEL_ID = "stabilityai/stable-diffusion-2-1-base"
OFFICIAL_REPRODUCTION_SCHEDULER = "DPM"  # DPMSolverMultistepScheduler in pipe_utils.SCHEDULER_CLASSES
OFFICIAL_REPRODUCTION_REVISION = "fp16"


def apply_official_reproduction_defaults(args, argv=None) -> typing.Dict[str, typing.Any]:
    """Inject official upstream Gaussian Shading *generation* defaults for the
    standalone reproduction runners (``run_watermark.py`` / ``run_removal.py``) ONLY.

    Sets the official model / scheduler / fp16 weight revision unless the user
    passed them explicitly on the command line. This is intentionally NOT wired
    into the formal generator (``experiments/generate_watermarked_images.py``),
    which keeps the shared matched-cohort model/scheduler (RedbeardNZ + DDIM) so
    GS stays comparable to the other watermark methods. Other watermark methods
    are never touched (guarded by ``wm_type == "GS"``).

    Note: the fork's global compute dtype is ``torch.float32`` (see
    ``pipe_provider.DTYPE``); ``revision="fp16"`` selects the fp16 *weight*
    variant only. Full fp16 compute parity with upstream is a documented residual
    difference, not silently changed here.
    """
    import sys as _sys

    argv = _sys.argv if argv is None else argv

    def _explicit(flag: str) -> bool:
        return any(a == flag or a.startswith(flag + "=") for a in argv)

    applied: typing.Dict[str, typing.Any] = {}
    if getattr(args, "wm_type", None) != "GS":
        return applied
    if hasattr(args, "modelid_target") and not _explicit("--modelid_target"):
        args.modelid_target = OFFICIAL_REPRODUCTION_MODEL_ID
        applied["modelid_target"] = args.modelid_target
    if hasattr(args, "scheduler_target") and not _explicit("--scheduler_target"):
        args.scheduler_target = OFFICIAL_REPRODUCTION_SCHEDULER
        applied["scheduler_target"] = args.scheduler_target
    if not _explicit("--model_revision") and getattr(args, "model_revision", None) in (None, ""):
        args.model_revision = OFFICIAL_REPRODUCTION_REVISION
        applied["model_revision"] = args.model_revision
    applied["dtype_note"] = (
        "compute dtype follows the fork global (torch.float32); official upstream "
        "uses fp16 — revision='fp16' selects the fp16 weight variant only"
    )
    applied["inversion_note"] = (
        "official-compatible inversion via invert_z0 (empty prompt, guidance_scale=1, "
        "DPM inverse-scheduler timesteps); not byte-identical to upstream "
        "inverse_stable_diffusion.py"
    )
    return applied
