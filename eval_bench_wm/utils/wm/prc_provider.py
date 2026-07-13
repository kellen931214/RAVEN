"""
Original by https://github.com/XuandongZhao/PRC-Watermark and heavily modified for debugging purposes and getting access to internals.
Please give them credit and adhere to their license agreement.
"""

import typing

import argparse
import numpy as np
import torch

from PIL import Image

from .wm_provider import WmProvider

from utils.image_utils import torch_to_PIL

## PRC
from scipy.sparse import csr_matrix
from scipy.special import binom, erf, lambertw
from scipy.linalg import orth
from ldpc import bp_decoder
import sys
import galois

## import package in the context of GS
# from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
# from cryptography.hazmat.backends import default_backend
# import numpy as np
# from scipy.stats import norm

# default fpr 1e-5
parser = argparse.ArgumentParser(add_help=False)

parser.add_argument('--fpr', type=float, default=0.00001, help="False positive rate")
parser.add_argument('--prc_t', type=int, default=3, help="PRC t parameter")
parser.add_argument("--prc_from_file", type=str, default="prc_4")
parser.add_argument("--prc_keys_path", type=str, default="keys")

class PRCProvider(WmProvider):
    """
    Original by https://github.com/XuandongZhao/PRC-Watermark and heavily modified for debugging and getting access to internals.
    """
    def __init__(self,
                 fpr: float = 1e-5,
                 prc_t: int = 3,
                 prc_from_file: str = "prc_4",
                 prc_keys_path: str = "keys",
                 **kwargs):
        super().__init__(**kwargs)

        import os
        import pickle

        # PRC parameters
        self.fpr = fpr
        self.t = prc_t
        # self.message_length = message_length

        # Compute codeword length n = C*H*W
        _, C, H, W = self.latent_shape
        self.n = C * H * W

        # Generate encoding + decoding keys
        self.GF = galois.GF(2)

        if prc_from_file:
            key_path = os.path.join(prc_keys_path, f"{prc_from_file}.pkl")
            with open(key_path, "rb") as f:
                self.encoding_key, self.decoding_key = pickle.load(f)
        else:
            self.encoding_key, self.decoding_key = self.__KeyGen()

        # # Pre-sample random payload messages for each batch element
        # self.message_bits_list = [
        #     np.random.randint(0, 2, size=self.message_length).tolist()
        #     for _ in range(self.batch_size)
        # ]

    def get_wm_type(self) -> str:
        return "PRC"

    def get_wm_latents(self, **kwargs) -> typing.Dict[str, any]:
        """
        Produce watermarked initial latents (zT) and return payload info.
        """

        
        self.prc_codeword = self.__Encode()
        self.wm_latents = self.__sample().reshape(self.latent_shape).to(self.device)

        #### Check below
        # Prepare message bits strings
        # message_bits_str_list = ["".join(str(b) for b in bits)
        #                          for bits in self.message_bits_list]

        return {
            "zT_torch": self.wm_latents,
            # "message_bits_str_list": message_bits_str_list
            # TODO: check after
        }

    #####PRC  begin by this line
    

    # def apply_channel_probs(x, channel_probs):
    #     e = GF(np.random.binomial(1, channel_probs))
    #     return x + e

    ### Given a GF(2) matrix, do row elimination and return the first k rows of A that form an invertible matrix
    def __boolean_row_reduce(self, A, print_progress=False):
        n, k = A.shape
        A_rr = A.copy()
        perm = np.arange(n)
        for j in range(k):
            idxs = j + np.nonzero(A_rr[j:, j])[0]
            if idxs.size == 0:
                print("The given matrix is not invertible")
                return None
            A_rr[[j, idxs[0]]] = A_rr[[idxs[0], j]]  # For matrices you have to swap them this way
            (perm[j], perm[idxs[0]]) = (perm[idxs[0]], perm[j])  # Weirdly, this is MUCH faster if you swap this way instead of using perm[[i,j]]=perm[[j,i]]
            A_rr[idxs[1:]] += A_rr[j]
            if print_progress and (j%5==0 or j+1==k):
                sys.stdout.write(f'\rDecoding progress: {j + 1} / {k}')
                sys.stdout.flush()
        if print_progress: print()
        return perm[:k]


    # def str_to_bin(string):
    #     bin_str = ''.join(format(i, '08b') for i in bytearray(string, encoding ='utf-8'))
    #     return [int(b) for b in bin_str]

    # def bin_to_str(bin_list):
    #     bin_str = ''.join(map(str, bin_list))
    #     byte_array = bytearray(int(bin_str[i:i+8], 2) for i in range(0, len(bin_str), 8) if bin_str[i:i+8]!='00000000')
    #     return byte_array.decode('utf-8')

    ### Key generation algorithm.
    ## Inputs:
    # n - block length (i.e., length of PRC codeword).
    # message_length - length of messages you want to encode
    # false_positive_rate - the false positive rate you're willing to tolerate
    # t - sparsity of parity checks. larger values help pseudorandomness
    # g - dimension of random code used. larger values help pseudorandomness
    # r - number of parity checks used. smaller values help pseudorandomness
    # noise_rate - amount of noise for __Encode to add to codewords. larger values help pseudorandomness
    def __KeyGen(self, message_length=512, g=None, r=None, noise_rate=None):
        # Set basic scheme parameters
        num_test_bits = int(np.ceil(np.log2(1 / self.fpr)))
        secpar = int(np.log2(binom(self.n, self.t)))
        if g is None: g = secpar
        # if noise_rate is None: noise_rate = np.exp(lambertw(-np.log(2) / secpar, -1)).real
        # if noise_rate is None: noise_rate = 1 - 2**(-(secpar - 3*np.log2(g))/g**2)
        if noise_rate is None: noise_rate = 1 - 2 ** (-secpar / g ** 2)
        k = message_length + g + num_test_bits
        if r is None: r = self.n - k - secpar

        # Sample n by k generator matrix (all but the first n-r of these will be over-written)
        generator_matrix = self.GF.Random((self.n, k))

        # Sample scipy.sparse parity-check matrix together with the last n-r rows of the generator matrix
        row_indices = []
        col_indices = []
        data = []
        for row in range(r):
            chosen_indices = np.random.choice(self.n - r + row, self.t - 1, replace=False)
            chosen_indices = np.append(chosen_indices, self.n - r + row)
            row_indices.extend([row] * self.t)
            col_indices.extend(chosen_indices)
            data.extend([1] * self.t)
            generator_matrix[self.n - r + row] = generator_matrix[chosen_indices[:-1]].sum(axis=0)
        parity_check_matrix = csr_matrix((data, (row_indices, col_indices)))

        # Compute scheme parameters
        max_bp_iter = int(np.log(self.n) / np.log(self.t))

        # Sample one-time pad and test bits
        one_time_pad = self.GF.Random(self.n)
        test_bits = self.GF.Random(num_test_bits)

        # Permute bits
        permutation = np.random.permutation(self.n)
        generator_matrix = generator_matrix[permutation]
        one_time_pad = one_time_pad[permutation]
        parity_check_matrix = parity_check_matrix[:, permutation]

        encoding_key = (generator_matrix, one_time_pad, test_bits, g, noise_rate)
        decoding_key = (generator_matrix, parity_check_matrix, one_time_pad, self.fpr, noise_rate, test_bits, g, max_bp_iter, self.t)

        return encoding_key, decoding_key


    ### Encoding algorithm
    ## Inputs:
    # encoding_key - Encoding key output by __KeyGen.
    # message - Message to encode, as an array of k bits. If none is provided a random message is used.
    def __Encode(self, message=None):
        generator_matrix, one_time_pad, test_bits, g, noise_rate = self.encoding_key
        n, k = generator_matrix.shape

        from .prc_message import MESSAGES
        self.message = MESSAGES

        if self.message is None:
            payload = np.concatenate((test_bits, self.GF.Random(k - len(test_bits))))
        else:
            assert len(self.message) <= k-len(test_bits)-g, "Message is too long"
            payload = np.concatenate((test_bits, self.GF.Random(g), self.GF(self.message), self.GF.Zeros(k-len(test_bits)-g-len(self.message))))

        error = self.GF(np.random.binomial(1, noise_rate, n))

        return 1 - 2 * torch.tensor(payload @ generator_matrix.T + one_time_pad + error, dtype=float)


    ### Detector
    ## Inputs:
    # decoding_key - Decoding key output by __KeyGen.
    # posteriors - The posterior expectations of sign(z) as a torch.tensor.
    ## Returns:
    # True/False - Detection result.
    def __Detect(self, posteriors, false_positive_rate=None):
        generator_matrix, parity_check_matrix, one_time_pad, false_positive_rate_key, noise_rate, test_bits, g, max_bp_iter, t = self.decoding_key
        if false_positive_rate is not None:
            fpr = false_positive_rate
        else:
            fpr = false_positive_rate_key

        posteriors = (1 - 2 * noise_rate) * (1 - 2 * np.array(one_time_pad, dtype=float)) * posteriors.numpy(force=True)

        r = parity_check_matrix.shape[0]
        Pi = np.prod(posteriors[parity_check_matrix.indices.reshape(r, t)], axis=1)
        log_plus = np.log((1 + Pi) / 2)
        log_minus = np.log((1 - Pi) / 2)
        log_prod = log_plus + log_minus

        const = 0.5 * np.sum(np.power(log_plus, 2) + np.power(log_minus, 2) - 0.5 * np.power(log_prod, 2))
        threshold = np.sqrt(2 * const * np.log(1 / fpr)) + 0.5 * log_prod.sum()

        print("Threshold:", threshold, "Value:", log_plus.sum())
        return log_plus.sum() >= threshold, log_plus.sum()


    ### Decoder
    ## Inputs:
    # decoding_key - Decoding key output by __KeyGen.
    # posteriors - The posterior expectations of sign(z) as a torch.tensor.
    ## Returns:
    # recovered_message - The recovered message. If the test bits are incorrect, outputs None.
    def __Decode(self, posteriors, print_progress=False, max_bp_iter=None):
        generator_matrix, parity_check_matrix, one_time_pad, false_positive_rate_key, noise_rate, test_bits, g, max_bp_iter_key, t = self.decoding_key
        if max_bp_iter is None:
            max_bp_iter = max_bp_iter_key
        # print(self.decoding_key)

        posteriors = (1 - 2 * noise_rate) * (1 - 2 * np.array(one_time_pad, dtype=float)) * posteriors.numpy(force=True)
        channel_probs = (1 - np.abs(posteriors)) / 2
        x_recovered = (1 - np.sign(posteriors)) // 2

        # Apply the belief-propagation decoder.
        if print_progress:
            print("Running belief propagation...")
        bpd = bp_decoder(parity_check_matrix, channel_probs=channel_probs, max_iter=max_bp_iter, bp_method="product_sum")
        x_decoded = bpd.decode(x_recovered)

        # Compute a confidence score.
        bpd_probs = 1 / (1 + np.exp(bpd.log_prob_ratios))
        confidences = 2 * np.abs(0.5 - bpd_probs)

        # Order codeword bits by confidence.
        confidence_order = np.argsort(-confidences)
        ordered_generator_matrix = generator_matrix[confidence_order]
        ordered_x_decoded = x_decoded[confidence_order]

        # Find the first (according to the confidence order) linearly independent set of rows of the generator matrix.
        top_invertible_rows = self.__boolean_row_reduce(ordered_generator_matrix, print_progress=print_progress)
        if top_invertible_rows is None:
            return None

        # Solve the system.
        if print_progress:
            print("Solving linear system...")
        recovered_string = np.linalg.solve(ordered_generator_matrix[top_invertible_rows], self.GF(ordered_x_decoded[top_invertible_rows]))

        if not (recovered_string[:len(test_bits)] == test_bits).all():
            return np.array(recovered_string[len(test_bits) + g:]), None
            # return None

        return np.array(recovered_string[len(test_bits) + g:]), True

    def __calculate_bit_accuracy(self,
                             original_message_binary,  # list or ndarray of 0/1
                             extracted_message_bits_str: str) -> float:
        """
        Calculate bit accuracy between extracted bits string and original bits array

        @param original_message_binary: original message as list or ndarray of bits (0/1)
        @param extracted_message_bits_str: extracted message as string of '0'/'1'

        @return: bit accuracy (float)
        """
        
        # 計算匹配的 bits 數量
        min_length = min(len(original_message_binary), len(extracted_message_bits_str))
        matching_bits = sum(1 for x, y in zip(original_message_binary, extracted_message_bits_str) if x == y)
        bit_accuracy = matching_bits / min_length
        
        return bit_accuracy


    def get_accuracies(self, latents: torch.Tensor) -> typing.Dict[str, any]:
        """
        Detect and decode messages from given latents.
        """
        var = 1.5
        reversed_prc = self.__recover_posteriors(latents.to(self.dtype).flatten().cpu(), variances=float(var)).flatten().cpu()

        detection_result, value = self.__Detect(reversed_prc)
        decode_result, check = self.__Decode(reversed_prc)

        # iterate and calulate bit accuracies
        bit_accuracies = []
        recovered_message_bits_str_list = []
        recovered_message_bits = decode_result
        
        decoding_result = check is not None
        orig_message_binary = self.message
        recovered_message_bits_str = recovered_message_bits 

        bit_accuracy = self.__calculate_bit_accuracy(orig_message_binary, recovered_message_bits_str)

        bit_accuracies.append(bit_accuracy)
        recovered_message_bits_str_list.append(recovered_message_bits_str)

        # decoding_result = (decode_result is not None)
        combined_result = detection_result or decoding_result

        # print(f'Detection: {detection_result}; Decoding: {decoding_result}; Combined: {combined_result}')
        log_message = f'Detection: {detection_result}; Decoding: {decoding_result}; Combined: {combined_result}'
        print(log_message)

        return {
            "value": value,
            "bit_accuracies": bit_accuracies,
            "message_bits_str_list": recovered_message_bits_str_list,
            "detection_success": combined_result,
            "log_message": log_message,
        }
    
    # pseudo gaussians 
    def __sample(self, basis=None):
        # pseudogaussian = codeword * torch.abs(torch.randn_like(codeword, dtype=torch.float64))
        codeword_np = self.prc_codeword.numpy()
        pseudogaussian_np = codeword_np * np.abs(np.random.randn(*codeword_np.shape))
        pseudogaussian = torch.from_numpy(pseudogaussian_np).to(dtype=self.dtype)
        if basis is None:
            return pseudogaussian
        return pseudogaussian @ basis.T


    def __recover_posteriors(self, z, basis=None, variances=None):
        if variances is None:
            default_variance = 1.5
            denominators = np.sqrt(2 * default_variance * (1+default_variance)) * torch.ones_like(z)
        elif type(variances) is float:
            denominators = np.sqrt(2 * variances * (1 + variances))
        else:
            denominators = torch.sqrt(2 * variances * (1 + variances))

        if basis is None:
            return erf(z / denominators)
        else:
            return erf((z @ basis) / denominators)

    def __random_basis(self, n):
        gaussian = torch.randn(n, n, dtype=torch.double)
        return orth(gaussian)
