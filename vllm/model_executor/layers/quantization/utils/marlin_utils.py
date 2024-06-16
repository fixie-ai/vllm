"""This file is used for /tests and /benchmarks"""
<<<<<<< HEAD
import numpy
import torch

from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQ_MARLIN_MAX_PARALLEL, GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_TILE)
=======
import random

import numpy
import torch

from vllm.model_executor.layers.quantization.utils.format_24 import (
    mask_creator, sparse_semi_structured_from_dense_cutlass)
from vllm.model_executor.layers.quantization.utils.marlin_24_perms import (
    marlin_24_perm, marlin_24_scale_perm, marlin_24_scale_perm_single)
from vllm.model_executor.layers.quantization.utils.marlin_perms import (
    marlin_perm, marlin_scale_perm, marlin_scale_perm_single)
>>>>>>> fixie-ai/vllm/main
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_pack_factor, quantize_weights, sort_weights)

__cuda_arch = torch.cuda.get_device_capability()

<<<<<<< HEAD
=======
MARLIN_TILE = 16

>>>>>>> fixie-ai/vllm/main

def is_marlin_supported():
    return __cuda_arch[0] >= 8


<<<<<<< HEAD
# Precompute permutations for Marlin weight and scale shuffling # noqa: E501
#
# Marlin works on [16,64] tiles. The goal of the permutations is to reorder the weight data so that it is compatible noqa: # noqa: E501
# with the tensor-core format that is described here:
# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type # noqa: E501
#
# As a result of this reordering, the vector loads inside the kernel will get the data as it is needed for tensor-core # noqa: E501
# (without the need to use ldmatrix instructions) # noqa: E501
def _get_perms(num_bits):
    perm_list = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                    2 * (i % 4),
                    2 * (i % 4) + 1,
                    2 * (i % 4 + 4),
                    2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend([p + 256 * j for p in perm1])

    perm = numpy.array(perm_list)

    if num_bits == 4:
        interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = numpy.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single


_perm = {}
_scale_perm = {}
_scale_perm_single = {}
for num_bits in [4, 8]:
    perm, scale_perm, scale_perm_single = _get_perms(num_bits)
    _perm[num_bits] = perm
    _scale_perm[num_bits] = scale_perm
    _scale_perm_single[num_bits] = scale_perm_single


def marlin_permute_weights(q_w,
                           size_k,
                           size_n,
                           num_bits,
                           tile=GPTQ_MARLIN_TILE):
=======
def marlin_permute_weights(q_w, size_k, size_n, perm, tile=MARLIN_TILE):
>>>>>>> fixie-ai/vllm/main
    assert q_w.shape == (size_k, size_n)
    assert size_k % tile == 0, f"size_k = {size_k}, tile = {tile}"
    assert size_n % tile == 0, f"size_k = {size_n}, tile = {tile}"

    # Permute weights to 16x64 marlin tiles
    q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // tile, size_n * tile))

<<<<<<< HEAD
    q_w = q_w.reshape(
        (-1, _perm[num_bits].numel()))[:, _perm[num_bits]].reshape(q_w.shape)
=======
    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)
>>>>>>> fixie-ai/vllm/main

    return q_w


<<<<<<< HEAD
def marlin_weights(q_w, size_k, size_n, num_bits):
    # Permute
    q_w = marlin_permute_weights(q_w, size_k, size_n, num_bits)
=======
def marlin_weights(q_w, size_k, size_n, num_bits, perm):
    # Permute
    q_w = marlin_permute_weights(q_w, size_k, size_n, perm)
>>>>>>> fixie-ai/vllm/main

    # Pack
    pack_factor = get_pack_factor(num_bits)
    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(numpy.uint32)

    q_packed = numpy.zeros((q_w.shape[0], q_w.shape[1] // pack_factor),
                           dtype=numpy.uint32)
<<<<<<< HEAD

=======
>>>>>>> fixie-ai/vllm/main
    for i in range(pack_factor):
        q_packed |= q_w[:, i::pack_factor] << num_bits * i

    q_packed = torch.from_numpy(q_packed.astype(numpy.int32)).to(orig_device)

    return q_packed


<<<<<<< HEAD
def marlin_permute_scales(s, size_k, size_n, group_size, num_bits):
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(_scale_perm[num_bits])))[:,
                                                        _scale_perm[num_bits]]
    else:
        s = s.reshape(
            (-1,
             len(_scale_perm_single[num_bits])))[:,
                                                 _scale_perm_single[num_bits]]
=======
def marlin_permute_scales(s, size_k, size_n, group_size, scale_perm,
                          scale_perm_single):
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
>>>>>>> fixie-ai/vllm/main
    s = s.reshape((-1, size_n)).contiguous()

    return s


def marlin_quantize(
    w: torch.Tensor,
    num_bits: int,
    group_size: int,
    act_order: bool,
):
    size_k, size_n = w.shape

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    # Quantize (and apply act_order if provided)
    w_ref, q_w, s, g_idx, rand_perm = quantize_weights(w, num_bits, group_size,
                                                       act_order)

    # For act_order, sort the "weights" and "g_idx" so that group ids are
    # increasing
    sort_indices = torch.empty(0, dtype=torch.int, device=w.device)
    if act_order:
        q_w, g_idx, sort_indices = sort_weights(q_w, g_idx)

    # Reformat to marlin
<<<<<<< HEAD
    marlin_q_w = marlin_weights(q_w, size_k, size_n, num_bits)
    marlin_s = marlin_permute_scales(s, size_k, size_n, group_size, num_bits)
=======
    marlin_q_w = marlin_weights(q_w, size_k, size_n, num_bits,
                                marlin_perm[num_bits])
    marlin_s = marlin_permute_scales(s, size_k, size_n, group_size,
                                     marlin_scale_perm[num_bits],
                                     marlin_scale_perm_single[num_bits])
>>>>>>> fixie-ai/vllm/main

    # Create result
    res_list = [w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, rand_perm]
    for i in range(len(res_list)):
        res_list[i] = res_list[i].to(w.device)

    return res_list


<<<<<<< HEAD
class MarlinWorkspace:

    def __init__(self, out_features):
        assert (out_features % GPTQ_MARLIN_MIN_THREAD_N == 0), (
            "out_features = {} is undivisible by GPTQ_MARLIN_MIN_THREAD_N = {}"
            .format(out_features, GPTQ_MARLIN_MIN_THREAD_N))

        max_workspace_size = ((out_features // GPTQ_MARLIN_MIN_THREAD_N) *
                              GPTQ_MARLIN_MAX_PARALLEL)
=======
def inject_24(w, size_k, size_n):
    assert w.shape == (size_k, size_n)

    mask = mask_creator(w.t()).t().cuda().bool()

    return (mask * w).contiguous(), mask.contiguous()


def check_24(w, num_rows_to_sample=50, _verbose=False):
    BLOCK_SIZE = 4
    MAX_NON_ZEROS = 2

    w = w.t().contiguous()

    print("check_24: w.shape = {}".format(w.shape))

    num_rows, num_cols = w.shape
    sampled_row_idxs = random.choices(range(num_rows), k=num_rows_to_sample)
    if _verbose:
        print(f"Sampled row idxs = {sampled_row_idxs}")

    total_segments = 0
    non_24_segments = 0
    for i in sampled_row_idxs:
        for j in range(0, num_cols - BLOCK_SIZE, BLOCK_SIZE):
            total_segments += 1
            block = w[i, j:j + BLOCK_SIZE]
            num_nonzero = torch.count_nonzero(block)
            if num_nonzero > MAX_NON_ZEROS:
                print("i = {} j = {} block = {}".format(i, j, block))
                non_24_segments += 1

    print(f"{non_24_segments} / {total_segments} do not have 2:4 structure.")


def compress_quantized_24_weight(q_24, size_k, size_n, num_bits):
    assert q_24.shape == (size_k, size_n)

    # Remove zp to normalize over 0
    max_q_val = (1 << num_bits) - 1
    zp = (max_q_val + 1) // 2
    q_24_no_zp = q_24 - zp

    # Compress
    q_24_no_zp = q_24_no_zp.t().contiguous()
    q_24_no_zp_comp, meta = sparse_semi_structured_from_dense_cutlass(
        q_24_no_zp)
    q_24_no_zp_comp = q_24_no_zp_comp.t().contiguous()

    # Restore zp
    q_24_comp = q_24_no_zp_comp + zp

    # Resize meta to its actual shape (without moving any data)
    meta = meta.resize_(meta.shape[1] // 2, meta.shape[0] * 2)

    return q_24_comp, meta


def marlin_24_quantize(
    w: torch.Tensor,
    num_bits: int,
    group_size: int,
):
    size_k, size_n = w.shape

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    # Inject 2:4 sparsity
    w_24, mask_24 = inject_24(w, size_k, size_n)

    # Quantize
    w_24_ref, q_w_24, s, g_idx, rand_perm = quantize_weights(w_24,
                                                             num_bits,
                                                             group_size,
                                                             act_order=False)

    # Compress quantized weight
    q_w_24_comp, meta = compress_quantized_24_weight(q_w_24, size_k, size_n,
                                                     num_bits)
    size_k_comp = size_k // 2

    # Reformat to marlin
    marlin_24_q_w_comp = marlin_weights(q_w_24_comp, size_k_comp, size_n,
                                        num_bits, marlin_24_perm[num_bits])
    marlin_24_s = marlin_permute_scales(s, size_k, size_n, group_size,
                                        marlin_24_scale_perm[num_bits],
                                        marlin_24_scale_perm_single[num_bits])

    # Create result
    res_list = [w_24_ref, marlin_24_q_w_comp, meta, marlin_24_s]
    for i in range(len(res_list)):
        res_list[i] = res_list[i].to(w.device)

    return res_list


def compute_max_diff(output, output_ref):
    return torch.mean(torch.abs(output - output_ref)) / torch.mean(
        torch.abs(output_ref))


class MarlinWorkspace:

    def __init__(self, out_features, min_thread_n, max_parallel):
        assert (out_features % min_thread_n == 0), (
            "out_features = {} is undivisible by min_thread_n = {}".format(
                out_features, min_thread_n))

        max_workspace_size = ((out_features // min_thread_n) * max_parallel)
>>>>>>> fixie-ai/vllm/main

        self.scratch = torch.zeros(max_workspace_size,
                                   dtype=torch.int,
                                   device="cuda")
