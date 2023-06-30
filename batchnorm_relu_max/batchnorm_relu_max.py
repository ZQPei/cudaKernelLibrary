import os
import sys
import numpy as np


def init(N, D, C):
    inp = np.random.rand(N, D, C) - 0.5
    gamma = np.random.rand(C) - 0.5
    beta = np.random.rand(C) - 0.5
    return inp, gamma, beta

def batch_norm_relu_max_numpy(inp, gamma, beta):
    out_bn_relu = np.maximum(inp * gamma + beta, 0)
    out_max = out_bn_relu.max(axis=1)
    return out_max

def test_batch_norm_relu_max(N, D, C, in_type, out_type):
    """
        inp: [N, D, C]
        gamma: [C]
        beta: [C]
    """

    inp, gamma, beta = init(N, D, C)

    inp = inp.astype(in_type)
    gamma = gamma.astype(in_type)
    beta = beta.astype(in_type)

    out = batch_norm_relu_max_numpy(inp, gamma, beta)

    out = out.astype(out_type)

    inp.tofile(f"./data/input_{N}_{D}_{C}_{in_type}_{out_type}.bin")
    gamma.tofile(f"./data/gamma_{N}_{D}_{C}_{in_type}_{out_type}.bin")
    beta.tofile(f"./data/beta_{N}_{D}_{C}_{in_type}_{out_type}.bin")
    out.tofile(f"./data/output_ref_{N}_{D}_{C}_{in_type}_{out_type}.bin")
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    test_batch_norm_relu_max(18000, 32, 64, "float32", "float32")
    test_batch_norm_relu_max(18000, 32, 64, "float16", "float32")
    test_batch_norm_relu_max(18000, 32, 64, "float16", "float16")
