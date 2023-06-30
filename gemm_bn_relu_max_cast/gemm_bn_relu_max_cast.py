import os
import sys
import numpy as np


def init(M, N, K):
    A = np.random.rand(M, K)
    B = np.random.rand(N, K)
    gamma = np.random.rand(N)
    beta = np.random.rand(N)
    return A, B, gamma, beta

def gemm_bn_relu_max_cast_numpy(A, B, gamma, beta):
    out_gemm = np.matmul(A, B.transpose(1,0))
    out_bn_relu = np.maximum(out_gemm * gamma + beta, 0).reshape(18000, -1, 64)
    out_max = out_bn_relu.max(axis=1)
    return out_max

def test_gemm_bn_relu_max_cast(M, MM, N, K, in_type, out_type):
    """
        A: [M, K]
        B: [N, K]
        gamma: [N]
        beta: [N]
    """

    A, B, gamma, beta = init(M, N, K)

    A = A.astype("float16")
    B = B.astype("float16")
    gamma = gamma.astype("float16")
    beta = beta.astype("float16")

    C = gemm_bn_relu_max_cast_numpy(A, B, gamma, beta)

    # A.tofile(f"./data/A_{M}_{N}_{K}_{in_type}_{in_type}.bin")
    # B.tofile(f"./data/B_{M}_{N}_{K}_{in_type}_{in_type}.bin")
    # C.tofile(f"./data/C_ref_{M}_{N}_{K}_{in_type}_{in_type}.bin")

    A = A.astype("float32")
    C = C.astype("float32")

    B_pad = np.zeros([N, 16], dtype=np.float16)
    B_pad[:, :10] = B

    A.tofile(f"./data/A_{M}_{N}_{K}_{in_type}_{out_type}.bin")
    B.tofile(f"./data/B_{M}_{N}_{K}_{in_type}_{out_type}.bin")
    B_pad.tofile(f"./data/B_pad_{M}_{N}_{K}_{in_type}_{out_type}.bin")
    gamma.tofile(f"./data/gamma_{M}_{N}_{K}_{in_type}_{out_type}.bin")
    beta.tofile(f"./data/beta_{M}_{N}_{K}_{in_type}_{out_type}.bin")
    C.tofile(f"./data/C_ref_{M}_{N}_{K}_{in_type}_{out_type}.bin")
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    test_gemm_bn_relu_max_cast(576000, 18000, 64, 10, "float32", "float32")
