import os
import sys
import numpy as np


def init(M, N, K):
    A = np.random.rand(M, K) - 0.5
    B = np.random.rand(K, N) - 0.5
    return A, B

def hgemm_numpy(A, B):
    C = np.matmul(A, B)
    return C

def test_batch_norm_relu_max(M, N, K, in_type, out_type):
    """
        A: [M, K]
        B: [N, K]
        C: [M, N]
    """

    A, B = init(M, N, K)

    A = A.astype(in_type)
    B = B.astype(in_type)

    C = hgemm_numpy(A, B)

    C = C.astype(out_type)

    A.tofile(f"./data/A_{M}_{N}_{K}_{in_type}_{out_type}.bin")
    A.transpose(1,0).tofile(f"./data/AT_{M}_{N}_{K}_{in_type}_{out_type}.bin")
    B.tofile(f"./data/B_{M}_{N}_{K}_{in_type}_{out_type}.bin")
    B.transpose(1,0).tofile(f"./data/BT_{M}_{N}_{K}_{in_type}_{out_type}.bin")
    C.tofile(f"./data/C_ref_{M}_{N}_{K}_{in_type}_{out_type}.bin")


if __name__ == "__main__":
    test_batch_norm_relu_max(1024, 1024, 1024, "float16", "float16")
    test_batch_norm_relu_max(640000, 128, 32, "float16", "float16")
    test_batch_norm_relu_max(640000, 32, 16, "float16", "float16")
