import os
import sys
import numpy as np


def init(M, N, K):
    A = np.random.rand(M, K) - 0.5
    B = np.random.rand(N, K) - 0.5
    return A, B

def gemm_cast_numpy(A, B):
    C = np.matmul(A, B.transpose(1,0))
    return C

def test_batch_norm_relu_max(M, N, K, in_type, out_type):
    """
        A: [M, K]
        B: [N, K]
    """

    A, B = init(M, N, K)

    A = A.astype("float16")
    B = B.astype("float16")

    C = gemm_cast_numpy(A, B)

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
    C.tofile(f"./data/C_ref_{M}_{N}_{K}_{in_type}_{out_type}.bin")
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    test_batch_norm_relu_max(576000, 64, 10, "float32", "float32")
