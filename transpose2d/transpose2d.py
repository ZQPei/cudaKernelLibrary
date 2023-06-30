import os
import sys
import numpy as np


def init(N, C):
    A = np.random.rand(N, C) - 0.5
    return A

def transpose2d(A):
    B = A.transpose(1, 0)
    return B

def test_transpose2d(N, C, dtype):
    """
        A: [M, K]
        B: [N, K]
    """

    A = init(N, C)

    A = A.astype(dtype)
    B = transpose2d(A)

    A.tofile(f"./data/input_{N}_{C}.bin")
    B.tofile(f"./data/output_ref_{N}_{C}.bin")
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    test_transpose2d(1024, 1024, "float32")
