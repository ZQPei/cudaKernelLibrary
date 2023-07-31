import os
import sys
import numpy as np


def init(M, K1, K2, N):
    inp = np.random.rand(M, K1) - 0.5
    w1 = np.random.rand(K1, K2) - 0.5
    w2 = np.random.rand(K2, N) - 0.5
    return inp, w1, w2

def sgemm_b2b_numpy(inp, w1, w2):
    mid = np.matmul(inp, w1)
    out = np.matmul(mid, w2)
    return mid, out

def test_sgemm_b2b(M, K1, K2, N):
    """
        inp: [M, K1]
        w1:  [K1, K2]
        w2:  [K2, N]
    """

    inp, w1, w2 = init(M, K1, K2, N)

    inp = inp.astype("float32")
    w1 = w1.astype("float32")
    w2 = w2.astype("float32")

    mid, out = sgemm_b2b_numpy(inp, w1, w2)
    mid = mid.astype("float32")
    out = out.astype("float32")

    inp.tofile(f"./data/inp_{M}_{K1}_{K2}_{N}.bin")
    w1.tofile(f"./data/w1_{M}_{K1}_{K2}_{N}.bin")
    w2.tofile(f"./data/w2_{M}_{K1}_{K2}_{N}.bin")
    mid.tofile(f"./data/mid_ref_{M}_{K1}_{K2}_{N}.bin")
    out.tofile(f"./data/out_ref_{M}_{K1}_{K2}_{N}.bin")
    # import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    test_sgemm_b2b(1024, 128, 128, 128)
    test_sgemm_b2b(1024, 1024, 128, 128)
    test_sgemm_b2b(1024, 1024, 128, 1024)
    test_sgemm_b2b(640000, 128, 128, 128)
