import os
import sys
import numpy as np


def init(MD, MR, K0, K1, N1, K2, N2):
    inp = np.random.rand(MD*MR, K0).astype(np.float32)
    w1 = np.random.rand(N1, K1).astype(np.float16)
    g1 = np.random.rand(N1).astype(np.float16)
    b1 = np.random.rand(N1).astype(np.float16)
    w2 = np.random.rand(N2, K2).astype(np.float16)
    g2 = np.random.rand(N2).astype(np.float16)
    b2 = np.random.rand(N2).astype(np.float16)
    return inp, w1, g1, b1, w2, g2, b2

def gemm_bn_relu_max_tile_cat_gemm_bn_relu_max_numpy(inp, w1, g1, b1, w2, g2, b2, MD, MR, K0, K1, N1, K2, N2):
    # pad
    outPad = np.zeros([MD*MR, K1], dtype=inp.dtype)
    outPad[:,:K0] = inp
    outPad = outPad.astype(w1.dtype)

    # gemm1
    out_gemm1 = np.matmul(outPad, w1.transpose(1,0))
    out_bnrelu1 = np.maximum(out_gemm1 * g1 + b1, 0).reshape(MD, MR, N1)
    out_max1 = out_bnrelu1.max(axis=1, keepdims=True)
    out_tile1 = np.repeat(out_max1, 32, axis=1)
    outGemm1 = np.zeros([MD, MR, K2], dtype=out_bnrelu1.dtype)
    outGemm1 = np.concatenate([out_bnrelu1, out_tile1], axis=2).reshape(-1, K2)

    # gemm2
    out_gemm2 = np.matmul(outGemm1, w2.transpose(1,0))
    out_bnrelu2 = np.maximum(out_gemm2 * g2 + b2, 0).reshape(MD, MR, N2)
    out_max2 = out_bnrelu2.max(axis=1)
    outGemm2 = out_max2.astype(np.float32)

    return outPad, outGemm1, outGemm2, out_gemm1, out_gemm2

def test_gemm_bn_relu_max_tile_cat_gemm_bn_relu_max_numpy(MD, MR, K0, K1, N1, K2, N2):
    """
        A: [M, K]
        B: [N, K]
        gamma: [N]
        beta: [N]
    """

    inp, w1, g1, b1, w2, g2, b2 = init(MD, MR, K0, K1, N1, K2, N2)

    outPad, outGemm1, outGemm2, out_gemm1, out_gemm2 = gemm_bn_relu_max_tile_cat_gemm_bn_relu_max_numpy(inp, w1, g1, b1, w2, g2, b2, MD, MR, K0, K1, N1, K2, N2)

    _postfix = f"{MD}_{MR}_{K0}_{K1}_{N1}_{K2}_{N2}.bin"

    inp.tofile(f"./data/inp_{_postfix}")
    w1.tofile(f"./data/w1_{_postfix}")
    g1.tofile(f"./data/g1_{_postfix}")
    b1.tofile(f"./data/b1_{_postfix}")
    w2.tofile(f"./data/w2_{_postfix}")
    g2.tofile(f"./data/g2_{_postfix}")
    b2.tofile(f"./data/b2_{_postfix}")
    outPad.tofile(f"./data/outPad_{_postfix}")
    outGemm1.tofile(f"./data/outGemm1_{_postfix}")
    outGemm2.tofile(f"./data/outGemm2_{_postfix}")

    out_gemm1.tofile(f"./data/out_gemm1_{_postfix}")
    out_gemm2.tofile(f"./data/out_gemm2_{_postfix}")
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    test_gemm_bn_relu_max_tile_cat_gemm_bn_relu_max_numpy(18000, 32, 10, 16, 32, 64, 64)
    test_gemm_bn_relu_max_tile_cat_gemm_bn_relu_max_numpy(19000, 32, 10, 16, 32, 64, 64)
    test_gemm_bn_relu_max_tile_cat_gemm_bn_relu_max_numpy(20000, 32, 10, 16, 32, 64, 64)
    test_gemm_bn_relu_max_tile_cat_gemm_bn_relu_max_numpy(21000, 32, 10, 16, 32, 64, 64)
