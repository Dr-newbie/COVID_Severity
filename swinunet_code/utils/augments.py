import numpy as np
import torch
from numpy.random import randint

def patch_rand_drop(args, x, x_rep=None, max_drop=0.3, max_block_sz=0.25, tolr=0.05):
    """
    랜덤 패치 삭제를 통해 이미지 데이터를 증강하는 함수.

    Args:
        args: 설정 인자를 담은 객체 (local_rank 포함).
        x (torch.Tensor): 입력 텐서 (C, H, W, D 형식).
        x_rep (torch.Tensor, optional): 대체 데이터로 사용할 텐서. Default는 None.
        max_drop (float, optional): 전체 픽셀 대비 삭제 비율. Default는 0.3.
        max_block_sz (float, optional): 삭제할 패치의 최대 크기 비율. Default는 0.25.
        tolr (float, optional): 삭제할 패치 크기의 최소 허용 비율. Default는 0.05.

    Returns:
        torch.Tensor: 증강된 텐서.
    """
    c, h, w, z = x.size()
    n_drop_pix = np.random.uniform(0, max_drop) * h * w * z
    mx_blk_height = int(h * max_block_sz)
    mx_blk_width = int(w * max_block_sz)
    mx_blk_slices = int(z * max_block_sz)
    tolr = (int(tolr * h), int(tolr * w), int(tolr * z))
    total_pix = 0

    while total_pix < n_drop_pix:
        rnd_r = randint(0, h - tolr[0])
        rnd_c = randint(0, w - tolr[1])
        rnd_s = randint(0, z - tolr[2])
        rnd_h = min(randint(tolr[0], mx_blk_height) + rnd_r, h)
        rnd_w = min(randint(tolr[1], mx_blk_width) + rnd_c, w)
        rnd_z = min(randint(tolr[2], mx_blk_slices) + rnd_s, z)

        if x_rep is None:
            x_uninitialized = torch.empty(
                (c, rnd_h - rnd_r, rnd_w - rnd_c, rnd_z - rnd_s), dtype=x.dtype, device=args.local_rank
            ).normal_()
            x_uninitialized = (x_uninitialized - torch.min(x_uninitialized)) / (
                torch.max(x_uninitialized) - torch.min(x_uninitialized)
            )
            x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_uninitialized
        else:
            x[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z] = x_rep[:, rnd_r:rnd_h, rnd_c:rnd_w, rnd_s:rnd_z]

        total_pix += (rnd_h - rnd_r) * (rnd_w - rnd_c) * (rnd_z - rnd_s)

    return x

def aug_rand(args, samples):
    """
    랜덤 증강을 수행하는 함수. 주어진 샘플에 랜덤 패치 삭제를 적용.

    Args:
        args: 설정 인자를 담은 객체.
        samples (torch.Tensor): 입력 텐서 (N, C, H, W, D 형식).

    Returns:
        torch.Tensor: 증강된 텐서.
    """
    img_n = samples.size()[0]
    x_aug = samples.detach().clone()

    for i in range(img_n):
        x_aug[i] = patch_rand_drop(args, x_aug[i])
        idx_rnd = randint(0, img_n)
        if idx_rnd != i:
            x_aug[i] = patch_rand_drop(args, x_aug[i], x_aug[idx_rnd])

    return x_aug
