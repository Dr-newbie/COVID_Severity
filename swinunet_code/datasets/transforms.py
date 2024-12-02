import torch
from monai.transforms import Compose, Resize, ScaleIntensityRange, ToTensor, RandRotate90, RandFlip, RandZoom
import torch.nn.functional as F


def get_transforms(transform, img_size, is_train=True):
    """
    데이터 증강 및 전처리 파이프라인을 생성
    """
    if transform == 'true':
        if is_train:
            return Compose([
                Resize(spatial_size=(img_size, img_size, img_size), mode='trilinear'),
                ScaleIntensityRange(a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
                RandRotate90(prob=0.5, spatial_axes=(0, 1)),
                RandFlip(prob=0.5, spatial_axis=0),
                RandZoom(prob=0.3, min_zoom=0.9, max_zoom=1.1),
                ToTensor()
            ])
        else:
            return Compose([
                Resize(spatial_size=(img_size, img_size, img_size), mode='trilinear'),
                ScaleIntensityRange(a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
                ToTensor()
            ])
    else:
        return Compose([
            Resize(spatial_size=(img_size, img_size, img_size), mode='trilinear'),
            ScaleIntensityRange(a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            ToTensor()
        ])


def safe_resize(img, spatial_size):
    """
    이미지를 지정된 크기로 안전하게 리사이즈
    """
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)로 차원 추가
    img_resized = F.interpolate(img, size=spatial_size, mode='trilinear', align_corners=False)
    return img_resized.squeeze(0).squeeze(0)  # 차원 제거 후 반환
