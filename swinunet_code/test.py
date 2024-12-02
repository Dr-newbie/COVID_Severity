import os
import torch
from torch.utils.data import DataLoader
from datasets.dataset import MedicalImageWithQuantDataset
from datasets.transforms import get_transforms
from models.Multimodal import MultiModalAttentionModel
from test_utils import test_model, test_model_attention


def test(args, device):
    """
    학습된 Swin UNETR 기반 모델 테스트 함수
    Args:
        args: Argument parser로 전달된 설정 값
        device: 테스트에 사용할 장치
    """
    print("Test start")

    # 모델 생성
    model = MultiModalAttentionModel(
        num_classes=args.num_classes,
        img_size=args.img_size,
        feature_size=args.feature_size
    ).to(device)

    # 데이터 변환 및 데이터셋 생성
    test_transforms = get_transforms(args.transform, args.img_size, is_train=False)
    test_dataset = MedicalImageWithQuantDataset(csv_file=args.test_csv, image_transform=test_transforms, img_size=args.img_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 학습된 가중치 로드
    if os.path.exists(args.checkpoint_path):
        model.load_state_dict(torch.load(args.checkpoint_path), strict=False)
        print("학습된 모델 가중치를 로드했습니다.")
    else:
        raise FileNotFoundError(f"No checkpoint found at {args.checkpoint_path}")

    # 테스트 모드에 따른 실행
    if args.feature_extract == 'medcam':
        # MedCAM 기반 테스트 실행
        for i in range(args.n_repeats):
            test_model(model, test_loader, device, args, epoch=i)

    elif args.feature_extract == 'attention_map':
        # Attention Map 기반 테스트 실행
        for i in range(args.n_repeats):
            test_model_attention(model, test_loader, device, args, epoch=i)

