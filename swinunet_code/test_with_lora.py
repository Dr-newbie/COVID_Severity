import os
import torch
from torch.utils.data import DataLoader
from datasets.dataset import MedicalImageWithQuantDataset
from datasets.transforms import get_transforms
from models.Multimodal import MultiModalAttentionModel
from models.apply_lora_to_swin_unetr import apply_lora_to_swin_unetr #need update
from test_utils.test_model import test_model
from test_utils.test_model_attention import test_model_attention #need update 


def test_with_lora(args, device):
    """
    LoRA를 적용한 Swin UNETR 모델의 테스트 함수
    Args:
        args: Argument parser로 전달된 설정 값
        device: 테스트에 사용할 장치
    """
    print("LoRA Test start")

    # Swin UNETR 모델 생성
    model = MultiModalAttentionModel(
        num_classes=args.num_classes,
        img_size=args.img_size,
        feature_size=args.feature_size
    ).to(device)

    # LoRA 적용
    apply_lora_to_swin_unetr(model, r=args.r, alpha=args.alpha)
    print("LoRA를 모델에 적용했습니다.")

    # 학습된 LoRA + Swin 파라미터 로드
    if os.path.exists(args.lora_saved_path):
        lora_state_dict = torch.load(args.lora_saved_path, map_location=device)
        model.load_state_dict(lora_state_dict, strict=False)
        print("LoRA 가중치를 모델에 로드했습니다.")
    else:
        raise FileNotFoundError(f"No LoRA checkpoint found at {args.lora_saved_path}")

    print(model)

    # 테스트 데이터 로더 준비
    test_transforms = get_transforms(args.transform, args.img_size, is_train=False)
    test_dataset = MedicalImageWithQuantDataset(csv_file=args.test_csv, image_transform=test_transforms, img_size=args.img_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 테스트 실행
    if args.feature_extract == 'medcam':
        test_accuracy = test_model(model, test_loader, device, args)
    elif args.feature_extract == 'attention_map':
        test_accuracy = test_model_attention(model, test_loader, device, args)

    print(f"Test Accuracy: {test_accuracy:.2f}%")
