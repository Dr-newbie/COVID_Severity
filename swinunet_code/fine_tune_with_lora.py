import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from datasets.dataset import MedicalImageWithQuantDataset
from datasets.transforms import get_transforms
from models.Multimodal import MultiModalAttentionModel
from models.apply_lora_to_swin_unetr import apply_lora_to_swin_unetr#need update
from utils.early_stopping import EarlyStopping
from utils.plot_utils import plot_loss_accuracy #need update
from train_utils import train_epoch, validate_epoch #need update


def fine_tune_with_lora(args, device):
    """
    LoRA를 적용한 Swin UNETR 모델의 미세 조정 함수
    Args:
        args: Argument parser로 전달된 설정 값
        device: 학습에 사용할 장치
    """
    # Swin UNETR 모델 생성
    model = MultiModalAttentionModel(
        num_classes=args.num_classes,
        img_size=args.img_size,
        feature_size=args.feature_size
    ).to(device)

    print("LoRA Finetuning start")

    # 학습된 모델 가중치 로드
    if os.path.exists(args.checkpoint_path):
        swin_checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model_state_dict = model.state_dict()
        model.load_state_dict(swin_checkpoint, strict=False)
        print("기존 학습된 모델 가중치를 로드했습니다.")
    else:
        raise FileNotFoundError(f"No checkpoint found at {args.checkpoint_path}")

    # LoRA 적용
    apply_lora_to_swin_unetr(model, r=args.r, alpha=args.alpha)
    print("LoRA를 모델에 적용했습니다.")
    print('model with LoRA applied:', model)

    # 모든 LoRA가 아닌 파라미터는 Freeze
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False

    # Trainable 파라미터 확인
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable parameter: {name}")

    # 데이터셋 및 데이터 로더 설정
    train_transforms = get_transforms(args.transform, args.img_size, is_train=True)
    val_transforms = get_transforms(args.transform, args.img_size, is_train=False)

    train_dataset = MedicalImageWithQuantDataset(csv_file=args.train_csv, image_transform=train_transforms, img_size=args.img_size)
    val_dataset = MedicalImageWithQuantDataset(csv_file=args.val_csv, image_transform=val_transforms, img_size=args.img_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 손실 함수 및 옵티마이저 설정
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)  # Fine-tuning 시 학습률 낮게 설정
    criterion = torch.nn.CrossEntropyLoss()

    # 학습/검증 손실 및 정확도 기록
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # Early Stopping 설정
    early_stopping = EarlyStopping(patience=args.patience, checkpoint_path=args.lora_saved_path)

    # 학습 루프
    for epoch in range(args.epochs):
        # Training step
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        # Validation step
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, epoch)

        # 손실 및 정확도 기록
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

    # 손실 및 정확도 플롯 생성
    plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs, args)

    # LoRA 적용된 모델 가중치 저장
    torch.save(model.state_dict(), args.lora_saved_path)
    print(f"LoRA가 적용된 모델 가중치가 '{args.lora_saved_path}'에 저장되었습니다.")
