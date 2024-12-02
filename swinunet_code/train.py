import os
import torch
from torch import optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from datasets.dataset import MedicalImageWithQuantDataset
from datasets.transforms import get_transforms
from models.Multimodal import MultiModalAttentionModel
from utils.early_stopping import EarlyStopping
from utils.plot_utils import plot_loss_accuracy  #need udate
from train_utils import train_epoch, validate_epoch #need update
import pandas as pd


def train(args, device):
    """
    Swin UNETR 기반 멀티모달 학습 함수
    Args:
        args: Argument parser로 전달된 설정 값
        device: 학습에 사용할 장치
    """
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Swin UNETR 모델 생성
    model = MultiModalAttentionModel(
        num_classes=args.num_classes,
        img_size=args.img_size,
        feature_size=args.feature_size
    ).to(device)

    print("Training start")

    # SSL 학습된 가중치를 불러와 모델에 적용
    if os.path.exists(args.checkpoint_ssl):
        ssl_checkpoint = torch.load(args.checkpoint_ssl)
        model_state_dict = model.swin_unetr.state_dict()

        # SSL 가중치 중 현재 모델에 맞는 것들만 필터링
        filtered_ssl_checkpoint = {k: v for k, v in ssl_checkpoint.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}

        # 필터링한 가중치를 로드
        model.swin_unetr.load_state_dict(filtered_ssl_checkpoint, strict=False)
        print("SSL 가중치를 멀티모달 모델에 로드했습니다.")
    else:
        raise FileNotFoundError(f"No SSL checkpoint found at {args.checkpoint_ssl}")

    # 데이터셋 및 데이터 로더 설정
    train_transforms = get_transforms(args.transform, args.img_size, is_train=True)
    val_transforms = get_transforms(args.transform, args.img_size, is_train=False)

    train_dataset = MedicalImageWithQuantDataset(csv_file=args.train_csv, image_transform=train_transforms, img_size=args.img_size)
    val_dataset = MedicalImageWithQuantDataset(csv_file=args.val_csv, image_transform=val_transforms, img_size=args.img_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 손실 함수, 옵티마이저, 스케일러 설정
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = GradScaler()

    # 모델 정보 출력
    print_model_info(model, optimizer, train_loader, args)

    # Early Stopping 설정
    early_stopping = EarlyStopping(patience=args.patience, checkpoint_path=args.checkpoint_path)

    # 학습 루프
    for epoch in range(args.epochs):
        # Training step
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        # Validation step
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, epoch)

        # 손실 및 정확도 기록
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Early stopping 체크
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # 손실 및 정확도 플롯 생성
    plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, args)

    # 최종 모델 저장
    torch.save(model.state_dict(), args.checkpoint_path)
    print(f"model saved at {args.checkpoint_path}")


def print_model_info(model, optimizer, data_loader, args):
    """
    모델 구조 및 학습 가능한 파라미터 정보를 출력합니다.
    Args:
        model: 학습 모델
        optimizer: 옵티마이저
        data_loader: 데이터 로더
        args: Argument parser로 전달된 설정 값
    """
    model.eval()  # 평가 모드로 전환
    with torch.no_grad():
        for img_inputs, labels, img_path, cyto_data in data_loader:
            img_inputs, cyto_data = img_inputs.to(args.device), cyto_data.to(args.device)
            output = model(img_inputs, cyto_data)
            print("Model output shape:", output.shape)
            break

    # 학습 가능한 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")

    # 학습률 출력
    for param_group in optimizer.param_groups:
        print(f"Learning Rate: {param_group['lr']}")

    # 모델의 총 레이어 수 출력
    total_layers = sum(1 for _ in model.modules())
    print(f"Total number of layers in the model: {total_layers}")
