import torch
from torch.cuda.amp import autocast
from tqdm import tqdm


def train_epoch(model, loader, optimizer, criterion, device, epoch, scaler=None):
    """
    모델의 학습을 수행하는 함수.

    Args:
        model: 학습 중인 모델
        loader: 학습 데이터 로더
        optimizer: 옵티마이저
        criterion: 손실 함수
        device: 사용 중인 디바이스 (CPU 또는 GPU)
        epoch: 현재 에포크
        scaler: Mixed Precision Training을 위한 GradScaler 객체 (선택 사항)

    Returns:
        loss: 학습 손실
        accuracy: 학습 정확도
    """
    model.train()  # 모델을 학습 모드로 전환
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(loader, desc=f'Epoch {epoch+1} [Train]')

    for img_inputs, labels, img_path, cyto_data in progress_bar:
        img_inputs, labels, cyto_data = img_inputs.to(device), labels.to(device), cyto_data.to(device)
        optimizer.zero_grad()

        if scaler:  # AMP 활성화 시
            with autocast():
                outputs = model(img_inputs, cyto_data)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:  # AMP 비활성화 시
            outputs = model(img_inputs, cyto_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix(loss=running_loss / total, accuracy=100. * correct / total)

    loss = running_loss / len(loader)
    accuracy = correct / total
    return loss, accuracy
