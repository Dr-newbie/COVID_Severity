import torch
from tqdm import tqdm


def validate_epoch(model, loader, criterion, device, epoch):
    """
    모델의 검증을 수행하는 함수.

    Args:
        model: 학습된 모델
        loader: 검증 데이터 로더
        criterion: 손실 함수
        device: 사용 중인 디바이스 (CPU 또는 GPU)
        epoch: 현재 에포크

    Returns:
        loss: 검증 손실
        accuracy: 검증 정확도
    """
    model.eval()  # 모델을 평가 모드로 전환
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(loader, desc=f'Epoch {epoch+1} [Validation]')

    with torch.no_grad():  # 검증 과정에서는 gradient를 계산하지 않음
        for img_inputs, labels, img_path, cyto_data in progress_bar:
            img_inputs, labels, cyto_data = img_inputs.to(device), labels.to(device), cyto_data.to(device)
            
            # 모델 예측
            outputs = model(img_inputs, cyto_data)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix(loss=running_loss / total, accuracy=100. * correct / total)

    loss = running_loss / len(loader)
    accuracy = correct / total
    return loss, accuracy
