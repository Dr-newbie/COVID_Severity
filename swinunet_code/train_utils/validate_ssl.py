import torch


def validate_ssl_model(ssl_model, val_loader, criterion_ssl, device):
    """
    SSL 모델의 검증 루프 함수.

    Args:
        ssl_model (torch.nn.Module): SSL 모델.
        val_loader (torch.utils.data.DataLoader): 검증 데이터 로더.
        criterion_ssl: SSL 손실 계산을 위한 커스텀 손실 함수 객체.
        device (torch.device): GPU 또는 CPU 디바이스.

    Returns:
        float: 검증 데이터셋에서의 평균 손실 값.
    """
    ssl_model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for img_inputs in val_loader:
            img_inputs = img_inputs[0].to(device)  # 데이터 배치에서 이미지 입력만 가져옴
            rot_pred, contrastive_pred, rec_x = ssl_model(img_inputs)

            # 타겟 생성
            target_rot = torch.randint(0, 4, (rot_pred.size(0),), dtype=torch.long).to(device)
            target_contrastive = torch.randint(0, 2, (contrastive_pred.size(0),), dtype=torch.long).to(device)

            # 손실 계산
            rot_loss = criterion_ssl.rotation_loss(rot_pred, target_rot)
            contrastive_loss = criterion_ssl.contrastive_loss(contrastive_pred, target_contrastive)
            rec_loss = criterion_ssl.reconstruction_loss(rec_x, img_inputs)

            total_loss = rot_loss + contrastive_loss + rec_loss
            running_loss += total_loss.item()

    return running_loss / len(val_loader)
