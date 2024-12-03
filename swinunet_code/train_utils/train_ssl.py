import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from utils.augments import aug_rand  # 필요한 경우 추가로 import

def ssl_train_epoch(args, ssl_model, train_loader, optimizer, criterion_ssl, scaler, device, epoch):
    """
    Self-Supervised Learning(SSL) 모델 학습 루프 함수.

    Args:
        ssl_model (torch.nn.Module): SSL 모델.
        train_loader (torch.utils.data.DataLoader): 학습 데이터 로더.
        optimizer (torch.optim.Optimizer): 옵티마이저.
        criterion_ssl: SSL 손실 계산을 위한 커스텀 손실 함수 객체.
        scaler (torch.cuda.amp.GradScaler): AMP 스케일러.
        device (torch.device): GPU 또는 CPU 디바이스.
        epoch (int): 현재 학습 에포크 번호.

    Returns:
        float: 학습 데이터셋에서의 평균 손실 값.
    """
    ssl_model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f'SSL Epoch {epoch+1} [Train]')

    for img_inputs in progress_bar:
        img_inputs = img_inputs[0].to(device)
        img_view1 = aug_rand(args,img_inputs)  # 데이터 증강
        img_view2 = aug_rand(args,img_inputs)

        optimizer.zero_grad()
        with autocast():
            # 두 뷰에서 예측 수행
            rot1_pred, contrastive1_pred, rec_x1 = ssl_model(img_view1)
            rot2_pred, contrastive2_pred, rec_x2 = ssl_model(img_view2)

            # 타겟 생성
            target_rot = torch.randint(0, 4, (rot1_pred.size(0),), dtype=torch.long).to(device)
            target_contrastive = torch.randint(0, 2, (contrastive1_pred.size(0),), dtype=torch.long).to(device)

            # 손실 계산
            rot_loss = (criterion_ssl.rotation_loss(rot1_pred, target_rot) +
                        criterion_ssl.rotation_loss(rot2_pred, target_rot)) / 2
            contrastive_loss = (criterion_ssl.contrastive_loss(contrastive1_pred, contrastive2_pred) +
                                criterion_ssl.contrastive_loss(contrastive2_pred, contrastive1_pred)) / 2
            rec_loss = (criterion_ssl.reconstruction_loss(rec_x1, img_view1) +
                        criterion_ssl.reconstruction_loss(rec_x2, img_view2)) / 2

            total_loss = rot_loss + contrastive_loss + rec_loss

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += total_loss.item()
        progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

    return running_loss / len(train_loader)
