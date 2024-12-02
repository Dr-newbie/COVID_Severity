import os
import logging
import torch
from torch import optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from datasets.dataset import MedicalImageWithQuantDataset
from datasets.transforms import get_transforms
from models.ssl_head import SSLHead
from models.Loss import Loss
from utils.early_stopping import EarlyStopping
from train_utils.train_ssl import ssl_train_epoch
from train_utils.validate_ssl import validate_ssl_model

def ssl_train(args, device):

    ssl_model = SSLHead(args).to(device)
    print("SSL training strat")


    if args.pretrained_weights and os.path.exists(args.pretrained_weights):
        checkpoint = torch.load(args.pretrained_weights)
        ssl_model.load_state_dict(checkpoint, strict = False)

    else :
        print("No pretrained weights found, training from scratch")
    
    criterion_ssl = Loss(args.batch_size, args)
    optimizer_ssl = optim.Adam(ssl_model.parameters(), lr=args.learning_rate * 0.1)  # 학습률 10배 감소
    scaler_ssl = GradScaler()

    # 데이터 변환 및 데이터셋 생성
    train_transforms = get_transforms(args.transform, args.img_size, is_train=True)
    val_transforms = get_transforms(args.transform, args.img_size, is_train=False)

    train_dataset = MedicalImageWithQuantDataset(csv_file=args.train_csv, image_transform=train_transforms, img_size=args.img_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_dataset = MedicalImageWithQuantDataset(csv_file=args.val_csv, image_transform=val_transforms, img_size=args.img_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Early Stopping 설정
    early_stopping = EarlyStopping(patience=args.patience, checkpoint_path=args.checkpoint_ssl)

    # SSL 학습 루프
    for epoch in range(args.ssl_epochs):
        # Training step
        ssl_loss = ssl_train_epoch(
            ssl_model, train_loader, optimizer_ssl, criterion_ssl, 
            scaler_ssl, device, epoch
        )
        # Validation step
        val_loss = validate_ssl_model(ssl_model, val_loader, criterion_ssl, device)

        print(f'SSL Epoch {epoch+1} Train Loss: {ssl_loss:.4f}, Validation Loss: {val_loss:.4f}')
        logging.info(f'SSL Epoch {epoch+1} - Train Loss: {ssl_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Early stopping 체크
        early_stopping(val_loss, ssl_model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            logging.info("Early stopped")
            break

    # SSL 모델 가중치 저장
    torch.save(ssl_model.state_dict(), args.checkpoint_ssl)
    print(f"SSL 학습 완료. 가중치가 {args.checkpoint_ssl}에 저장되었습니다.")
    logging.info("SSL training completed")    