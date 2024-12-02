import argparse
from ssl_train import ssl_train
from train import train
from fine_tune_with_lora import fine_tune_with_lora
from test_with_lora import test_with_lora
from test import test
import os
import torch

def clear():
    os.system('clear')

def parse_args():
    parser = argparse.ArgumentParser(description="Run medical image model")
   # 파일 경로 관련 인자
    parser.add_argument('--train_csv', type=str, default='../swin_data/Train_Data.csv', help='Training CSV file path')
    parser.add_argument('--val_csv', type=str, default='../swin_data/Validation_Data.csv', help='Validation CSV file path')
    parser.add_argument('--test_csv', type=str, default='../swin_data/Test_Data.csv', help='Test CSV file path')
    parser.add_argument('--checkpoint_path', type=str, default='../model_swinvit.pt', help='Checkpoint file path')
    parser.add_argument('--checkpoint_ssl',type=str, default='../model_swinvit.pt', help='SSL .pt path for training, 이건 SSL pretrained 완료 된 pt')
    parser.add_argument('--ssl_checkpoint_path',type=str, default=' ', help='SSL pretrain path')
    parser.add_argument('--ssl_train_csv', type=str, default='../swin_data/filtered_nifti_cleaned.csv', help='Training CSV file path')
    parser.add_argument('--pretrained_weights', type=str, help='Path to pretrained weights for SSL pretrain')


    # SSL 관련 인자
    parser.add_argument('--ssl_epochs', type=int, default=100, help='Number of SSL training epochs')

    # 학습 관련 하이퍼파라미터 인자
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--transform", type=str, help='Data augmentation, Resie, Rotate & Zoom', choices=['true','false']) # Data augmentation 관련 parser

    # 장치 설정 인자
    parser.add_argument('--device', type=str,  help='Device to use for training')

    # 모델 파라미터 인자
    parser.add_argument('--feature_size', type=int, default=96, help='Feature size for SwinUNETR')
    parser.add_argument('--img_size', type=int, default=96, help='Input image size')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of output classes')
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument('--cyto_input_size', type=int, default=178, help='Input cyto size')

    # 기타 인자
    parser.add_argument('--n_repeats', type=int, default=1, help='Number of times to repeat the test')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'ssl_train','extract_embedding', 'test_with_lora','fine_tune_with_lora'], required=True, help='Mode: train, test, or ssl_train')
    # AMP 설정
    parser.add_argument('--amp', action='store_true', help='Enable Automatic Mixed Precision (AMP) for faster training')

    #LoRA 설정
    parser.add_argument('--use_lora', action='store_true', help='Apply LoRA to the model if set')
    parser.add_argument('--lora_saved_path', type=str, help='path to store LoRA')
    parser.add_argument('--lora_checkpoint_path', type=str, help='path to bring LoRA finetuned path')
    parser.add_argument('--r', default=1,type=int, help='hyperparam r for lora')
    parser.add_argument('--alpha', default=4,type=int, help='hyperparam alpha for lora')
    parser.add_argument('--feature_extract', type=str, help='feature extract method', choices=['medcam','attention_map'])

    return parser.parse_args()


if __name__ == "__main__":

    clear()

    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.mode == 'ssl_train':
        ssl_train(args, device)
    
    elif args.mode == 'train':
        train(args, device)

    elif args.mode == 'fine_tune_with_lora':
        fine_tune_with_lora(args, device)
    
    elif args.mode == 'test':
        test(args, device)

    elif args.mode == 'test_with_lora':
        test_with_lora(args, device)
