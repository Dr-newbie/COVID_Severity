import os
import csv
import warnings
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from medcam import medcam
from utils.model_wrapper import ModelWrapper


def test_model(model, loader, device, args, epoch=0):
    """
    모델 테스트 함수. MedCAM을 사용하여 attention map 생성 및 결과 저장.

    Args:
        model: 테스트할 모델.
        loader: 테스트 데이터 로더.
        device: 사용 중인 디바이스 (CPU 또는 GPU).
        args: Argument parser로 전달된 설정 값.
        epoch: 현재 테스트 중인 에포크.

    Returns:
        accuracy (float): 테스트 정확도.
    """
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    # 모델 래핑
    wrapped_model = ModelWrapper(model)

    # Debugging: 래핑된 모델 레이어 출력
    for name, module in wrapped_model.named_modules():
        print(name)

    # MedCAM 적용
    wrapped_model = medcam.inject(wrapped_model, save_maps=False, output_dir="attention_maps", layer='model.swin_unetr.swinViT.layers4.0', label=1)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    print(f'Model with MedCAM: {wrapped_model}')
    results_csv_path = 'swin_unet_predict.csv'

    # CSV 파일 초기화
    if epoch == 0:
        with open(results_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Image Path", "True Label", "Predicted Label"])

    wrapped_model.eval()  # 평가 모드로 전환
    progress_bar = tqdm(loader, desc=f'Test Epoch {epoch+1}')

    with torch.no_grad():
        for batch_idx, (img_inputs, labels, img_path, cyto_data) in enumerate(progress_bar):
            img_inputs = img_inputs.to(device)
            labels = labels.to(device)
            cyto_data = cyto_data.to(device)

            # 입력 데이터를 병합
            inputs = torch.cat([img_inputs.view(img_inputs.size(0), -1), cyto_data], dim=1)

            # 모델 예측 수행
            outputs = wrapped_model(inputs)
            _, predicted = outputs.max(1)

            # 결과 저장
            with open(results_csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([img_path[0], labels.item(), predicted.item()])

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix(accuracy=100. * correct / total)

    # 최종 정확도 계산
    accuracy = 100. * correct / total
    print(f"Label prediction of Swin done. Results saved to {results_csv_path}")

    # Confusion Matrix 생성 및 저장
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1', '2'], yticklabels=['0', '1', '2'])
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - Acc: {accuracy:.2f}%')

    save_path = f'confusion_matrix_epoch_{epoch+1}_acc_{accuracy:.2f}.png'
    plt.savefig(save_path)
    plt.show()

    print(f"Confusion matrix plot saved to {save_path}")
    print(f'Test Accuracy for Epoch {epoch+1}: {accuracy:.2f}%')

    return accuracy
