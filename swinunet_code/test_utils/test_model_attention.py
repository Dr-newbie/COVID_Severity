import os
import cv2
import nibabel as nib
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from utils.attention_extraction import extract_attention_map


def test_model_attention(model, loader, device, args, epoch=0):
    """
    Attention Map을 추출하고 시각화하는 테스트 함수.

    Args:
        model (torch.nn.Module): 테스트할 모델.
        loader (torch.utils.data.DataLoader): 테스트 데이터 로더.
        device (torch.device): CPU 또는 GPU 디바이스.
        args: Argument parser로 전달된 설정 값.
        epoch (int): 현재 테스트 중인 에포크.

    Returns:
        accuracy (float): 테스트 정확도.
    """
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    # 결과 CSV 파일 경로 설정
    results_csv_path = 'swin_unet_predict.csv'
    if epoch == 0:
        with open(results_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Image Path", "True Label", "Predicted Label"])

    model.eval()
    progress_bar = tqdm(loader, desc=f'Test Epoch {epoch+1}')

    with torch.no_grad():
        for batch_idx, (img_inputs, labels, img_path, cyto_data) in enumerate(progress_bar):
            img_inputs = img_inputs.to(device)
            labels = labels.to(device)
            cyto_data = cyto_data.to(device)

            # 모델 출력 계산
            outputs = model(img_inputs, cyto_data)
            _, predicted = outputs.max(1)

            # 결과 저장
            with open(results_csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([img_path[0], labels.item(), predicted.item()])

            # Attention Map 추출
            attn_map = extract_attention_map(model, img_inputs)
            attn_map = attn_map[0].cpu().numpy()  # 첫 번째 샘플의 Attention Map 사용

            # Attention Map 리사이즈 및 처리
            attention_map_resized = cv2.resize(attn_map, (img_inputs.shape[-1], img_inputs.shape[-2]))
            if attention_map_resized.dtype != np.uint8:
                attention_map_resized = ((attention_map_resized - attention_map_resized.min()) /
                                         (attention_map_resized.max() - attention_map_resized.min()) * 255).astype(np.uint8)

            # 원본 이미지 처리
            original_img = nib.load(img_path[0]).get_fdata()
            slice_index = original_img.shape[2] // 2
            original_slice = original_img[:, :, slice_index]
            original_slice = ((original_slice - original_slice.min()) /
                              (original_slice.max() - original_slice.min()) * 255).astype(np.uint8)
            original_slice_color = cv2.cvtColor(original_slice, cv2.COLOR_GRAY2BGR)

            # Overlay 이미지 생성 및 저장
            overlay = cv2.applyColorMap(attention_map_resized, cv2.COLORMAP_JET)
            overlay = cv2.resize(overlay, (original_slice_color.shape[1], original_slice_color.shape[0]))
            overlay_image = cv2.addWeighted(original_slice_color, 0.6, overlay, 0.4, 0)

            save_dir = f"attention_maps/image_{batch_idx}"
            os.makedirs(save_dir, exist_ok=True)
            plt.imshow(overlay_image)
            plt.axis("off")
            plt.title(f"Overlay Image {batch_idx}")
            plt.savefig(f"{save_dir}/overlay_image_{batch_idx}.png")
            plt.close()

            # 정확도 계산
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
    plt.title(f'Confusion Matrix  - Acc: {accuracy:.2f}%')

    save_path = f'confusion_matrix_epoch_{epoch+1}_acc_{accuracy:.2f}.png'
    plt.savefig(save_path)
    plt.show()
    print(f"Confusion matrix plot saved to {save_path}")
    print(f'Test Accuracy for Epoch {epoch+1}: {accuracy:.2f}%')

    return accuracy
