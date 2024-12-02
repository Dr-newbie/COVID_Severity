from models.LoRA import LoRALinear
import torch.nn as nn

def apply_lora_to_swin_unetr(model, r, alpha):
    """
    Swin-UNETR 모델의 qkv 계층에 LoRA를 적용하는 함수.

    Args:
        model (nn.Module): Swin-UNETR 모델.
        r (int): LoRA의 rank 하이퍼파라미터.
        alpha (float): LoRA의 scaling 하이퍼파라미터.

    Returns:
        None: 모델 내 qkv 계층에 LoRA가 적용됨.
    """
    qkv_layers = []  # qkv 연산에 해당하는 nn.Linear 계층을 저장할 리스트

    # 모델의 모든 nn.Linear 계층 중에서 'qkv'가 포함된 계층을 탐색하여 qkv_layers에 저장
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'qkv' in name:
            qkv_layers.append((name, module))
    
    # 아래에서부터 6개의 qkv 계층에만 LoRA 적용
    for name, module in qkv_layers:
        lora_linear = LoRALinear(module, r=r, alpha=alpha)
        parent_module = dict(model.named_modules())[name.rsplit('.', 1)[0]]  # 부모 모듈 가져오기
        setattr(parent_module, name.split('.')[-1], lora_linear)  # 부모 모듈에 LoRA 적용
