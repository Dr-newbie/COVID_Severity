import torch


def extract_attention_map(model, img_inputs):
    """
    Swin 모델에서 Attention Map을 추출하는 함수.

    Args:
        model (torch.nn.Module): Attention Map을 추출할 모델.
        img_inputs (torch.Tensor): 모델에 입력될 이미지 텐서.

    Returns:
        torch.Tensor: 추출된 Attention Map.

    Raises:
        ValueError: Attention Map이 모델에서 추출되지 않을 경우.
    """
    attention_maps = []
    img_inputs = img_inputs.to(next(model.parameters()).device)

    for name, module in model.named_modules():
        if hasattr(module, "attn") and hasattr(module.attn, "qkv"):
            b, c, d, h, w = img_inputs.shape
            print(f"Original img_inputs shape: {img_inputs.shape}")

            # 3D 입력을 2D로 변환하여 (batch * d * h * w, c) 형태로 맞춤
            img_inputs_reshaped = img_inputs.view(b, -1).transpose(0, 1).reshape(-1, c)
            print(f"Reshaped img_inputs shape for qkv: {img_inputs_reshaped.shape}")

            try:
                # qkv 적용 후 (num_tokens, 3 * dim)으로 변환
                qkv = module.attn.qkv(img_inputs_reshaped)
                qkv = qkv.view(b * d * h * w, 3, module.attn.num_heads, -1 // module.attn.num_heads)
                q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
                print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")

                # Attention Map 계산
                attention_map = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
                attention_maps.append(attention_map.mean(dim=1))
                print(f"Attention map shape: {attention_map.shape}")
            except Exception as e:
                print(f"Error in attention map calculation: {e}")
                raise e

    if attention_maps:
        return attention_maps[0]
    else:
        raise ValueError("No attention maps found in the model.")
