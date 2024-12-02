import torch
from torch.nn import functional as F


class Contrast(torch.nn.Module):
    def __init__(self, args, batch_size, temperature=0.5):
        super().__init__()
        device = torch.device(f"cuda:{args.local_rank}")
        self.batch_size = batch_size
        self.register_buffer("temp", torch.tensor(temperature).to(torch.device(f"cuda:{args.local_rank}")))
        self.register_buffer("neg_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())

    def forward(self, x_i, x_j):
        z_i = F.normalize(x_i, dim=1)
        z_j = F.normalize(x_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim, self.batch_size)
        sim_ji = torch.diag(sim, -self.batch_size)
        pos = torch.cat([sim_ij, sim_ji], dim=0)
        nom = torch.exp(pos / self.temp)
        denom = self.neg_mask * torch.exp(sim / self.temp)
        return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * self.batch_size)


import torch
import torch.nn.functional as F
class Loss:
    def __init__(self, batch_size, args, alpha1=1.0, alpha2=1.0, alpha3=1.0):
        self.alpha1 = alpha1  # 회전 손실 가중치
        self.alpha2 = alpha2  # 대조 손실 가중치
        self.alpha3 = alpha3  # 재구성 손실 가중치
        self.rot_loss = torch.nn.CrossEntropyLoss()
        self.reconstruction_loss_fn = torch.nn.MSELoss()  # 재구성 손실로 MSE 사용

    def rotation_loss(self, output_rot, target_rot):
        # target_rot의 차원이 2D 이상이면, 필요한 경우 차원 변환
        if target_rot.dim() > 1:
            target_rot = torch.argmax(target_rot, dim=1)
        return self.alpha1 * self.rot_loss(output_rot, target_rot)

    def contrastive_loss(self, output_contrastive, target_contrastive):
        # 대조 손실 계산
        if target_contrastive.dim() < output_contrastive.dim():
            target_contrastive = target_contrastive.unsqueeze(1).expand_as(output_contrastive)
        similarity = F.cosine_similarity(output_contrastive, target_contrastive, dim=-1)
        return self.alpha2 * (1 - similarity.mean())

    def reconstruction_loss(self, rec_x, target_x):
        return self.alpha3 * self.reconstruction_loss_fn(rec_x, target_x)

    def __call__(self, output_rot, target_rot, output_contrastive, target_contrastive, rec_x, original_x):
        # 최종 손실 계산
        rot_loss = self.rotation_loss(output_rot, target_rot)
        contrastive_loss = self.contrastive_loss(output_contrastive, target_contrastive)
        rec_loss = self.reconstruction_loss(rec_x, original_x)
        total_loss = rot_loss + contrastive_loss + rec_loss
        return total_loss


