"""
model.py — 最终模型架构
============================================================
配置: cdim192_ref
  - 单尺度 CNN k=21, cnn_dim=192
  - Cosine Classifier (temp=16.0)
  - 参数量约 8.5M

架构:
  ProtT5 [L, 1024]
    ├── mean pool → [1024] → MLP → [256]
    ├── N端 [:100] → Conv1d(1024, 192, k=21) → BN → ReLU → MaxPool → [192]
    └── C端 [-100:] → Conv1d(1024, 192, k=21) → BN → ReLU → MaxPool → [192]
  concat [256 + 192 + 192] = [640]
    → Linear(640, 256) → BN → ReLU → Dropout(0.6)
    → CosineClassifier(256, 6, temp=16.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, temperature=16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = nn.Parameter(torch.tensor(float(temperature)))

    def forward(self, x):
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        return self.scale * torch.mm(x_norm, w_norm.t())


class TerminalCNN(nn.Module):
    """N/C 端单尺度 CNN"""
    def __init__(self, d_in=1024, cnn_dim=192, kernel_size=21):
        super().__init__()
        self.conv = nn.Conv1d(d_in, cnn_dim, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(cnn_dim)

    def forward(self, x):
        """x: [B, D, L] → [B, cnn_dim]"""
        h = F.relu(self.bn(self.conv(x)))
        return torch.max(h, dim=-1)[0]


class SecretionModel(nn.Module):
    def __init__(self, d_emb=1024, cnn_dim=192, kernel_size=21,
                 hidden_global=256, hidden_fusion=256, dropout=0.6,
                 input_dropout=0.1, num_classes=6, cosine_temperature=16.0):
        super().__init__()
        self.input_dropout = nn.Dropout(input_dropout)

        # N/C 端 CNN
        self.cnn_N = TerminalCNN(d_emb, cnn_dim, kernel_size)
        self.cnn_C = TerminalCNN(d_emb, cnn_dim, kernel_size)

        # 全局 MLP
        self.global_mlp = nn.Sequential(
            nn.Linear(d_emb, hidden_global),
            nn.BatchNorm1d(hidden_global),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 融合投影
        fusion_dim = hidden_global + 2 * cnn_dim  # 256 + 192 + 192 = 640
        self.feat_proj = nn.Sequential(
            nn.Linear(fusion_dim, hidden_fusion),
            nn.BatchNorm1d(hidden_fusion),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 分类器
        self.classifier = CosineClassifier(hidden_fusion, num_classes, cosine_temperature)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, global_feat, n_seq, c_seq):
        global_feat = self.input_dropout(global_feat)

        g_repr = self.global_mlp(global_feat)
        n_repr = self.cnn_N(n_seq)
        c_repr = self.cnn_C(c_seq)

        fusion = torch.cat([g_repr, n_repr, c_repr], dim=-1)
        feat = self.feat_proj(fusion)
        logits = self.classifier(feat)

        return logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)