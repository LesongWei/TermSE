import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class MultiScaleCNN(nn.Module):
    """
    对 N/C 端的 [B, D, L] 做多尺度卷积 + 全局最大池化
    输出一个 [B, out_dim] 的表示。
    """ #(3, 5, 9 21)
    def __init__(self, d_in=1024, cnn_dim=256, kernel_sizes=(5, 9, 21)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d_in, cnn_dim, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(cnn_dim) for _ in kernel_sizes
        ])
        self.kernel_sizes = kernel_sizes
        self.out_dim = cnn_dim * len(kernel_sizes)

    def forward(self, x):
        """
        x: [B, D, L]
        """
        feats = []
        for conv, bn in zip(self.convs, self.bns):
            h = conv(x)                # [B, cnn_dim, L]
            h = bn(h)
            h = F.relu(h)
            # Global max pooling over length dimension
            h = torch.max(h, dim=-1)[0]   # [B, cnn_dim]
            # h = torch.mean(h, dim=-1)
            feats.append(h)

        # [B, cnn_dim * n_kernels]
        out = torch.cat(feats, dim=-1)
        # out = torch.stack(feats, dim=1)
        return out


class MotifNet(nn.Module):
    """
    融合：
      - 序列全局 embedding (global_feat) [B, 1024]
      - N/C 端多尺度 CNN 表征 [B, 1024, W]
      - 结构特征 s_feat [B, F]
    的 6 分类模型。
    """
    def __init__(
        self,
        d_emb=1024,
        window_size=200,
        struct_dim=10,      # 你现在的结构特征维度
        num_classes=6,
        cnn_dim=256,
        hidden_global=256,
        hidden_struct=32,
        hidden_fusion=512,
        dropout=0.3,
    ):
        super().__init__()

        self.d_emb = d_emb
        self.window_size = window_size
        self.struct_dim = struct_dim
        self.num_classes = num_classes

        # ====== N/C 端多尺度 CNN（共享权重）======
        self.cnn_branch = MultiScaleCNN(d_in=d_emb, cnn_dim=cnn_dim)
        cnn_out_dim = self.cnn_branch.out_dim  # cnn_dim * n_kernels

        # ====== Global embedding 分支 ======
        self.global_mlp = nn.Sequential(
            nn.Linear(d_emb, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_global),
            nn.ReLU(),
        )

        # ====== 结构特征分支 ======
        self.struct_mlp = nn.Sequential(
            nn.Linear(struct_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_struct),
            nn.ReLU(),
        )

        # ====== 融合分类器 ======
        fusion_in_dim = hidden_global + 2 * cnn_out_dim + hidden_struct
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_fusion),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_fusion, hidden_fusion // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_fusion // 2, num_classes),
        )

    def forward(self, global_feat, n_seq, c_seq, s_feat):
        """
        参数:
          global_feat: [B, 1024]
          n_seq:       [B, 1024, W]
          c_seq:       [B, 1024, W]
          s_feat:      [B, struct_dim]

        返回:
          logits:      [B, num_classes]
        """

        # 1) N / C 端 CNN
        n_repr = self.cnn_branch(n_seq)    # [B, cnn_out_dim]
        c_repr = self.cnn_branch(c_seq)    # [B, cnn_out_dim]

        # 2) Global embedding 分支
        g_repr = self.global_mlp(global_feat)   # [B, hidden_global]

        # 3) 结构特征分支
        s_repr = self.struct_mlp(s_feat)        # [B, hidden_struct]

        # 4) 融合
        fusion = torch.cat([g_repr, n_repr, c_repr, s_repr], dim=-1)

        logits = self.classifier(fusion)        # [B, num_classes]
        return logits
    




class AdMotifNet(nn.Module):
    """
    融合：
      - 序列全局 embedding (global_feat) [B, 1024]
      - N/C 端多尺度 CNN 表征 [B, 1024, W]
      - 结构特征 s_feat [B, F]
    的 6 分类模型。
    """
    def __init__(
        self,
        d_emb=1024,
        window_size=200,
        struct_dim=256,      # 你现在的结构特征维度
        num_classes=6,
        cnn_dim=256, #256
        hidden_global=256,  #256
        hidden_struct=64,
        hidden_fusion=512, #512
        dropout=0.5,
    ):
        super().__init__()

        self.d_emb = d_emb
        self.window_size = window_size
        self.struct_dim = struct_dim
        self.num_classes = num_classes

        # ====== N/C 端多尺度 CNN（共享权重）======
        self.cnn_N = MultiScaleCNN(d_in=d_emb, cnn_dim=cnn_dim ,kernel_sizes=([21])) # 5, 9, 21
        self.cnn_C = MultiScaleCNN(d_in=d_emb, cnn_dim=cnn_dim, kernel_sizes=([21]))
        cnn_out_dim = self.cnn_N.out_dim  # cnn_dim * n_kernels

        # ====== Global embedding 分支 ======
        self.global_mlp = nn.Sequential(
            nn.Linear(d_emb, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_global),
            nn.ReLU(),
        )

        # ====== 结构特征分支 ======
        self.struct_mlp = nn.Sequential(
            nn.Linear(struct_dim, struct_dim//2),
            nn.BatchNorm1d(struct_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(struct_dim//2, 64),
            nn.ReLU(),
        )

        # ====== 融合分类器 ======
        # fusion_in_dim = hidden_global  + 64 #+ 2 * cnn_out_dim
        fusion_in_dim = hidden_global  + 2 * cnn_out_dim + hidden_struct

        
        self.classifier1 = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_fusion),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_fusion, hidden_fusion//2),
        )
        
        self.classifier2 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_fusion//2, num_classes),
        )

    def forward(self, global_feat, n_seq, c_seq, s_feat):

        """
        参数:
          global_feat: [B, 1024]
          n_seq:       [B, 1024, W]
          c_seq:       [B, 1024, W]
          s_feat:      [B, struct_dim]

        返回:
          logits:      [B, num_classes]
        """
        
        n_repr = self.cnn_N(n_seq)    # [B, cnn_out_dim]
        c_repr = self.cnn_C(c_seq)    # [B, cnn_out_dim]

        # 2) Global embedding 分支
        g_repr = self.global_mlp(global_feat)   # [B, hidden_global]

        # 3) 结构特征分支
        
        s_repr = self.struct_mlp(s_feat)        # [B, hidden_struct]
        
        # 4) 融合
        fusion = torch.cat([g_repr, n_repr, c_repr, s_repr], dim=-1) #, s_repr
        
        subfusion = self.classifier1(fusion)

        # feats = self.projector(fusion)
        
        logits = self.classifier2(subfusion)

        # logits = self.classifier(fusion)        # [B, num_classes]
        return logits, g_repr



class VIBMotifNet(nn.Module):
    """
    融合：
      - 序列全局 embedding (global_feat) [B, 1024]
      - N/C 端多尺度 CNN 表征 [B, 1024, W]
      - 结构特征 s_feat [B, F]
    的 6 分类模型。
    """
    def __init__(
        self,
        d_emb=1024,
        window_size=200,
        struct_dim=256,      # 你现在的结构特征维度
        num_classes=6,
        cnn_dim=256,
        hidden_global=256,
        hidden_struct=64,
        hidden_fusion=512,
        dropout=0.4,
    ):
        super().__init__()

        self.d_emb = d_emb
        self.window_size = window_size
        self.struct_dim = struct_dim
        self.num_classes = num_classes

        # ====== N/C 端多尺度 CNN（共享权重）======
        self.cnn_N = MultiScaleCNN(d_in=d_emb, cnn_dim=cnn_dim ,kernel_sizes=(5, 9, 21))
        self.cnn_C = MultiScaleCNN(d_in=d_emb, cnn_dim=cnn_dim, kernel_sizes=(5, 9, 21))
        cnn_out_dim = self.cnn_N.out_dim  # cnn_dim * n_kernels

        # ====== Global embedding 分支 ======
        self.global_mlp = nn.Sequential(
            nn.Linear(d_emb, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_global),
            nn.ReLU(),
        )

        # ====== 结构特征分支 ======
        self.struct_mlp = nn.Sequential(
            nn.Linear(struct_dim, struct_dim//2),
            nn.BatchNorm1d(struct_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(struct_dim//2, 64),
            nn.ReLU(),
        )

        # ====== 融合分类器 ======
        fusion_in_dim = hidden_global + 2 * cnn_out_dim

        self.encoder_mu = nn.Linear(hidden_fusion, 256)  # 均值
        self.encoder_logvar = nn.Linear(hidden_fusion, 256) # 对数方差 (数值更稳定)
        
        self.classifier1 = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_fusion),
        )
        
        self.classifier2 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes), # 输入维度变为 z_dim (256)
        )
        
        # self.projector = nn.Sequential(
        #     nn.Linear(fusion_in_dim, 128)
        # )

    def reparameterize(self, mu, logvar):
        """重参数化技巧：z = mu + std * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, global_feat, n_seq, c_seq, s_feat):
        """
        参数:
          global_feat: [B, 1024]
          n_seq:       [B, 1024, W]
          c_seq:       [B, 1024, W]
          s_feat:      [B, struct_dim]

        返回:
          logits:      [B, num_classes]
        """
        
      

        # 1) N / C 端 CNN
        n_repr = self.cnn_N(n_seq)    # [B, cnn_out_dim]
        c_repr = self.cnn_C(c_seq)    # [B, cnn_out_dim]

        # 2) Global embedding 分支
        g_repr = self.global_mlp(global_feat)   # [B, hidden_global]

        # 3) 结构特征分支
        
        s_repr = self.struct_mlp(s_feat)        # [B, hidden_struct]
        
        # 4) 融合
        fusion = torch.cat([g_repr, n_repr, c_repr], dim=-1)

        
        subfusion = self.classifier1(fusion)

        mu = self.encoder_mu(subfusion)
        logvar = self.encoder_logvar(subfusion)

        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        # feats = self.projector(fusion)
        
        logits = self.classifier2(z)

        # logits = self.classifier(fusion)        # [B, num_classes]
        return logits, z, mu, logvar