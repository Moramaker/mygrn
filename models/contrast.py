import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import config

class TriViewContrast(nn.Module):
    """三视图对比学习模块（修复版）"""
    def __init__(self, encoder, adjuster, temperature=None, contrast_weight=None, 
                 align_weight=None, backdoor_weight=None):
        """
        安全初始化对比学习模块
        
        参数:
            encoder: 编码器
            adjuster: 后门调整器
            temperature: 温度参数 (默认从配置获取)
            contrast_weight: 对比损失权重 (默认从配置获取)
            align_weight: 对齐损失权重 (默认从配置获取)
            backdoor_weight: 后门损失权重 (默认从配置获取)
        """
        super().__init__()
        self.encoder = encoder
        self.adjuster = adjuster
        
        # 安全获取配置值
        self.temp = temperature or getattr(config, 'TEMPERATURE', 0.07)
        self.contrast_weight = contrast_weight or getattr(config, 'CONTRAST_WEIGHT', 1.0)
        self.align_weight = align_weight or getattr(config, 'ALIGN_WEIGHT', 0.4)
        self.backdoor_weight = backdoor_weight or getattr(config, 'BACKDOOR_WEIGHT', 0.3)
        
        # 记录参数
        print(f"对比学习参数: 温度={self.temp}, 对比权重={self.contrast_weight}, "
              f"对齐权重={self.align_weight}, 后门权重={self.backdoor_weight}")

    def forward(self, data):
        """前向传播计算对比损失"""
        # 获取不同视图的表示
        origin_rep = self.encoder(data, "origin")
        causal_rep = self.encoder(data, "causal")
        diff_rep = self.encoder(data, "diffusion")
        
        # 后门调整
        adjusted_rep = self.adjuster(origin_rep, causal_rep, diff_rep)
        
        # 计算对比损失
        contrast_loss = self._contrastive_loss(origin_rep, adjusted_rep)
        
        # 计算视图对齐损失
        align_loss = self._alignment_loss(origin_rep, causal_rep, diff_rep)
        
        # 计算后门调整损失
        backdoor_loss = self._backdoor_loss(adjusted_rep, causal_rep)
        
        return contrast_loss, align_loss, backdoor_loss

    def _contrastive_loss(self, view1, view2):
        """计算对比损失"""
        # 归一化表示
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.mm(view1, view2.t()) / self.temp
        
        # 计算对比损失
        labels = torch.arange(view1.size(0)).to(view1.device)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss * self.contrast_weight

    def _alignment_loss(self, origin, causal, diffusion):
        """计算视图对齐损失"""
        # 计算视图间相似度
        origin_causal = F.cosine_similarity(origin, causal, dim=1).mean()
        origin_diff = F.cosine_similarity(origin, diffusion, dim=1).mean()
        causal_diff = F.cosine_similarity(causal, diffusion, dim=1).mean()
        
        # 平均对齐损失
        align_loss = 1.0 - (origin_causal + origin_diff + causal_diff) / 3.0
        
        return align_loss * self.align_weight

    def _backdoor_loss(self, adjusted, causal):
        """计算后门调整损失"""
        return F.mse_loss(adjusted, causal) * self.backdoor_weight