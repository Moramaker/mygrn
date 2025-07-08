import torch
import torch.nn as nn
from config.config import config
from models.encoder import TriViewGATEncoder
from models.adjuster import BackdoorAdjuster
from models.contrast import TriViewContrast

class TriViewModel(nn.Module):
    """端到端的三视图模型（完整修复版）"""
    def __init__(self, input_dim):
        super().__init__()
        # 安全获取配置值
        output_dim = getattr(config, 'OUTPUT_DIM', 64)
        
        # 初始化编码器
        self.encoder = TriViewGATEncoder(input_dim)
        
        # 初始化后门调整器
        self.adjuster = BackdoorAdjuster(
            input_dim=3 * output_dim,
            hidden_dim=getattr(config, 'ADJUSTER_HIDDEN_DIM', 256),
            output_dim=getattr(config, 'ADJUSTER_OUTPUT_DIM', 128),
            final_dim=output_dim
        )
        
        # 初始化对比学习模块 - 传递所有必需参数
        self.contrast = TriViewContrast(
            encoder=self.encoder,
            adjuster=self.adjuster,
            temperature=getattr(config, 'TEMPERATURE', 0.07),
            contrast_weight=getattr(config, 'CONTRAST_WEIGHT', 1.0),
            align_weight=getattr(config, 'ALIGN_WEIGHT', 0.4),
            backdoor_weight=getattr(config, 'BACKDOOR_WEIGHT', 0.3)
        )
        
        # 下游预测器
        self.predictor = nn.Sequential(
            nn.Linear(3 * output_dim, getattr(config, 'HIDDEN_DIM', 128)),
            nn.ReLU(),
            nn.Linear(getattr(config, 'HIDDEN_DIM', 128), 1)
        )
    
    def forward(self, data, task="contrast"):
        """
        前向传播
        
        参数:
            task: 
                "contrast" - 返回对比损失
                "prediction" - 返回预测结果
        """
        if task == "contrast":
            return self.contrast(data)
        elif task == "prediction":
            origin_rep = self.encoder(data, "origin")
            causal_rep = self.encoder(data, "causal")
            diff_rep = self.encoder(data, "diffusion")
            combined = torch.cat([origin_rep, causal_rep, diff_rep], dim=-1)
            return self.predictor(combined)
        else:
            raise ValueError("无效任务模式")