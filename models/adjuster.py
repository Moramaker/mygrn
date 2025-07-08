import torch
import torch.nn as nn
from config.config import config

class BackdoorAdjuster(nn.Module):
    """后门调整器模块（修复版）"""
    def __init__(self, input_dim=None, hidden_dim=None, output_dim=None, final_dim=None):
        """
        安全初始化后门调整器
        
        参数:
            input_dim: 输入维度 (默认从配置获取)
            hidden_dim: 隐藏层维度 (默认从配置获取)
            output_dim: 中间输出维度 (默认从配置获取)
            final_dim: 最终输出维度 (默认从配置获取)
        """
        super().__init__()
        # 安全获取配置值
        input_dim = input_dim or getattr(config, 'ADJUSTER_INPUT_DIM', 192)  # 3 * 64
        hidden_dim = hidden_dim or getattr(config, 'ADJUSTER_HIDDEN_DIM', 256)
        output_dim = output_dim or getattr(config, 'ADJUSTER_OUTPUT_DIM', 128)
        final_dim = final_dim or getattr(config, 'OUTPUT_DIM', 64)
        
        # 构建模型
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, final_dim)
        )
        
        # 记录参数
        print(f"初始化后门调整器: 输入维度={input_dim}, 隐藏层={hidden_dim}, "
              f"中间输出={output_dim}, 最终输出={final_dim}")
    
    def forward(self, origin_rep, causal_rep, diff_rep):
        """前向传播"""
        # 拼接三个视图的表示
        combined = torch.cat([origin_rep, causal_rep, diff_rep], dim=-1)
        return self.model(combined)