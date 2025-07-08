import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
from config.config import config

class TriViewGATEncoder(nn.Module):
    """三视图GAT编码器"""
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, gat_heads=None, gat_dropout=None, proj_layers=None):
        """
        初始化编码器
        
        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度 (可选)
            output_dim: 输出表示维度 (可选)
            gat_heads: GAT注意力头数 (可选)
            gat_dropout: GAT丢弃率 (可选)
            proj_layers: 投影层数 (可选)
        """
        super().__init__()
        # 安全获取配置值，提供默认值
        hidden_dim = hidden_dim or getattr(config, 'HIDDEN_DIM', 128)
        output_dim = output_dim or getattr(config, 'OUTPUT_DIM', 64)
        gat_heads = gat_heads or getattr(config, 'GAT_HEADS', 4)
        gat_dropout = gat_dropout or getattr(config, 'GAT_DROPOUT', 0.1)
        proj_layers = proj_layers or getattr(config, 'PROJ_LAYERS', 2)
        
        # 打印参数用于调试
        print(f"初始化GAT编码器: 输入维度={input_dim}, 隐藏维度={hidden_dim}, 输出维度={output_dim}")
        print(f"GAT参数: 头数={gat_heads}, 丢弃率={gat_dropout}, 投影层数={proj_layers}")
        
        # 共享的GAT编码器
        self.gat1 = geom_nn.GATConv(
            input_dim, 
            hidden_dim // gat_heads,
            heads=gat_heads,
            dropout=gat_dropout
        )
        self.gat2 = geom_nn.GATConv(
            hidden_dim, 
            output_dim,
            heads=1,
            dropout=gat_dropout
        )
        
        # 视图特定投影头
        self.origin_proj = self._build_projection(output_dim, proj_layers)
        self.causal_proj = self._build_projection(output_dim, proj_layers)
        self.diff_proj = self._build_projection(output_dim, proj_layers)
        
        # ID嵌入层
        self.id_embedding = nn.Embedding(10000, output_dim)

    def _build_projection(self, dim, num_layers):
        """构建视图投影模块"""
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dim, dim))
        return nn.Sequential(*layers)

    def forward(self, data, view_type="origin"):
        """
        前向传播
        
        参数:
            view_type: 
                "origin" - 原始视图
                "causal" - 因果视图
                "diffusion" - 扩散视图
                "id" - 基因ID嵌入
        """
        # 视图路由
        if view_type == "origin":
            x = data.x
        elif view_type == "causal":
            x = data.causal_view
        elif view_type == "diffusion":
            x = data.diffusion_view
        elif view_type == "id":
            return self.id_embedding(data.gene_ids)
        else:
            raise ValueError(f"无效视图类型: {view_type}")
        
        # GAT编码
        x = F.elu(self.gat1(x, data.edge_index))
        x = F.elu(self.gat2(x, data.edge_index))
        
        # 视图投影
        if view_type == "origin":
            return self.origin_proj(x)
        elif view_type == "causal":
            return self.causal_proj(x)
        elif view_type == "diffusion":
            return self.diff_proj(x)