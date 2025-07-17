import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
import torch_geometric.nn as geom_nn

class BaseModel(nn.Module):
    """所有模型的基类，支持任务模式参数"""
    def forward(self, *args, **kwargs):
        task = kwargs.pop('task', None)
        return self._forward(*args, **kwargs, task=task)
    
    def _forward(self, *args, task=None, **kwargs):
        raise NotImplementedError("子类必须实现_forward方法")

class BiViewGATEncoder(nn.Module):
    """二视图GAT编码器"""
    def __init__(self, input_dim):
        super().__init__()
        hidden_dim = config.HIDDEN_DIM
        output_dim = config.OUTPUT_DIM
        gat_heads = config.GAT_HEADS
        gat_dropout = config.GAT_DROPOUT
        proj_layers = config.PROJ_LAYERS
        
        print(f"初始化GAT编码器: 输入维度={input_dim}, 隐藏维度={hidden_dim}, 输出维度={output_dim}")
        print(f"GAT参数: 头数={gat_heads}, 丢弃率={gat_dropout}, 投影层数={proj_layers}")
        
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
        
        self.origin_proj = self._build_projection(output_dim, proj_layers)
        self.causal_proj = self._build_projection(output_dim, proj_layers)
        
        self.id_embedding = nn.Embedding(10000, output_dim)

    def _build_projection(self, dim, num_layers):
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dim, dim))
        return nn.Sequential(*layers)

    def forward(self, data, view_type="origin"):
        if view_type == "origin":
            x = data.x
        elif view_type == "causal":
            x = data.causal_view
        elif view_type == "id":
            return self.id_embedding(data.gene_ids)
        else:
            raise ValueError(f"无效视图类型: {view_type}")
        
        x = F.elu(self.gat1(x, data.edge_index))
        x = F.elu(self.gat2(x, data.edge_index))
        
        if view_type == "origin":
            return self.origin_proj(x)
        elif view_type == "causal":
            return self.causal_proj(x)

class BiViewContrast(nn.Module):
    """二视图对比学习模块"""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.temp = config.TEMPERATURE
        self.contrast_weight = config.CONTRAST_WEIGHT
        self.align_weight = config.ALIGN_WEIGHT
        
        print(f"对比学习参数: 温度={self.temp}, 对比权重={self.contrast_weight}, 对齐权重={self.align_weight}")

    def forward(self, data):
        origin_rep = self.encoder(data, "origin")
        causal_rep = self.encoder(data, "causal")
        
        contrast_loss = self._contrastive_loss(origin_rep, causal_rep)
        align_loss = self._alignment_loss(origin_rep, causal_rep)
        
        return contrast_loss, align_loss

    def _contrastive_loss(self, view1, view2):
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        
        sim_matrix = torch.mm(view1, view2.t()) / self.temp
        labels = torch.arange(view1.size(0)).to(view1.device)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss * self.contrast_weight

    def _alignment_loss(self, origin, causal):
        origin_causal = F.cosine_similarity(origin, causal, dim=1).mean()
        align_loss = 1.0 - origin_causal
        return align_loss * self.align_weight

class BiViewModel(nn.Module):
    """端到端的二视图模型"""
    def __init__(self, input_dim):
        super().__init__()
        output_dim = config.OUTPUT_DIM
        
        self.encoder = BiViewGATEncoder(input_dim)
        self.contrast = BiViewContrast(self.encoder)
        self.predictor = nn.Sequential(
            nn.Linear(2 * output_dim, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM, 1)
        )
    
    def forward(self, data, task="contrast"):
        if task == "contrast":
            return self.contrast(data)
        elif task == "prediction":
            origin_rep = self.encoder(data, "origin")
            causal_rep = self.encoder(data, "causal")
            combined = torch.cat([origin_rep, causal_rep], dim=-1)
            return self.predictor(combined)
        else:
            raise ValueError("无效任务模式")