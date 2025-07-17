import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
import torch_geometric.nn as geom_nn

class BaseModel(nn.Module):
    """所有模型的基类，支持任务模式参数"""
    def forward(self, *args, **kwargs):
        """安全的前向传播方法，支持'task'参数"""
        task = kwargs.pop('task', None)  # 安全获取task参数
        return self._forward(*args, **kwargs, task=task)
    
    def _forward(self, *args, task=None, **kwargs):
        """由子类实现的具体前向逻辑"""
        raise NotImplementedError("子类必须实现_forward方法")

class CausalAdjuster(nn.Module):
    """因果调整器模块"""
    def __init__(self, input_dim=None, hidden_dim=None, output_dim=None):
        """初始化因果调整器"""
        super().__init__()
        input_dim = input_dim or 2 * config.ENCODER_OUTPUT_DIM
        hidden_dim = hidden_dim or config.ADJUSTER_HIDDEN_DIM
        output_dim = output_dim or config.ADJUSTER_OUTPUT_DIM
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, config.ENCODER_OUTPUT_DIM)
        )
    
    def forward(self, original_repr, causal_repr):
        """前向传播"""
        combined = torch.cat([original_repr, causal_repr], dim=-1)
        return self.model(combined)

class DualViewGATEncoder(nn.Module):
    """双视图GAT编码器"""
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, gat_heads=None, gat_dropout=None):
        """初始化编码器"""
        super().__init__()
        hidden_dim = hidden_dim or config.HIDDEN_DIM
        output_dim = output_dim or config.ENCODER_OUTPUT_DIM
        gat_heads = gat_heads or config.GAT_HEADS
        gat_dropout = gat_dropout or config.GAT_DROPOUT
        
        # 使用更高效的GATv2Conv
        self.gat1 = geom_nn.GATv2Conv(
            input_dim, 
            hidden_dim // gat_heads,
            heads=gat_heads,
            dropout=gat_dropout,
            add_self_loops=False  # 减少不必要的计算
        )
        self.gat2 = geom_nn.GATv2Conv(
            hidden_dim, 
            output_dim,
            heads=1,
            dropout=gat_dropout,
            add_self_loops=False  # 减少不必要的计算
        )
        
        self.original_projection = nn.Linear(output_dim, output_dim)
        self.causal_projection = nn.Linear(output_dim, output_dim)

    def forward(self, data, view_type="original"):
        """
        前向传播
        
        参数:
            view_type: "original" 或 "causal"
        """
        if view_type == "original":
            x = data.original_features
        elif view_type == "causal":
            x = data.causal_features
        else:
            raise ValueError(f"无效视图类型: {view_type}")
        
        x = F.elu(self.gat1(x, data.edge_index))
        x = F.elu(self.gat2(x, data.edge_index))
        
        if view_type == "original":
            return self.original_projection(x)
        elif view_type == "causal":
            return self.causal_projection(x)

class DualViewContrastiveLearning(nn.Module):
    """双视图对比学习模块"""
    def __init__(self, encoder, adjuster, temperature=None, contrastive_weight=None, 
                 alignment_weight=None, causal_weight=None):
        """初始化对比学习模块"""
        super().__init__()
        self.encoder = encoder
        self.adjuster = adjuster
        
        self.temperature = temperature or config.TEMPERATURE
        self.contrastive_weight = contrastive_weight or config.CONTRASTIVE_WEIGHT
        self.alignment_weight = alignment_weight or config.ALIGNMENT_WEIGHT
        self.causal_weight = causal_weight or config.CAUSAL_WEIGHT

    def forward(self, data):
        """前向传播计算对比损失"""
        original_repr = self.encoder(data, "original")
        causal_repr = self.encoder(data, "causal")
        
        adjusted_repr = self.adjuster(original_repr, causal_repr)
        
        contrastive_loss = self._contrastive_loss(original_repr, adjusted_repr)
        alignment_loss = self._alignment_loss(original_repr, causal_repr)
        causal_loss = self._causal_loss(adjusted_repr, causal_repr)
        
        return contrastive_loss, alignment_loss, causal_loss

    def _contrastive_loss(self, view1, view2):
        """计算对比损失（使用批量加速）"""
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        
        # 使用矩阵运算替代循环
        sim_matrix = torch.mm(view1, view2.t()) / self.temperature
        labels = torch.arange(view1.size(0)).to(view1.device)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss * self.contrastive_weight

    def _alignment_loss(self, original, causal):
        """计算视图对齐损失"""
        return 1.0 - F.cosine_similarity(original, causal, dim=1).mean() * self.alignment_weight

    def _causal_loss(self, adjusted, causal):
        """计算因果调整损失"""
        return F.mse_loss(adjusted, causal) * self.causal_weight

class DualViewModel(nn.Module):
    """端到端的双视图模型"""
    def __init__(self, input_dim):
        super().__init__()
        
        self.encoder = DualViewGATEncoder(input_dim)
        self.adjuster = CausalAdjuster()
        self.contrastive_learner = DualViewContrastiveLearning(
            encoder=self.encoder,
            adjuster=self.adjuster
        )
        self.predictor = nn.Sequential(
            nn.Linear(2 * config.ENCODER_OUTPUT_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM, 1)
        )
    
    def forward(self, data, task="contrastive"):
        """
        前向传播
        
        参数:
            task: "contrastive" 或 "prediction"
        """
        if task == "contrastive":
            return self.contrastive_learner(data)
        elif task == "prediction":
            original_repr = self.encoder(data, "original")
            causal_repr = self.encoder(data, "causal")
            combined = torch.cat([original_repr, causal_repr], dim=-1)
            return self.predictor(combined)
        else:
            raise ValueError("无效任务模式")