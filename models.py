import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
import torch_geometric.nn as geom_nn

class BaseModel(nn.Module):
    """所有模型的基类，支持任务模式参数"""
    def forward(self, *args, **kwargs):
        """
        安全的前向传播方法
        支持'task'参数
        """
        task = kwargs.pop('task', None)  # 安全获取task参数
        return self._forward(*args, **kwargs, task=task)
    
    def _forward(self, *args, task=None, **kwargs):
        """由子类实现的具体前向逻辑"""
        raise NotImplementedError("子类必须实现_forward方法")

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
        input_dim = input_dim or getattr(config, 'ADJUSTER_INPUT_DIM', 192)  # 3 * 64
        hidden_dim = hidden_dim or getattr(config, 'ADJUSTER_HIDDEN_DIM', 256)
        output_dim = output_dim or getattr(config, 'ADJUSTER_OUTPUT_DIM', 128)
        final_dim = final_dim or getattr(config, 'OUTPUT_DIM', 64)
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, final_dim)
        )
        
        print(f"初始化后门调整器: 输入维度={input_dim}, 隐藏层={hidden_dim}, "
              f"中间输出={output_dim}, 最终输出={final_dim}")
    
    def forward(self, origin_rep, causal_rep, diff_rep):
        """前向传播"""
        combined = torch.cat([origin_rep, causal_rep, diff_rep], dim=-1)
        return self.model(combined)

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
        hidden_dim = hidden_dim or getattr(config, 'HIDDEN_DIM', 128)
        output_dim = output_dim or getattr(config, 'OUTPUT_DIM', 64)
        gat_heads = gat_heads or getattr(config, 'GAT_HEADS', 4)
        gat_dropout = gat_dropout or getattr(config, 'GAT_DROPOUT', 0.1)
        proj_layers = proj_layers or getattr(config, 'PROJ_LAYERS', 2)
        
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
        self.diff_proj = self._build_projection(output_dim, proj_layers)
        
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
        
        x = F.elu(self.gat1(x, data.edge_index))
        x = F.elu(self.gat2(x, data.edge_index))
        
        if view_type == "origin":
            return self.origin_proj(x)
        elif view_type == "causal":
            return self.causal_proj(x)
        elif view_type == "diffusion":
            return self.diff_proj(x)

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
        
        self.temp = temperature or getattr(config, 'TEMPERATURE', 0.07)
        self.contrast_weight = contrast_weight or getattr(config, 'CONTRAST_WEIGHT', 1.0)
        self.align_weight = align_weight or getattr(config, 'ALIGN_WEIGHT', 0.4)
        self.backdoor_weight = backdoor_weight or getattr(config, 'BACKDOOR_WEIGHT', 0.3)
        
        print(f"对比学习参数: 温度={self.temp}, 对比权重={self.contrast_weight}, "
              f"对齐权重={self.align_weight}, 后门权重={self.backdoor_weight}")

    def forward(self, data):
        """前向传播计算对比损失"""
        origin_rep = self.encoder(data, "origin")
        causal_rep = self.encoder(data, "causal")
        diff_rep = self.encoder(data, "diffusion")
        
        adjusted_rep = self.adjuster(origin_rep, causal_rep, diff_rep)
        
        contrast_loss = self._contrastive_loss(origin_rep, adjusted_rep)
        align_loss = self._alignment_loss(origin_rep, causal_rep, diff_rep)
        backdoor_loss = self._backdoor_loss(adjusted_rep, causal_rep)
        
        return contrast_loss, align_loss, backdoor_loss

    def _contrastive_loss(self, view1, view2):
        """计算对比损失"""
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        
        sim_matrix = torch.mm(view1, view2.t()) / self.temp
        labels = torch.arange(view1.size(0)).to(view1.device)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss * self.contrast_weight

    def _alignment_loss(self, origin, causal, diffusion):
        """计算视图对齐损失"""
        origin_causal = F.cosine_similarity(origin, causal, dim=1).mean()
        origin_diff = F.cosine_similarity(origin, diffusion, dim=1).mean()
        causal_diff = F.cosine_similarity(causal, diffusion, dim=1).mean()
        
        align_loss = 1.0 - (origin_causal + origin_diff + causal_diff) / 3.0
        
        return align_loss * self.align_weight

    def _backdoor_loss(self, adjusted, causal):
        """计算后门调整损失"""
        return F.mse_loss(adjusted, causal) * self.backdoor_weight

class TriViewModel(nn.Module):
    """端到端的三视图模型（完整修复版）"""
    def __init__(self, input_dim):
        super().__init__()
        output_dim = getattr(config, 'OUTPUT_DIM', 64)
        
        self.encoder = TriViewGATEncoder(input_dim)
        self.adjuster = BackdoorAdjuster(
            input_dim=3 * output_dim,
            hidden_dim=getattr(config, 'ADJUSTER_HIDDEN_DIM', 256),
            output_dim=getattr(config, 'ADJUSTER_OUTPUT_DIM', 128),
            final_dim=output_dim
        )
        self.contrast = TriViewContrast(
            encoder=self.encoder,
            adjuster=self.adjuster,
            temperature=getattr(config, 'TEMPERATURE', 0.07),
            contrast_weight=getattr(config, 'CONTRAST_WEIGHT', 1.0),
            align_weight=getattr(config, 'ALIGN_WEIGHT', 0.4),
            backdoor_weight=getattr(config, 'BACKDOOR_WEIGHT', 0.3)
        )
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