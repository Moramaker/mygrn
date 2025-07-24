import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from config import config
import logging

# 配置日志
logger = logging.getLogger(__name__)

# === 添加CVAE相关实现 ===
class CVAEGeneGenerator(nn.Module):
    """条件变分自编码器，用于生成基因表达的反事实样本"""
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 为条件输入（平均表达量）
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        
        self.mean_fc = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar_fc = nn.Linear(hidden_dim // 2, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim // 2),  # +1 为条件输入
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x, c):
        """将输入和条件编码为潜在空间的均值和方差"""
        inputs = torch.cat([x, c.unsqueeze(1)], dim=1)
        h = self.encoder(inputs)
        return self.mean_fc(h), self.logvar_fc(h)
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧，从正态分布中采样"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        """将潜在变量和条件解码为重构样本"""
        inputs = torch.cat([z, c.unsqueeze(1)], dim=1)
        return self.decoder(inputs)
    
    def forward(self, x, c):
        """模型前向传播"""
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
    
    def generate_counterfactual(self, x, target_c):
        """生成反事实样本：给定原始样本x，生成目标条件target_c下的样本"""
        with torch.no_grad():
            # 使用原始样本的条件进行编码
            original_c = x.mean(dim=1)
            mu, logvar = self.encode(x, original_c)
            z = self.reparameterize(mu, logvar)
            # 使用目标条件进行解码
            return self.decode(z, target_c)


class GeneRegulationDataset(InMemoryDataset):
    """基因调控网络数据集"""
    def __init__(self, root=None, transform=None, use_cvae=True):
        self.root = root or config.DATA_ROOT
        self.use_cvae = use_cvae  # 是否使用CVAE生成反事实样本
        super().__init__(self.root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_dir(self):
        return os.path.join(
            self.root, "raw", 
            f"{config.NETWORK_TYPE} Dataset", 
            config.CELL_TYPE, 
            f"TFs+{config.GENE_NUM}"
        )
    
    @property
    def processed_dir(self):
        return os.path.join(
            self.root, "processed", 
            config.NETWORK_TYPE, 
            config.CELL_TYPE, 
            f"{config.GENE_NUM}"
        )
    
    @property
    def raw_file_names(self):
        return ["BL--network.csv", "BL--ExpressionData.csv"]
    
    @property
    def processed_file_names(self):
        return [f"grn_data_{config.NETWORK_TYPE}_{config.CELL_TYPE}_{config.GENE_NUM}.pt"]
    
    def process(self):
        # 加载表达数据
        expr_path = os.path.join(self.raw_dir, "BL--ExpressionData.csv")
        expr_df = pd.read_csv(expr_path)
        
        # 加载网络数据
        network_path = os.path.join(self.raw_dir, "BL--network.csv")
        network_df = pd.read_csv(network_path, names=["Gene1", "Gene2"])
        
        # 基因名称映射
        genes = expr_df.iloc[:, 0].tolist()
        name_to_idx = {gene: idx for idx, gene in enumerate(genes)}
        gene_ids = torch.arange(len(genes))
        
        # 特征处理
        features = torch.tensor(expr_df.iloc[:, 1:].values, dtype=torch.float32)
        features = normalize_expression_data(features)
        
        # 构建图结构
        edges = []
        for _, row in network_df.iterrows():
            gene1, gene2 = row["Gene1"], row["Gene2"]
            if gene1 in name_to_idx and gene2 in name_to_idx:
                edges.append([name_to_idx[gene1], name_to_idx[gene2]])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # 生成因果视图（使用CVAE或原始方法）
        causal_view = self._generate_causal_view(features)
        
        # 创建数据对象
        graph_data = Data(
            x=features,              # 原始视图
            causal_view=causal_view,  # 因果视图
            edge_index=edge_index,    # 基因互作网络
            gene_ids=gene_ids,        # 基因ID
            feature_dim=features.shape[1],  # 特征维度
            num_nodes=len(genes)       # 节点数量
        )
        
        # 保存处理结果
        torch.save(self.collate([graph_data]), self.processed_paths[0])
    
    def _train_cvae(self, features):
        """训练CVAE模型（仅在需要时训练）"""
        input_dim = features.shape[1]
        model = CVAEGeneGenerator(
            input_dim=input_dim,
            hidden_dim=config.CVAE_HIDDEN_DIM,
            latent_dim=config.CVAE_LATENT_DIM
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=config.CVAE_LR)
        
        # 条件变量：每个样本的平均表达量
        conditions = features.mean(dim=1)
        
        # 训练循环
        model.train()
        for epoch in range(config.CVAE_EPOCHS):
            # 前向传播
            recon_x, mu, logvar = model(features, conditions)
            
            # 计算损失
            recon_loss = F.mse_loss(recon_x, features)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss /= features.size(0) * input_dim  # 标准化KL损失
            
            # 总损失（平衡重构和KL散度）
            loss = recon_loss + config.CVAE_KL_WEIGHT * kl_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 日志输出
            if (epoch + 1) % 50 == 0:
                logger.info(f"CVAE训练 Epoch [{epoch+1}/{config.CVAE_EPOCHS}], 损失: {loss.item():.4f}")
        
        return model
    
    def _generate_causal_view(self, features):
        """生成因果视图（基于CVAE的反事实增强）"""
        # 识别少数样本（表达量低于平均值的基因）
        minority_mask = (features.mean(dim=1) < features.mean())
        causal_view = features.clone()
        
        if not minority_mask.any():
            return causal_view
        
        # 如果不使用CVAE，使用原始插值方法
        if not self.use_cvae:
            majority_indices = torch.where(~minority_mask)[0]
            minority_indices = torch.where(minority_mask)[0]
            
            # 计算最近邻
            dist_matrix = torch.cdist(features[minority_indices], features[majority_indices])
            _, nn_indices = dist_matrix.min(dim=1)
            
            # 插值生成反事实样本
            synthetic = features[minority_indices] * (1 - config.ALIGN_WEIGHT) + \
                        features[majority_indices[nn_indices]] * config.ALIGN_WEIGHT
            causal_view[minority_indices] = synthetic
            return causal_view
        
        # 使用CVAE生成反事实样本
        logger.info(f"使用CVAE生成反事实样本，少数样本数量: {minority_mask.sum().item()}")
        
        # 训练CVAE模型
        cvae_model = self._train_cvae(features)
        
        # 生成目标条件（多数样本的平均表达量）
        majority_mean = features[~minority_mask].mean(dim=1).mean()
        target_conditions = torch.full_like(features[minority_mask].mean(dim=1), majority_mean)
        
        # 生成反事实样本
        cvae_model.eval()
        counterfactuals = cvae_model.generate_counterfactual(
            features[minority_mask], 
            target_conditions
        )
        
        # 替换少数样本
        causal_view[minority_mask] = counterfactuals
        return causal_view

def split_dataset(data):
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    num_nodes = data.num_nodes
    max_possible = num_nodes * (num_nodes - 1) // 2
    density = num_edges / max_possible if max_possible > 0 else 0.0
    
    if num_edges == 0:
        logger.warning("图中没有边，返回空数据集")
        empty_data = data.clone()
        empty_data.edge_index = torch.empty((2, 0), dtype=torch.long)
        return empty_data, empty_data.clone(), empty_data.clone(), density
    
    train_ratio, val_ratio, test_ratio = calculate_split_ratios(density)
    indices = torch.randperm(num_edges)
    train_end = int(train_ratio * num_edges)
    val_end = train_end + int(val_ratio * num_edges)
    
    train_mask = indices[:train_end]
    val_mask = indices[train_end:val_end]
    test_mask = indices[val_end:]
    
    train_data = create_subset(data, edge_index[:, train_mask])
    val_data = create_subset(data, edge_index[:, val_mask])
    test_data = create_subset(data, edge_index[:, test_mask])
    
    return train_data, val_data, test_data, density, (train_ratio, val_ratio, test_ratio)

def calculate_split_ratios(density):
    """根据网络密度计算数据集划分比例"""
    x = (torch.tensor(density, dtype=torch.float32) - 0.2) * 10
    sigmoid_value = torch.sigmoid(x).item()
    
    # 基础比例配置
    base_train = 0.7
    base_test = 0.2
    
    # 根据sigmoid值调整比例
    train_ratio = base_train - (sigmoid_value * 0.3)
    test_ratio = base_test + (sigmoid_value * 0.15)
    val_ratio = 1.0 - train_ratio - test_ratio
    
    # 确保各比例在合理范围内
    train_ratio = max(min(train_ratio, 0.8), 0.4)
    test_ratio = max(min(test_ratio, 0.4), 0.1)
    val_ratio = max(min(val_ratio, 0.2), 0.1)
    
    # 确保验证集比例不低于15%
    if val_ratio < 0.15:
        shortfall = 0.15 - val_ratio
        train_ratio -= shortfall * (train_ratio / (train_ratio + test_ratio))
        test_ratio -= shortfall * (test_ratio / (train_ratio + test_ratio))
        val_ratio = 0.15
    
    return train_ratio, val_ratio, test_ratio

def create_subset(original_data, edge_index):
    """创建子集数据对象"""
    subset = Data()
    for key in original_data.keys():
        if key != 'edge_index':
            attr = getattr(original_data, key)
            if torch.is_tensor(attr):
                setattr(subset, key, attr.clone())
            else:
                setattr(subset, key, attr)
    
    subset.edge_index = edge_index
    return subset

def negative_sampling(edge_index, num_nodes, num_neg_samples, density=None, existing_edges=None):
    if num_nodes < 2 or num_neg_samples <= 0:
        return torch.empty((2, 0), dtype=torch.long)
    
    # 使用提供的密度或计算密度
    if density is None:
        if existing_edges is None:
            existing_edges = edge_index
        density = existing_edges.size(1) / (num_nodes * (num_nodes - 1) / 2)
    
    # 根据密度调整采样策略
    if density < 0.1:  # 低密度网络
        # 对低密度网络使用拒绝采样，增加尝试次数
        pos_set = set(map(tuple, existing_edges.t().tolist()))
        neg_edges = []
        max_tries = min(num_neg_samples * 20, 1000000)  # 最大尝试次数
        
        while len(neg_edges) < num_neg_samples and len(pos_set) < (num_nodes * (num_nodes - 1) // 2):
            # 随机生成边
            u = torch.randint(0, num_nodes, (1,)).item()
            v = torch.randint(0, num_nodes, (1,)).item()
            
            # 确保 u < v 且不是正边
            if u < v and (u, v) not in pos_set:
                neg_edges.append((u, v))
        
        if neg_edges:
            return torch.tensor(neg_edges, dtype=torch.long).t().contiguous()
        else:
            return torch.empty((2, 0), dtype=torch.long)
    
    else:  # 高密度网络
        # 对高密度网络使用优化的全枚举法
        row = torch.arange(num_nodes).repeat(num_nodes)
        col = torch.repeat_interleave(torch.arange(num_nodes), num_nodes)
        mask = row < col
        all_possible = torch.stack([row[mask], col[mask]], dim=0)
        
        pos_set = set(map(tuple, existing_edges.t().tolist()))
        neg_mask = [tuple(edge.tolist()) not in pos_set for edge in all_possible.t()]
        neg_edges = all_possible[:, neg_mask]
        
        num_available = neg_edges.size(1)
        if num_available == 0:
            return torch.empty((2, 0), dtype=torch.long)
        
        num_samples = min(num_neg_samples, num_available)
        indices = torch.randperm(num_available)[:num_samples]
        return neg_edges[:, indices].contiguous()

def normalize_expression_data(expression_data):
    # 对数变换
    log_transformed = torch.log1p(expression_data)
    
    # Z-score标准化（行归一化）
    mean = log_transformed.mean(dim=1, keepdim=True)
    std = log_transformed.std(dim=1, keepdim=True)
    std = torch.clamp(std, min=1e-7) 
    
    normalized = (log_transformed - mean) / std
    return normalized