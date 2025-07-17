import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from config import config
import logging

# 配置日志
logger = logging.getLogger(__name__)

class GeneRegulationDataset(InMemoryDataset):
    """基因调控网络数据集"""
    def __init__(self, root=None, transform=None):
        self.root = root or config.DATA_ROOT
        super().__init__(self.root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)
    
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
        gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
        gene_ids = torch.arange(len(genes))
        
        # 特征处理
        raw_features = expr_df.iloc[:, 1:].values
        features = torch.tensor(raw_features, dtype=torch.float32)
        features = F.normalize(features, dim=1)
        
        # 构建图结构
        edges = []
        for _, row in network_df.iterrows():
            gene1, gene2 = row["Gene1"], row["Gene2"]
            if gene1 in gene_to_idx and gene2 in gene_to_idx:
                edges.append([gene_to_idx[gene1], gene_to_idx[gene2]])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # 生成因果视图
        causal_features = self._generate_causal_view(features)
        
        # 创建数据对象
        graph_data = Data(
            original_features=features,      # 原始视图
            causal_features=causal_features, # 因果视图
            edge_index=edge_index,           # 基因互作网络
            gene_ids=gene_ids,               # 基因ID
            feature_dim=features.shape[1],   # 特征维度
            num_nodes=len(genes)             # 节点数量
        )
        
        # 保存处理结果
        torch.save(self.collate([graph_data]), self.processed_paths[0])
    
    def _generate_causal_view(self, features):
        """生成因果视图（基于反事实增强）"""
        # 识别少数样本（表达量低于平均值的基因）
        minority_mask = (features.mean(dim=1) < features.mean())
        causal_features = features.clone()
        
        if minority_mask.any():
            majority_indices = torch.where(~minority_mask)[0]
            minority_indices = torch.where(minority_mask)[0]
            
            # 计算少数样本与多数样本的距离矩阵
            dist_matrix = torch.cdist(features[minority_indices], 
                                     features[majority_indices])
            _, nn_indices = dist_matrix.min(dim=1)
            
            # 生成反事实样本
            synthetic = features[minority_indices] * (1 - config.CAUSAL_WEIGHT) + \
                        features[majority_indices[nn_indices]] * config.CAUSAL_WEIGHT
            
            causal_features[minority_indices] = synthetic
        
        return causal_features

def split_dataset(data):
    """
    将基因调控网络数据集划分为训练集、验证集和测试集
    
    参数:
        data: 完整的图数据对象 (Data)
        
    返回:
        train_data, val_data, test_data: 划分后的Data对象
        density: 网络密度
    """
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
    train_ratio = max(0.7 - density * 0.5, 0.5)  # 确保最小0.5
    test_ratio = min(0.15 + density * 0.25, 0.3)  # 确保最大0.3
    val_ratio = 1.0 - train_ratio - test_ratio
    
    if val_ratio < 0.05:
        train_ratio = train_ratio * 0.9
        test_ratio = test_ratio * 0.9
        val_ratio = 1.0 - train_ratio - test_ratio
    
    logger.info(f"根据密度 {density:.4f} 计算划分比例: "
                f"训练 {train_ratio:.2f}, 验证 {val_ratio:.2f}, 测试 {test_ratio:.2f}")
    
    return train_ratio, val_ratio, test_ratio

def create_subset(original_data, edge_index):
    """创建子集数据对象"""
    subset = Data()
    
    # 复制所有属性，除了边索引
    for key in original_data.keys():
        if key != 'edge_index':
            attr = getattr(original_data, key)
            if torch.is_tensor(attr):
                setattr(subset, key, attr.clone())
            else:
                setattr(subset, key, attr)
    
    subset.edge_index = edge_index
    return subset

def negative_sampling(edge_index, num_nodes, num_neg_samples, existing_edges=None):
    """
    安全高效的负采样
    
    参数:
        edge_index: 正样本边索引 (2, num_edges)
        num_nodes: 节点数量
        num_neg_samples: 需要采样的负样本数量
        existing_edges: 需要排除的边 (默认包含所有正样本边)
        
    返回:
        负样本边的索引张量 (2, num_neg_samples)
    """
    if num_nodes < 2 or num_neg_samples <= 0:
        return torch.empty((2, 0), dtype=torch.long)
    
    total_possible_edges = num_nodes * (num_nodes - 1) // 2
    num_pos_edges = edge_index.size(1)
    
    if num_pos_edges >= total_possible_edges:
        return torch.empty((2, 0), dtype=torch.long)
    
    # 使用更高效的负采样方法
    row, col = edge_index
    pos_set = set(zip(row.tolist(), col.tolist()))
    
    neg_edges = []
    while len(neg_edges) < num_neg_samples:
        u = torch.randint(0, num_nodes, (num_neg_samples,))
        v = torch.randint(0, num_nodes, (num_neg_samples,))
        
        for i in range(num_neg_samples):
            if u[i] != v[i] and (u[i].item(), v[i].item()) not in pos_set:
                neg_edges.append([u[i].item(), v[i].item()])
                if len(neg_edges) == num_neg_samples:
                    break
    
    if not neg_edges:
        return torch.empty((2, 0), dtype=torch.long)
    
    return torch.tensor(neg_edges, dtype=torch.long).t().contiguous()