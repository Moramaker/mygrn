import torch
import numpy as np
from config.config import config

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
    # 边界条件检查
    if num_nodes < 2 or num_neg_samples <= 0:
        return torch.empty((2, 0), dtype=torch.long)
    
    # 计算可能的负样本总数
    total_possible_edges = num_nodes * (num_nodes - 1) // 2
    num_pos_edges = edge_index.size(1)
    
    # 如果图是完全图，无法采样负样本
    if num_pos_edges >= total_possible_edges:
        return torch.empty((2, 0), dtype=torch.long)
    
    # 创建所有可能的边索引
    row = torch.arange(num_nodes).repeat(num_nodes)
    col = torch.repeat_interleave(torch.arange(num_nodes), num_nodes)
    
    # 过滤自环和重复边
    mask = row < col
    all_possible = torch.stack([row[mask], col[mask]], dim=0)
    
    # 移除正样本边
    if existing_edges is None:
        existing_edges = edge_index
    
    # 将正样本边转换为可哈希的元组集合
    pos_set = set(map(tuple, existing_edges.t().tolist()))
    
    # 创建负样本掩码
    neg_mask = [tuple(edge.tolist()) not in pos_set for edge in all_possible.t()]
    neg_edges = all_possible[:, neg_mask]
    
    # 确保采样数量不超过可用负样本
    num_available = neg_edges.size(1)
    if num_available == 0:
        return torch.empty((2, 0), dtype=torch.long)
    
    num_samples = min(num_neg_samples, num_available)
    
    # 随机采样负样本
    indices = torch.randperm(num_available)[:num_samples]
    return neg_edges[:, indices].contiguous()