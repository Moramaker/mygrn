import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from config.config import config
import logging

# 配置日志
logger = logging.getLogger(__name__)

def split_dataset(data):
    """
    将基因调控网络数据集划分为训练集、验证集和测试集
    
    参数:
        data: 完整的图数据对象 (Data)
        
    返回:
        train_data, val_data, test_data: 划分后的Data对象
        density: 网络密度
    """
    # 验证输入
    if not hasattr(data, 'edge_index'):
        raise AttributeError("输入数据缺少 'edge_index' 属性")
    
    edge_index = data.edge_index
    
    # 检查边索引维度
    if edge_index.dim() != 2:
        raise ValueError(f"边索引维度错误: 期望2维，实际{edge_index.dim()}维")
    
    # 检查边索引形状
    if edge_index.size(0) != 2:
        raise ValueError(f"边索引形状错误: 期望第一维为2，实际{edge_index.size(0)}")
    
    # 获取边数量
    num_edges = edge_index.size(1)
    
    # 计算网络密度
    num_nodes = data.num_nodes
    max_possible = num_nodes * (num_nodes - 1) // 2
    density = num_edges / max_possible if max_possible > 0 else 0.0
    
    # 处理空图情况
    if num_edges == 0:
        logger.warning("图中没有边，返回空数据集")
        empty_data = data.clone()
        empty_data.edge_index = torch.empty((2, 0), dtype=torch.long)
        return empty_data, empty_data.clone(), empty_data.clone(), density
    
    # 计算划分比例
    train_ratio, val_ratio, test_ratio = calculate_split_ratios(density)
    
    # 生成随机索引
    indices = torch.randperm(num_edges)
    
    # 计算划分点
    train_end = int(train_ratio * num_edges)
    val_end = train_end + int(val_ratio * num_edges)
    
    # 创建掩码
    train_mask = indices[:train_end]
    val_mask = indices[train_end:val_end]
    test_mask = indices[val_end:]
    
    # 创建划分后的数据集
    train_data = create_subset(data, edge_index[:, train_mask])
    val_data = create_subset(data, edge_index[:, val_mask])
    test_data = create_subset(data, edge_index[:, test_mask])
    
    return train_data, val_data, test_data, density, (train_ratio, val_ratio, test_ratio)

def calculate_split_ratios(density):
    """根据网络密度计算数据集划分比例"""
    # 动态调整比例：密度越高，训练集比例越低
    train_ratio = max(0.7 - density * 0.2, 0.5)  # 确保最小0.5
    test_ratio = min(0.15 + density * 0.1, 0.3)  # 确保最大0.3
    val_ratio = 1.0 - train_ratio - test_ratio
    
    # 确保验证集比例合理
    if val_ratio < 0.05:
        # 重新平衡
        train_ratio = train_ratio * 0.9
        test_ratio = test_ratio * 0.9
        val_ratio = 1.0 - train_ratio - test_ratio
    
    logger.info(f"根据密度 {density:.4f} 计算划分比例: "
                f"训练 {train_ratio:.2f}, 验证 {val_ratio:.2f}, 测试 {test_ratio:.2f}")
    
    return train_ratio, val_ratio, test_ratio

def create_subset(original_data, edge_index):
    """创建子集数据对象"""
    # 关键修复：使用 original_data.keys() 而不是 original_data.keys
    subset = Data()
    
    # 复制所有属性
    for key in original_data.keys():  # 调用 keys() 方法
        if key != 'edge_index':  # 排除边索引
            attr = getattr(original_data, key)
            if torch.is_tensor(attr):
                setattr(subset, key, attr.clone())
            else:
                setattr(subset, key, attr)
    
    # 设置新的边索引
    subset.edge_index = edge_index
    
    return subset