import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
from config.config import config

class GeneRegulationDataset(InMemoryDataset):
    """基因调控网络数据集"""
    def __init__(self, root=None, transform=None):
        self.root = root or config.DATA_ROOT
        super().__init__(self.root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
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
        raw_features = expr_df.iloc[:, 1:].values
        features = torch.tensor(raw_features, dtype=torch.float32)
        features = F.normalize(features, dim=1)
        
        # 构建图结构
        edges = []
        for _, row in network_df.iterrows():
            gene1, gene2 = row["Gene1"], row["Gene2"]
            if gene1 in name_to_idx and gene2 in name_to_idx:
                edges.append([name_to_idx[gene1], name_to_idx[gene2]])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # 生成多视图
        causal_view = self._generate_causal_view(features)
        diffusion_view = self._generate_diffusion_view(features)
        
        # 创建数据对象
        graph_data = Data(
            x=features,              # 原始视图
            causal_view=causal_view,  # 因果视图
            diffusion_view=diffusion_view,  # 扩散视图
            edge_index=edge_index,    # 基因互作网络
            gene_ids=gene_ids,        # 基因ID
            feature_dim=features.shape[1],  # 特征维度
            num_nodes=len(genes)       # 节点数量
        )
        
        # 保存处理结果
        torch.save(self.collate([graph_data]), self.processed_paths[0])
    
    def _generate_causal_view(self, features):
        """生成因果视图（基于反事实增强）"""
        # 识别少数样本（表达量低于平均值的基因）
        minority_mask = (features.mean(dim=1) < features.mean())
        causal_view = features.clone()
        
        if minority_mask.any():
            majority_indices = torch.where(~minority_mask)[0]
            minority_indices = torch.where(minority_mask)[0]
            
            # 计算少数样本与多数样本的距离矩阵
            dist_matrix = torch.cdist(features[minority_indices], 
                                     features[majority_indices])
            _, nn_indices = dist_matrix.min(dim=1)
            
            # 生成反事实样本
            synthetic = features[minority_indices] * (1 - config.BACKDOOR_WEIGHT) + \
                        features[majority_indices[nn_indices]] * config.BACKDOOR_WEIGHT
            
            causal_view[minority_indices] = synthetic
        
        return causal_view

    def _generate_diffusion_view(self, features):
        """生成扩散视图（基于前向噪声增强）"""
        # 扩散参数设置
        beta = torch.linspace(1e-4, 0.02, 100)  # 100步扩散
        gamma = 1 - beta
        gamma_bar = torch.cumprod(gamma, dim=0)
        
        # 随机选择时间步
        t = torch.randint(0, 100, (features.size(0),))
        gamma_bar_t = gamma_bar[t][:, None]
        
        # 添加高斯噪声
        noise = torch.randn_like(features)
        return torch.sqrt(gamma_bar_t) * features + \
               torch.sqrt(1 - gamma_bar_t) * noise