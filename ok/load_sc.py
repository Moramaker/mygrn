import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric import data
import pandas as pd
import os
import os.path as osp
import torch.nn.functional as F


class SingleCellDataset(InMemoryDataset):
    def __init__(self, root, network_type, cell_type, tfs, transform=None):
        self.network_type = network_type
        self.cell_type = cell_type
        self.tfs = tfs
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [
            f"{self.network_type} Dataset/{self.cell_type}/TFs+{self.tfs}/BL--network.csv",
            f"{self.network_type} Dataset/{self.cell_type}/TFs+{self.tfs}/BL--ExpressionData.csv"
        ]

    @property
    def processed_file_names(self):
        return [f"sc_data_{self.network_type}_{self.cell_type}_{self.tfs}.pt"]

    def process(self):
        # 加载表达数据
        expr_df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names[1]))
        genes = expr_df.iloc[:, 0].tolist()
        name_to_idx = {gene: idx for idx, gene in enumerate(genes)}
        
        # 处理特征数据
        features = torch.tensor(expr_df.iloc[:, 1:].values, dtype=torch.float32)
        features = F.normalize(features, dim=1)
        
        # 构建边索引
        network_df = pd.read_csv(osp.join(self.raw_dir, self.raw_file_names[0]), names=["Gene1", "Gene2"])
        edges = []
        for gene1, gene2 in zip(network_df["Gene1"], network_df["Gene2"]):
            if gene1 in name_to_idx and gene2 in name_to_idx:
                edges.append([name_to_idx[gene1], name_to_idx[gene2]])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
        
        # 创建图数据对象
        graph_data = data.Data(x=features, edge_index=edge_index, num_nodes=len(genes))
        
        # 保存处理后的数据
        torch.save(self.collate([graph_data]), self.processed_paths[0])