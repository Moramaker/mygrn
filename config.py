class Config:
    """全局配置类，包含所有模型和训练参数"""
    
    # === 核心模型参数 ===
    HIDDEN_DIM = 128           # GAT隐藏层维度
    OUTPUT_DIM = 64            # 输出表示维度
    GAT_HEADS = 4              # GAT注意力头数
    GAT_DROPOUT = 0.1          # GAT丢弃率
    PROJ_LAYERS = 2            # 投影层数
    
    # === 对比学习参数 ===
    TEMPERATURE = 0.07         # 对比损失的温度参数
    CONTRAST_WEIGHT = 1.0      # 对比损失权重
    ALIGN_WEIGHT = 0.4         # 视图对齐损失权重
    
    # === 训练参数 ===
    EPOCHS = 2000               # 总训练轮数
    CONTRAST_RATIO = 0.9       # 对比学习阶段比例
    BATCH_SIZE = 32            # 批次大小
    LR = 0.0001                 # 学习率
    WEIGHT_DECAY = 1e-5        # 权重衰减
    
    # === 数据集配置 ===
    DATA_ROOT = "./DATASET/single-cell"  # 数据集根目录
    NETWORK_TYPE = "Specific"
    CELL_TYPE = "hESC"
    GENE_NUM = 500

    # === 日志与保存 ===
    LOG_DIR = "./logs"
    MODEL_SAVE_PATH = "./model.pt"
    
    # === 网络密度预设 ===
    DENSITY_PRESETS = {
        'STRING': {
            'hESC500': 0.024, 'hESC1000': 0.021, 
            'hHEP500': 0.028, 'hHEP1000': 0.024,
            'mDC500': 0.038, 'mDC1000': 0.032,
            'mESC500': 0.024, 'mESC1000': 0.021,
            'mHSC-E500': 0.029, 'mHSC-E1000': 0.027,
            'mHSC-GM500': 0.040, 'mHSC-GM1000': 0.037,
            'mHSC-L500': 0.048, 'mHSC-L1000': 0.045
        },
        'Non-Specific': {
            'hESC500': 0.016, 'hESC1000': 0.014,
            'hHEP500': 0.015, 'hHEP1000': 0.013,
            'mDC500': 0.019, 'mDC1000': 0.016,
            'mESC500': 0.015, 'mESC1000': 0.013,
            'mHSC-E500': 0.022, 'mHSC-E1000': 0.020,
            'mHSC-GM500': 0.030, 'mHSC-GM1000': 0.029,
            'mHSC-L500': 0.048, 'mHSC-L1000': 0.043
        },
        'Specific': {
            'hESC500': 0.164, 'hESC1000': 0.165,
            'hHEP500': 0.379, 'hHEP1000': 0.377,
            'mDC500': 0.085, 'mDC1000': 0.082,
            'mESC500': 0.345, 'mESC1000': 0.347,
            'mHSC-E500': 0.578, 'mHSC-E1000': 0.566,
            'mHSC-GM500': 0.543, 'mHSC-GM1000': 0.565,
            'mHSC-L500': 0.525, 'mHSC-L1000': 0.507
        },
        'Lofgof': {
            'mESC500': 0.158, 'mESC1000': 0.154
        }
    }

# 全局配置实例
config = Config()