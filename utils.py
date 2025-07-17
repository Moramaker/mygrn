import os
import logging
from sklearn.metrics import roc_auc_score, average_precision_score

def setup_logger(log_dir):
    """初始化日志"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def calculate_metrics(labels, preds):
    """计算评估指标"""
    try:
        auc = roc_auc_score(labels, preds)
        auprc = average_precision_score(labels, preds)
        return {"auc": auc, "auprc": auprc}
    except ValueError:
        return {"auc": 0.5, "auprc": 0.5}    