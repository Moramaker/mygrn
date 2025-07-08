import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

def calculate_metrics(labels, preds):
    """
    计算分类性能指标
    
    参数:
        labels: 真实标签数组
        preds: 预测概率数组
        
    返回:
        metrics: 包含各项指标的字典
    """
    if len(preds) == 0 or len(labels) == 0:
        return {
            "auc": 0.5,
            "auprc": 0.5,
            "accuracy": 0.5
        }
    
    # AUC
    auc = roc_auc_score(labels, preds)
    
    # AUPRC
    auprc = average_precision_score(labels, preds)
    
    # 准确率
    predictions = (preds > 0.5).astype(int)
    accuracy = accuracy_score(labels, predictions)
    
    return {
        "auc": auc,
        "auprc": auprc,
        "accuracy": accuracy
    }