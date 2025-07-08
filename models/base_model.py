import torch.nn as nn

class BaseModel(nn.Module):
    """所有模型的基类，支持任务模式参数"""
    def forward(self, *args, ​**kwargs):
        """
        安全的前向传播方法
        支持'task'参数
        """
        task = kwargs.pop('task', None)  # 安全获取task参数
        return self._forward(*args, ​**kwargs, task=task)
    
    def _forward(self, *args, task=None, ​**kwargs):
        """由子类实现的具体前向逻辑"""
        raise NotImplementedError("子类必须实现_forward方法")