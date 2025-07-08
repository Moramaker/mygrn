def validate_data_attributes(data):
    """验证数据对象是否包含必要属性"""
    required_attrs = ['x', 'edge_index', 'causal_view', 'diffusion_view', 'num_nodes']
    missing = [attr for attr in required_attrs if not hasattr(data, attr)]
    
    if missing:
        raise AttributeError(f"数据对象缺少必要属性: {', '.join(missing)}")
    
    # 检查边索引维度
    if data.edge_index.dim() != 2:
        raise ValueError(f"边索引维度错误: 期望2维，实际{data.edge_index.dim()}维")
    
    # 检查边索引形状
    if data.edge_index.size(0) != 2:
        raise ValueError(f"边索引形状错误: 期望第一维为2，实际{data.edge_index.size(0)}")
    
    # 检查节点数量
    if data.num_nodes < 2:
        raise ValueError(f"节点数量不足: 需要至少2个节点，实际{data.num_nodes}")

def auto_fix_edge_index(edge_index):
    """自动修复边索引维度问题"""
    if edge_index.dim() == 2 and edge_index.size(0) == 2:
        return edge_index  # 无需修复
    
    if edge_index.dim() == 1:
        if edge_index.numel() % 2 == 0:
            return edge_index.view(2, -1).contiguous()
        else:
            raise ValueError("无法修复: 元素数量不能被2整除")
    
    if edge_index.dim() == 2 and edge_index.size(0) != 2:
        if edge_index.size(1) == 2:
            return edge_index.t().contiguous()
        else:
            raise ValueError("无法修复: 无效的形状")
    
    raise ValueError(f"无法修复: 不支持的维度 {edge_index.dim()}")