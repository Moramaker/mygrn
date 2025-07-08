import logging
import os
from datetime import datetime
import sys
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（假设utils在项目根目录下）
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到Python路径
sys.path.append(project_root)

# 现在可以导入config
from config.config import config

def setup_logger(log_dir=None):
    """设置日志记录器"""
    log_dir = log_dir or config.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # 创建日志记录器
    logger = logging.getLogger("GeneRegulationLogger")
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger