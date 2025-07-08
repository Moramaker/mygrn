import os
import json
import time
import itertools
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from train import main as train_main
from config import Config

class HyperparameterTuner:
    """超参数调优器"""
    
    def __init__(self, config):
        """
        初始化超参数调优器
        
        参数:
            config: 基础配置对象
        """
        self.config = config
        self.best_score = -float('inf')
        self.best_params = None
        self.results = []
        self.trial_count = 0
        
        # 创建日志目录
        self.log_dir = os.path.join(config.LOG_DIR, "hyperparam_tuning")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # TensorBoard记录器
        self.writer = SummaryWriter(self.log_dir)
    
    def define_search_space(self):
        """定义超参数搜索空间"""
        return {
            # 模型架构参数
            "hidden_dim": [64, 128, 256],
            "output_dim": [32, 64, 128],
            "gat_heads": [2, 4, 8],
            "proj_layers": [1, 2, 3],
            
            # 训练参数
            "lr": [0.001, 0.0005, 0.0001],
            "batch_size": [16, 32, 64],
            "weight_decay": [0, 1e-5, 1e-4],
            
            # 视图参数
            "causal_noise": [0.1, 0.2, 0.3, 0.4],
            "diffusion_steps": [30, 50, 70],
            "temperature": [0.2, 0.3, 0.4, 0.5],
            
            # 损失权重
            "contrast_weight": [0.5, 0.7, 1.0],
            "align_weight": [0.3, 0.4, 0.5],
            "backdoor_weight": [0.2, 0.3, 0.4],
            
            # 训练策略
            "contrast_ratio": [0.4, 0.5, 0.6],
            "epochs": [80, 100, 120]
        }
    
    def generate_trials(self, search_space, num_trials=50, strategy="random"):
        """
        生成试验参数组合
        
        参数:
            search_space: 参数搜索空间
            num_trials: 试验次数
            strategy: 生成策略 ("grid"或"random")
        """
        if strategy == "grid":
            # 网格搜索：生成所有组合
            keys = search_space.keys()
            values = [search_space[key] for key in keys]
            trials = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
            return trials[:num_trials]
        
        elif strategy == "random":
            # 随机搜索：随机采样
            trials = []
            for _ in range(num_trials):
                trial = {}
                for key, values in search_space.items():
                    if isinstance(values[0], int):
                        trial[key] = np.random.choice(values)
                    elif isinstance(values[0], float):
                        trial[key] = np.random.uniform(min(values), max(values))
                trials.append(trial)
            return trials
        
        else:
            raise ValueError(f"不支持的策略: {strategy}")
    
    def run_trial(self, trial_params):
        """
        运行单个试验
        
        参数:
            trial_params: 试验参数字典
        """
        self.trial_count += 1
        trial_id = f"trial_{self.trial_count}"
        trial_dir = os.path.join(self.log_dir, trial_id)
        os.makedirs(trial_dir, exist_ok=True)
        
        # 创建试验配置
        trial_config = Config()
        for key, value in trial_params.items():
            setattr(trial_config, key.upper(), value)
        
        # 设置试验特定路径
        trial_config.MODEL_SAVE_PATH = os.path.join(trial_dir, "model.pt")
        trial_config.LOG_DIR = os.path.join(trial_dir, "logs")
        
        # 记录试验参数
        print(f"\n 开始试验 {self.trial_count}:")
        print(json.dumps(trial_params, indent=2))
        
        # 运行训练
        start_time = time.time()
        try:
            auc_score = train_main(config=trial_config, trial_mode=True)
            elapsed = time.time() - start_time
            
            # 记录结果
            result = {
                "trial_id": trial_id,
                "params": trial_params,
                "score": auc_score,
                "time": elapsed
            }
            
            # 更新最佳结果
            if auc_score > self.best_score:
                self.best_score = auc_score
                self.best_params = trial_params
                print(f" 新最佳分数: {auc_score:.4f}")
            
            # TensorBoard记录
            self.writer.add_scalar('AUC', auc_score, self.trial_count)
            for key, value in trial_params.items():
                self.writer.add_scalar(f'Params/{key}', value, self.trial_count)
            
            return result
        
        except Exception as e:
            print(f" 试验失败: {str(e)}")
            return {
                "trial_id": trial_id,
                "params": trial_params,
                "error": str(e),
                "time": time.time() - start_time
            }
    
    def run_tuning(self, num_trials=30, strategy="random"):
        """
        执行超参数调优
        
        参数:
            num_trials: 试验次数
            strategy: 参数生成策略 ("grid"或"random")
        """
        print(f" 开始超参数调优 ({num_trials}次试验, 策略: {strategy})")
        
        # 定义搜索空间
        search_space = self.define_search_space()
        
        # 生成试验
        trials = self.generate_trials(search_space, num_trials, strategy)
        
        # 运行所有试验
        for trial_params in trials:
            result = self.run_trial(trial_params)
            self.results.append(result)
            
            # 保存中间结果
            self.save_results()
        
        # 保存最终结果
        self.save_results()
        self.writer.close()
        
        print(f"\n 调优完成! 最佳AUC: {self.best_score:.4f}")
        print("最佳参数:")
        print(json.dumps(self.best_params, indent=2))
        
        return self.best_params, self.best_score
    
    def save_results(self, filename="hyperparam_results.json"):
        """保存调优结果"""
        results_path = os.path.join(self.log_dir, filename)
        with open(results_path, "w") as f:
            json.dump({
                "best_params": self.best_params,
                "best_score": self.best_score,
                "all_results": self.results
            }, f, indent=2)
        print(f"结果已保存至 {results_path}")
    
    def visualize_results(self):
        """可视化调优结果"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_theme()
            
            # 创建结果数据框
            import pandas as pd
            df = pd.DataFrame(self.results)
            
            # 过滤失败试验
            df = df[df['score'].notna()]
            
            # 参数重要性分析
            param_names = list(self.define_search_space().keys())
            fig, axes = plt.subplots(nrows=len(param_names), ncols=1, figsize=(10, 5*len(param_names)))
            
            for i, param in enumerate(param_names):
                ax = axes[i]
                sns.scatterplot(data=df, x=param, y='score', ax=ax)
                ax.set_title(f"{param} vs AUC")
                ax.set_xlabel(param)
                ax.set_ylabel("AUC Score")
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, "param_importance.png"))
            
            # 最佳参数对比
            best_df = pd.DataFrame([self.best_params])
            best_df['score'] = self.best_score
            
            plt.figure(figsize=(12, 6))
            sns.barplot(data=best_df.melt(id_vars='score'), x='variable', y='value', hue='score')
            plt.title("Best Parameters")
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(self.log_dir, "best_params.png"))
            
            print(" 结果可视化已保存")
            
        except ImportError:
            print("可视化需要matplotlib和seaborn库")

if __name__ == "__main__":
    # 初始化配置
    config = Config()
    
    # 创建调优器
    tuner = HyperparameterTuner(config)
    
    # 运行调优
    best_params, best_score = tuner.run_tuning(num_trials=30, strategy="random")
    
    # 可视化结果
    tuner.visualize_results()
    
    # 使用最佳参数训练最终模型
    print("\n使用最佳参数训练最终模型...")
    for key, value in best_params.items():
        setattr(config, key.upper(), value)
    
    # 设置最终模型路径
    config.MODEL_SAVE_PATH = os.path.join(config.LOG_DIR, "final_model.pt")
    
    # 运行训练
    from train import main
    main(config=config)