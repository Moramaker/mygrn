import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练
from data_processing import GeneRegulationDataset, split_dataset, negative_sampling
from models import DualViewModel
from utils import setup_logger, calculate_metrics
from tqdm import tqdm
from config import config
import time

def main():
    """主训练函数"""
    logger = setup_logger(config.LOG_DIR)
    start_time = time.time()
    
    logger.info(f"=== 训练开始 ===")
    logger.info(f"配置参数并加载数据")
    
    dataset = GeneRegulationDataset()
    data = dataset[0]
    
    train_data, val_data, test_data, density, ratios = split_dataset(data)
    train_ratio, val_ratio, test_ratio = ratios
    logger.info(f"网络密度: {density:.4f}, 动态划分比例: 训练{train_ratio*100:.1f}%, 验证{val_ratio*100:.1f}%, 测试{test_ratio*100:.1f}%")
    
    # 使用多个工作线程加速数据加载
    train_loader = DataLoader([train_data], batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader([val_data], batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader([test_data], batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualViewModel(input_dim=data.feature_dim).to(device)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # 混合精度训练设置
    scaler = GradScaler(enabled=device.type == 'cuda')
    
    contrastive_epochs = int(config.EPOCHS * config.CONTRASTIVE_RATIO)
    finetune_epochs = config.EPOCHS - contrastive_epochs
    
    logger.info(f"=== 阶段1：双视图对比预训练 ({contrastive_epochs}轮) ===")
    best_pretrain_loss = float('inf')
    early_stopping_counter = 0
    
    contrastive_pbar = tqdm(range(contrastive_epochs), desc="对比预训练", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")
    
    for epoch in contrastive_pbar:
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            with autocast(enabled=device.type == 'cuda'):
                contrastive_loss, alignment_loss, causal_loss = model(batch, task="contrastive")
                loss = contrastive_loss + alignment_loss + causal_loss
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        contrastive_pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
        
        # 早停机制
        if avg_loss < best_pretrain_loss:
            best_pretrain_loss = avg_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            early_stopping_counter = 0
            contrastive_pbar.set_postfix({"loss": f"{avg_loss:.4f}", "best": f"{best_pretrain_loss:.4f}", "saved": "✓"})
            logger.info(f"[Epoch {epoch+1}] 新最佳损失: {avg_loss:.4f} | 模型已保存")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(f"[Epoch {epoch+1}] 早停触发：验证损失连续 {config.EARLY_STOPPING_PATIENCE} 轮未改善")
                break
    
    logger.info(f"=== 阶段1完成: 最佳损失 {best_pretrain_loss:.4f} ===")
    
    logger.info(f"=== 阶段2：下游任务微调 ({finetune_epochs}轮) ===")
    best_val_auc = 0.0
    early_stopping_counter = 0
    
    finetune_pbar = tqdm(range(finetune_epochs), desc="下游微调", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")

    for epoch in finetune_pbar:
        model.train()
        epoch_loss = 0
        all_preds, all_labels = [], []
        
        for batch in train_loader:
            batch = batch.to(device)
            
            with autocast(enabled=device.type == 'cuda'):
                pos_pred = model(batch, task="prediction").squeeze()
                
                num_neg_samples = batch.edge_index.size(1)
                neg_edges = negative_sampling(
                    edge_index=batch.edge_index,
                    num_nodes=batch.num_nodes,
                    num_neg_samples=num_neg_samples,
                    existing_edges=data.edge_index
                )
                
                neg_batch = batch.clone().to(device)
                neg_batch.edge_index = neg_edges.to(device)
                
                neg_pred = model(neg_batch, task="prediction").squeeze()
                
                preds = torch.cat([pos_pred, neg_pred])
                labels = torch.cat([
                    torch.ones_like(pos_pred),
                    torch.zeros_like(neg_pred)
                ])
                
                loss = F.binary_cross_entropy_with_logits(preds, labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            with torch.no_grad():
                all_preds.append(preds.sigmoid().detach().cpu())
                all_labels.append(labels.cpu())
        
        # 验证模型
        val_metrics = evaluate_model(model, val_loader, data.edge_index, device)
        
        if all_preds:
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()
            train_metrics = calculate_metrics(all_labels, all_preds)
        else:
            train_metrics = {"auc": 0.5, "auprc": 0.5}
        
        avg_loss = epoch_loss / len(train_loader)
        
        # 更新学习率
        scheduler.step(val_metrics["auc"])
        
        finetune_pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "train_auc": f"{train_metrics['auc']:.4f}",
            "val_auc": f"{val_metrics['auc']:.4f}",
            "best_val": f"{best_val_auc:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
        })
        
        # 早停机制
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            early_stopping_counter = 0
            finetune_pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "train_auc": f"{train_metrics['auc']:.4f}",
                "val_auc": f"{val_metrics['auc']:.4f}",
                "best_val": f"{best_val_auc:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                "saved": "✓"
            })
            logger.info(f"[Epoch {epoch+1}] 新最佳验证AUC: {val_metrics['auc']:.4f} | 模型已保存")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(f"[Epoch {epoch+1}] 早停触发：验证AUC连续 {config.EARLY_STOPPING_PATIENCE} 轮未改善")
                break
    
    logger.info("=== 最终测试评估 ===")
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    test_metrics = evaluate_model(model, test_loader, data.edge_index, device)
    
    elapsed_time = time.time() - start_time
    logger.info(f"训练完成! 耗时: {elapsed_time/60:.2f}分钟")
    logger.info(f"验证集最佳AUC: {best_val_auc:.4f}")
    logger.info(f"测试集结果 - AUC: {test_metrics['auc']:.4f} | AUPRC: {test_metrics['auprc']:.4f}")

@torch.no_grad()  # 禁用梯度计算以加速推理
def evaluate_model(model, loader, all_pos_edges, device):
    """评估模型性能"""
    model.eval()
    all_preds, all_labels = [], []
    
    for batch in loader:
        batch = batch.to(device)
        
        pos_pred = model(batch, task="prediction").squeeze()
        
        num_neg_samples = batch.edge_index.size(1)
        neg_edges = negative_sampling(
            edge_index=batch.edge_index,
            num_nodes=batch.num_nodes,
            num_neg_samples=num_neg_samples,
            existing_edges=all_pos_edges
        )
        
        if neg_edges.numel() == 0:  # 处理没有负样本的情况
            continue
        
        neg_batch = batch.clone().to(device)
        neg_batch.edge_index = neg_edges.to(device)
        
        neg_pred = model(neg_batch, task="prediction").squeeze()
        
        preds = torch.cat([pos_pred, neg_pred])
        labels = torch.cat([
            torch.ones_like(pos_pred),
            torch.zeros_like(neg_pred)
        ])
        
        all_preds.append(preds.sigmoid().cpu())
        all_labels.append(labels.cpu())
    
    if all_preds:
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        return calculate_metrics(all_labels, all_preds)
    else:
        return {"auc": 0.5, "auprc": 0.5}

if __name__ == "__main__":
    main()