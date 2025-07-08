import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from data_processing import GeneRegulationDataset, split_dataset, negative_sampling
from models import TriViewModel
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
    
    train_loader = DataLoader([train_data], batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader([val_data], batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader([test_data], batch_size=config.BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TriViewModel(input_dim=data.feature_dim).to(device)
    optimizer = Adam(
        model.parameters(), 
        lr=config.LR, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    contrast_epochs = int(config.EPOCHS * config.CONTRAST_RATIO)
    finetune_epochs = config.EPOCHS - contrast_epochs
    
    logger.info(f"=== 阶段1：三视图对比预训练 ({contrast_epochs}轮) ===")
    best_loss = float('inf')
    
    contrast_pbar = tqdm(range(contrast_epochs), desc="对比预训练", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")
    
    for epoch in contrast_pbar:
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            contrast_loss, align_loss, backdoor_loss = model(batch, task="contrast")
            
            loss = contrast_loss + align_loss * 0.4 + backdoor_loss * config.BACKDOOR_WEIGHT
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        contrast_pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            contrast_pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "best": f"{best_loss:.4f}",
                "saved": "✓"
            })
            logger.info(f"[Epoch {epoch+1}] 新最佳损失: {avg_loss:.4f} | 模型已保存")

    logger.info(f"=== 阶段1完成: 最佳损失 {best_loss:.4f} ===")
    
    logger.info(f"=== 阶段2：下游任务微调 ({finetune_epochs}轮) ===")
    best_val_auc = 0.0
    
    finetune_pbar = tqdm(range(finetune_epochs), desc="下游微调", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")

    for epoch in finetune_pbar:
        model.train()
        epoch_loss = 0
        all_preds, all_labels = [], []
        
        for batch in train_loader:
            batch = batch.to(device)
            
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
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            all_preds.append(preds.sigmoid().detach().cpu())
            all_labels.append(labels.cpu())
        
        val_metrics = evaluate_model(model, val_loader, data.edge_index, device)
        
        if all_preds:
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()
            train_metrics = calculate_metrics(all_labels, all_preds)
        else:
            train_metrics = {"auc": 0.5, "auprc": 0.5}
        
        avg_loss = epoch_loss / len(train_loader)
        
        finetune_pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "train_auc": f"{train_metrics['auc']:.4f}",
            "val_auc": f"{val_metrics['auc']:.4f}",
            "best_val": f"{best_val_auc:.4f}"
        })
        
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            finetune_pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "train_auc": f"{train_metrics['auc']:.4f}",
                "val_auc": f"{val_metrics['auc']:.4f}",
                "best_val": f"{best_val_auc:.4f}",
                "saved": "✓"
            })
            logger.info(f"[Epoch {epoch+1}] 新最佳验证AUC: {val_metrics['auc']:.4f} | 模型已保存")
    
    logger.info("=== 最终测试评估 ===")
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    test_metrics = evaluate_model(model, test_loader, data.edge_index, device)
    
    elapsed_time = time.time() - start_time
    logger.info(f"训练完成! 耗时: {elapsed_time/60:.2f}分钟")
    logger.info(f"验证集最佳AUC: {best_val_auc:.4f}")
    logger.info(f"测试集结果 - AUC: {test_metrics['auc']:.4f} | AUPRC: {test_metrics['auprc']:.4f}")

def evaluate_model(model, loader, all_pos_edges, device):
    """评估模型性能"""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
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