# -*- coding: utf-8 -*-
import torch
import numpy as np
import os
import sys
import dgl
import torch.multiprocessing
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, average_precision_score, confusion_matrix)
import pandas as pd
from tqdm import tqdm

# 导入自定义模块
import config
from dataset import MyDataset, collate, pad_dmap, device
from model import GATPPI

try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except RuntimeError:
    pass

# --- 验证函数 ---
def validation(model, loader):
    model.eval()
    total_labels = torch.Tensor()
    total_preds = torch.Tensor()
    total_preds_score = torch.Tensor()
    
    print(f'正在验证 {len(loader.dataset)} 个样本...', flush=True)
    with torch.no_grad():
        for batch_idx, (p1, p2, G1, dmap1, G2, dmap2, y) in enumerate(loader):
            # 数据移动到 GPU (dmap 在 pad_dmap 中移动，G1/G2 在 collate 后是 list，需 batch)
            pad_dmap1 = pad_dmap(dmap1)
            pad_dmap2 = pad_dmap(dmap2)
            
            # dgl.batch 处理图列表
            batch_G1 = dgl.batch(G1).to(device)
            batch_G2 = dgl.batch(G2).to(device)
            y = y.to(device)
            
            # 模型前向传播
            output_score = model(batch_G1, pad_dmap1, batch_G2, pad_dmap2)
            
            # 计算预测标签 (阈值 0.5)
            output = torch.round(output_score.squeeze(1))
            
            # 收集结果 (转回 CPU)
            total_labels = torch.cat((total_labels.cpu(), y.float().cpu()), 0)
            total_preds_score = torch.cat((total_preds_score.cpu(), output_score.cpu()), 0)
            total_preds = torch.cat((total_preds.cpu(), output.cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_preds_score.numpy().flatten()

# --- 测试函数 (保存详细结果) ---
def test(model, loader):
    model.eval()
    total_labels = torch.Tensor()
    total_preds = torch.Tensor()
    total_preds_score = torch.Tensor()
    
    # 用于保存详细预测结果的列表
    detailed_results = []
    
    print(f'正在测试 {len(loader.dataset)} 个样本...', flush=True)
    with torch.no_grad():
        for batch_idx, (p1, p2, G1, dmap1, G2, dmap2, y) in enumerate(loader):
            pad_dmap1 = pad_dmap(dmap1)
            pad_dmap2 = pad_dmap(dmap2)
            batch_G1 = dgl.batch(G1).to(device)
            batch_G2 = dgl.batch(G2).to(device)
            
            output_score = model(batch_G1, pad_dmap1, batch_G2, pad_dmap2)
            output_label = torch.round(output_score.squeeze(1))
            
            # 收集
            total_labels = torch.cat((total_labels.cpu(), y.float().cpu()), 0)
            total_preds_score = torch.cat((total_preds_score.cpu(), output_score.cpu()), 0)
            total_preds = torch.cat((total_preds.cpu(), output_label.cpu()), 0)
            
            # 记录详细信息
            for i in range(len(p1)):
                detailed_results.append({
                    "receptor": p1[i],
                    "peptide": p2[i],
                    "label": y[i].item(),
                    "predict_score": output_score[i].item(),
                    "predict_label": output_label[i].item()
                })

    # 保存详细结果到 CSV
    result_dir = os.path.join(config.DATA_ROOT, "results")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    df = pd.DataFrame(detailed_results)
    df.to_csv(os.path.join(result_dir, "test_predictions.csv"), index=False)
    print(f"详细测试结果已保存至 {result_dir}/test_predictions.csv")

    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_preds_score.numpy().flatten()

# --- 主训练逻辑 ---
def train():
    # 1. 读取数据
    print("正在读取训练数据列表...", flush=True)
    all_protein1 = []
    all_protein2 = []
    all_Y = []
    
    if not os.path.exists(config.ACTIONS_FILE):
        print(f"错误: 找不到文件 {config.ACTIONS_FILE}")
        return

    with open(config.ACTIONS_FILE, 'r') as f:
        header_skipped = False
        for line in f:
            line = line.strip()
            if not line: continue
            row = line.split('\t')
            if not header_skipped:
                header_skipped = True
                continue
            all_protein1.append(row[0])#P1 是蛋白
            all_protein2.append(row[1])#P2 是小肽
            all_Y.append(float(row[2]))
            
    all_protein1 = np.array(all_protein1)
    all_protein2 = np.array(all_protein2)
    all_Y = np.array(all_Y)
    
    # 2. 交叉验证
    skf = StratifiedKFold(n_splits=config.TRAIN_ARGS['k_folds'], shuffle=True, random_state=42)
    
    fold_idx = 0
    # 修改逻辑：直接利用 K-Fold 划分训练集和验证集
    for train_idx, val_idx in skf.split(all_Y, all_Y):
        fold_idx += 1
        print(f"\n{'='*20} 开始第 {fold_idx} 折交叉验证 {'='*20}")
        
        # 划分训练集 / 验证集
        train_p1 = all_protein1[train_idx]# 获取蛋白的 ID
        train_p2 = all_protein2[train_idx]# 获取小肽的 ID
        train_y = all_Y[train_idx]
        
        valid_p1 = all_protein1[val_idx]
        valid_p2 = all_protein2[val_idx]
        valid_y = all_Y[val_idx]
        
        print(f"训练集大小: {len(train_y)}")
        print(f"验证集大小: {len(valid_y)}")
        
        # 构建数据集
        train_ds = MyDataset(train_p1, train_p2, train_y)
        valid_ds = MyDataset(valid_p1, valid_p2, valid_y)
        
        num_workers = 2
        if num_workers > 0:
            train_loader = DataLoader(train_ds, batch_size=config.TRAIN_ARGS['batch_size'], shuffle=True, collate_fn=collate, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
            valid_loader = DataLoader(valid_ds, batch_size=config.TRAIN_ARGS['batch_size'], shuffle=False, collate_fn=collate, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
        else:
            train_loader = DataLoader(train_ds, batch_size=config.TRAIN_ARGS['batch_size'], shuffle=True, collate_fn=collate, num_workers=0, pin_memory=True)
            valid_loader = DataLoader(valid_ds, batch_size=config.TRAIN_ARGS['batch_size'], shuffle=False, collate_fn=collate, num_workers=0, pin_memory=True)
        
        # 初始化模型
        model = GATPPI(config.MODEL_ARGS).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN_ARGS['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        criterion = torch.nn.BCELoss()
        
        max_f1 = 0# 记录最佳 F1 分数，用于保存最佳模型
        pkl_path = os.path.join(config.MODEL_SAVE_DIR, f"GAT_fold{fold_idx}.pkl")
        
        # 训练循环
        for epoch in range(config.TRAIN_ARGS['epochs']):
            model.train()
            total_loss = 0
            correct = 0
            n_batches = 0
            
            print(f"Epoch {epoch+1}/{config.TRAIN_ARGS['epochs']} ...")
            
            # 使用 tqdm 显示进度条
            train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
            for batch_idx, (p1, p2, G1, dmap1, G2, dmap2, y) in train_pbar:
                pad_dmap1 = pad_dmap(dmap1)
                pad_dmap2 = pad_dmap(dmap2)
                batch_G1 = dgl.batch(G1).to(device)
                batch_G2 = dgl.batch(G2).to(device)
                y = y.type(torch.FloatTensor).to(device)
                
                optimizer.zero_grad()
                y_pred = model(batch_G1, pad_dmap1, batch_G2, pad_dmap2)
                y_pred = y_pred.squeeze(1)
                
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                predicted = torch.round(y_pred)
                correct += (predicted == y).sum().item()
                n_batches += 1
                
                # 更新进度条后缀
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            scheduler.step()
            
            avg_loss = total_loss / n_batches
            train_acc = correct / len(train_loader.dataset)
            
            print(f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # 验证
            val_labels, val_preds, val_scores = validation(model, valid_loader)
            val_acc = accuracy_score(val_labels, val_preds)
            val_auc = roc_auc_score(val_labels, val_scores)
            val_f1 = f1_score(val_labels, val_preds)
            
            # 计算混淆矩阵及相关指标
            tn, fp, fn, tp = confusion_matrix(val_labels, val_preds).ravel()
            val_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Sensitivity / Recall
            val_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            val_tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Specificity
            val_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            print(f"Valid Acc: {val_acc:.4f}, AUC: {val_auc:.4f}, F1: {val_f1:.4f}")
            print(f"      TPR: {val_tpr:.4f}, FPR: {val_fpr:.4f}, TNR: {val_tnr:.4f}, FNR: {val_fnr:.4f}")
            
            # 记录日志
            with open(config.RESULT_FILE, 'a+') as f:
                f.write(f"Fold{fold_idx}\tEpoch{epoch+1}\tTrainLoss={avg_loss:.4f}\tTrainAcc={train_acc:.4f}\tValAcc={val_acc:.4f}\tValAUC={val_auc:.4f}\tValF1={val_f1:.4f}\tTPR={val_tpr:.4f}\tFPR={val_fpr:.4f}\tTNR={val_tnr:.4f}\tFNR={val_fnr:.4f}\n")
            
            # 保存最佳模型
            if val_f1 > max_f1:
                max_f1 = val_f1
                print(f"New best model (F1 {max_f1:.4f}), saving to {pkl_path}")
                torch.save(model.state_dict(), pkl_path)
                
if __name__ == "__main__":
    train()