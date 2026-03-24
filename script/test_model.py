from re import T
import sys
import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, 
    accuracy_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef, confusion_matrix,
    roc_auc_score
)
import dgl

# 1. 配置路径 (必须在导入 dataset 之前修改 config)
import config
config.EMBEDDING_FILE = "/mnt/e/zlc/Project/TPepPro/retrain/SEP_data_embeddings.npz"
config.CMAP_DIR = "/mnt/e/zlc/Project/TPepPro/retrain/contact_map4SEP"

# 2. 导入 dataset 和 model
from dataset import MyDataset, collate, pad_dmap
from model import GATPPI

# 设置 matplotlib 风格 (尝试使用 seaborn-whitegrid，如果不存在则忽略)
try:
    plt.style.use('seaborn-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        pass # 使用默认风格

def load_actions(data_path):
    """
    加载数据文件
    数据格式: Protein | SEP | label (蛋白在前，小肽在后)
    与训练数据格式一致，无需调换顺序
    """
    df = pd.read_csv(data_path, sep='\t')
    if df.shape[1] < 3:
        raise ValueError(f"数据文件格式错误，至少需要3列，当前列数: {df.shape[1]}")
    
    p1 = df.iloc[:, 0].astype(str).to_numpy()  # Protein (第一列)
    p2 = df.iloc[:, 1].astype(str).to_numpy()  # SEP (第二列)
    labels = df.iloc[:, 2].to_numpy()
    
    return p1, p2, labels

def compute_extended_metrics(y_true, y_probs, threshold=0.5):
    """
    计算扩展的评估指标，包括 accuracy, precision, recall, f1, mcc, auc_roc, auc_pr, confusion_matrix
    """
    y_pred = (y_probs >= threshold).astype(int)
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 计算 PR 曲线面积
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
    aupr = auc(recall_curve, precision_curve)
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auc_roc": float(roc_auc_score(y_true, y_probs)),
        "auc_pr": float(aupr),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]
    }
    return metrics

def plot_combined_roc(results, output_path):
    """
    绘制五折合并的 ROC 曲线
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10.colors
    if len(results) > len(colors):
        colors = plt.cm.Set3.colors
    
    for i, res in enumerate(results):
        fpr = res['fpr']
        tpr = res['tpr']
        roc_auc = res['metrics']['auc_roc']
        name = res.get('name', f"Fold {i}")
        label_text = f"{name} (AUC={roc_auc:.4f})"
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, color=color, label=label_text, linewidth=1.5)
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves (5 Folds)', fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_combined_pr(results, output_path):
    """
    绘制五折合并的 PR 曲线
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10.colors
    if len(results) > len(colors):
        colors = plt.cm.Set3.colors
        
    for i, res in enumerate(results):
        recall = res['recall_curve']
        precision = res['precision_curve']
        aupr = res['metrics']['auc_pr']
        name = res.get('name', f"Fold {i}")
        label_text = f"{name} (AUPR={aupr:.4f})"
        color = colors[i % len(colors)]
        ax.plot(recall, precision, color=color, label=label_text, linewidth=1.5)
        
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves (5 Folds)', fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    # 路径配置
    data_path = "/mnt/e/zlc/Project/TPepPro/retrain/SEP_data_action_with_neg.tsv"
    model_base_dir = "/mnt/e/zlc/Project/HySPI/saved_model"
    
    print(f"正在加载数据: {data_path}")
    p1, p2, labels = load_actions(data_path)
    #p1是 Peptide | p2 是  Protein  
    
    # 数据集准备
    dataset = MyDataset(p1, p2, labels)
    dataloader = DataLoader(dataset, batch_size=256, shuffle= True, collate_fn=collate)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    fold_results = []
    all_metrics_list = []
    
    # 遍历 5 折模型
    for fold in range(5):
        # 修正模型文件名匹配逻辑: GAT_fold{i+1}.pkl
        model_path = os.path.join(model_base_dir, f"GAT_fold{fold+1}.pkl")
        if not os.path.exists(model_path):
            print(f"警告: 未找到模型文件 {model_path}，跳过该折。")
            continue
            
        print(f"正在评估 Fold {fold}...")
        
        # 加载模型
        try:
            # 需要传入 args 来初始化模型，从 config 获取
            from config import MODEL_ARGS
            model = GATPPI(MODEL_ARGS)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
        except Exception as e:
            print(f"加载模型失败 {model_path}: {e}")
            continue
            
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Unpack batch data
                # MyDataset returns: p1, p2, G1, embed1, G2, embed2, label
                # collate returns lists of these
                batch_p1, batch_p2, batch_G1, batch_dmap1, batch_G2, batch_dmap2, batch_y = batch
                
                # Batch processing
                pad_dmap1 = pad_dmap(batch_dmap1).to(device)
                pad_dmap2 = pad_dmap(batch_dmap2).to(device)
                batch_G1_dgl = dgl.batch(batch_G1).to(device)
                batch_G2_dgl = dgl.batch(batch_G2).to(device)
                
                
                outputs = model(batch_G1_dgl, pad_dmap1, batch_G2_dgl, pad_dmap2)
                probs = outputs.cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(batch_y.numpy())
        
        y_true = np.array(all_labels)
        y_probs = np.array(all_probs)
        
        # 计算扩展指标
        metrics = compute_extended_metrics(y_true, y_probs)
        
        # 计算曲线数据
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
        
        print(f"Fold {fold} 结果: AUC={metrics['auc_roc']:.4f}, AUPR={metrics['auc_pr']:.4f}, Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
        
        all_metrics_list.append(metrics)
        fold_results.append({
            'name': f"Fold {fold}",
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve,
            'metrics': metrics
        })

    # 保存详细指标
    if all_metrics_list:
        df_metrics = pd.DataFrame(all_metrics_list)
        output_metrics_file = os.path.join(model_base_dir, "test_metrics.csv")
        df_metrics.to_csv(output_metrics_file, index=False)
        print(f"详细指标已保存至: {output_metrics_file}")
        
        # 排除非数值列 (如 confusion_matrix) 进行统计
        numeric_cols = ["accuracy", "precision", "recall", "f1", "mcc", "auc_roc", "auc_pr"]
        df_numeric = df_metrics[numeric_cols]
        
        # 打印平均指标
        print(f"Mean accuracy: {df_numeric['accuracy'].mean():.4f} +/- {df_numeric['accuracy'].std():.4f}")
        print(f"Mean precision: {df_numeric['precision'].mean():.4f} +/- {df_numeric['precision'].std():.4f}")
        print(f"Mean recall: {df_numeric['recall'].mean():.4f} +/- {df_numeric['recall'].std():.4f}")
        print(f"Mean f1: {df_numeric['f1'].mean():.4f} +/- {df_numeric['f1'].std():.4f}")
        print(f"Mean mcc: {df_numeric['mcc'].mean():.4f} +/- {df_numeric['mcc'].std():.4f}")
        print(f"Mean auc_roc: {df_numeric['auc_roc'].mean():.4f} +/- {df_numeric['auc_roc'].std():.4f}")
        print(f"Mean auc_pr: {df_numeric['auc_pr'].mean():.4f} +/- {df_numeric['auc_pr'].std():.4f}")
        
        # 保存汇总 JSON
        summary = {
            "mean_metrics": df_numeric.mean().to_dict(),
            "std_metrics": df_numeric.std().to_dict(),
            "folds": all_metrics_list
        }
        with open(os.path.join(model_base_dir, "test_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)
        print(f"汇总指标已保存至: {os.path.join(model_base_dir, 'test_summary.json')}")

    # 绘图
    if fold_results:
        plot_combined_roc(fold_results, os.path.join(model_base_dir, "roc_5folds.png"))
        plot_combined_pr(fold_results, os.path.join(model_base_dir, "pr_5folds.png"))
        print("曲线图已保存。")
        
        # 保存曲线原始数据 (可选)
        np.savez(os.path.join(model_base_dir, "test_metrics_5fold.npz"), results=fold_results)
        print("曲线原始数据已保存。")

if __name__ == "__main__":
    main()
