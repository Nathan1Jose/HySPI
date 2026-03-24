# -*- coding: utf-8 -*-
"""
面向用户的模型预测脚本 (User-facing Prediction Script)
功能：加载训练好的模型，对输入的蛋白质对（Receptor-Peptide）进行结合预测。
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目路径以便导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import config
    from dataset import MyDataset, collate, device
    from model import GATPPI
except ImportError as e:
    print(f"错误: 无法导入必要的模块。请确保 dataset.py, model.py, config.py 在同一目录下。\n详细错误: {e}")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="TPepPro 模型预测脚本")
    
    parser.add_argument('--input', type=str, required=True, 
                        help='输入文件路径 (TSV格式). 必须包含两列: ReceptorID 和 PeptideID。无表头或带有表头(需指定)。')
    
    parser.add_argument('--model', type=str, required=True, 
                        help='已训练的模型文件路径 (.pkl 或 .pth)')
    
    parser.add_argument('--output', type=str, default='prediction_results.tsv', 
                        help='预测结果输出文件路径 (默认为 prediction_results.tsv)')
    
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='批处理大小 (默认为 32)')
    
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='分类阈值 (默认为 0.5). 大于此值预测为结合(1)，否则为不结合(0)')

    parser.add_argument('--has_header', action='store_true', 
                        help='如果输入文件包含表头，请加上此参数')

    return parser.parse_args()

def load_input_data(input_file, has_header=False):
    """
    读取输入文件，返回 ReceptorID 和 PeptideID 列表
    """
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在。")
        sys.exit(1)
        
    print(f"正在读取输入文件: {input_file} ...")
    try:
        # 假设文件是制表符分隔，前两列是 ID
        df = pd.read_csv(input_file, sep='\t', header=0 if has_header else None)
        
        # 如果列数少于2，尝试逗号分隔
        if df.shape[1] < 2:
             df = pd.read_csv(input_file, sep=',', header=0 if has_header else None)
        
        if df.shape[1] < 2:
            print("错误: 输入文件格式不正确。请确保至少包含两列 (ReceptorID, PeptideID)。")
            sys.exit(1)
            
        # 转换为字符串类型，防止数字ID被当做数字处理
        receptors = df.iloc[:, 0].astype(str).tolist()
        peptides = df.iloc[:, 1].astype(str).tolist()
        
        print(f"成功读取 {len(receptors)} 对样本。")
        return receptors, peptides
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        sys.exit(1)

def main():
    args = parse_args()
    
    # 1. 检查模型文件
    if not os.path.exists(args.model):
        print(f"错误: 模型文件 {args.model} 不存在。")
        sys.exit(1)

    # 2. 加载数据
    # 注意: dataset.py 会自动加载 config.EMBEDDING_FILE 定义的 embedding 数据
    # 如果需要更改 embedding 路径，请在 config.py 中修改，或在此处修改 config.EMBEDDING_FILE
    receptors, peptides = load_input_data(args.input, args.has_header)
    
    # 创建伪标签 (预测时不需要真实标签，全设为0即可)
    dummy_labels = [0] * len(receptors)
    
    # 3. 创建数据集和 DataLoader
    print("正在初始化数据集...")
    try:
        test_ds = MyDataset(receptors, peptides, dummy_labels)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, 
                                 collate_fn=collate, num_workers=4, pin_memory=True)
    except Exception as e:
        print(f"数据集初始化失败: {e}")
        print("提示: 请确保 config.py 中的 CMAP_DIR 和 EMBEDDING_FILE 路径配置正确，且数据文件存在。")
        sys.exit(1)

    # 4. 加载模型
    print(f"正在加载模型: {args.model} ...")
    try:
        model = GATPPI(config.MODEL_ARGS).to(device)
        # 加载参数 (map_location 确保在 CPU/GPU 间兼容)
        state_dict = torch.load(args.model, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)

    # 5. 开始预测
    print("开始预测...")
    results = []
    
    with torch.no_grad():
        # 使用 tqdm 显示进度条
        for batch_idx, (p1, p2, G1, dmap1, G2, dmap2, _) in tqdm(enumerate(test_loader), total=len(test_loader), unit="batch"):
            # 将数据移动到设备
            # 注意: dataset.py 的 collate 函数已经修改为不自动 .to(device)，所以这里手动移动
            G1 = G1.to(device)
            dmap1 = dmap1.to(device)
            G2 = G2.to(device)
            dmap2 = dmap2.to(device)
            
            # 模型前向传播
            # model output is logits (without sigmoid) as per recent change in model.py
            # But wait, did I verify if user kept the change? 
            # In the previous turn I removed sigmoid from model.py to fix BCEWithLogitsLoss issue.
            # So here I should apply sigmoid manually to get probability.
            logits = model(G1, dmap1, G2, dmap2)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            # 处理 batch size 为 1 的情况 (probs 可能是 0-d array)
            if probs.ndim == 0:
                probs = [probs.item()]
            
            # 收集结果
            # 当前 batch 的 ID
            start_idx = batch_idx * args.batch_size
            end_idx = start_idx + len(p1)
            
            batch_receptors = receptors[start_idx:end_idx]
            batch_peptides = peptides[start_idx:end_idx]
            
            for i in range(len(p1)):
                prob = float(probs[i]) if isinstance(probs, (np.ndarray, list)) else float(probs)
                label = 1 if prob >= args.threshold else 0
                results.append({
                    'ReceptorID': p1[i],
                    'PeptideID': p2[i],
                    'Probability': f"{prob:.4f}",
                    'Prediction': label
                })

    # 6. 保存结果
    print(f"正在保存结果到 {args.output} ...")
    df_res = pd.DataFrame(results)
    df_res.to_csv(args.output, sep='\t', index=False)
    
    print("预测完成！")
    print(f"总计预测样本数: {len(df_res)}")
    print(f"结合 (1) 数量: {len(df_res[df_res['Prediction'] == 1])}")
    print(f"不结合 (0) 数量: {len(df_res[df_res['Prediction'] == 0])}")

if __name__ == "__main__":
    main()
