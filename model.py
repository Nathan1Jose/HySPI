# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import TAGConv
from dgl.nn.pytorch.glob import MaxPooling, AvgPooling

# --- 卷积层定义 (用于处理序列特征) ---
class ConvsLayer(torch.nn.Module):
    def __init__(self, emb_dim):
        super(ConvsLayer, self).__init__()
        self.embedding_size = emb_dim
        # 1D 卷积层定义
        self.conv1 = nn.Conv1d(in_channels=self.embedding_size, out_channels=128, kernel_size=3)
        self.mx1 = nn.MaxPool1d(3, stride=3)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3)
        self.mx2 = nn.MaxPool1d(3, stride=3)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3)
        # 注意: 这里的 pooling 大小可能需要根据实际序列长度调整，保持原逻辑
        self.mx3 = nn.MaxPool1d(130, stride=1)

    def forward(self, x):
        # x shape: [batch, 1, seq_len, emb_dim] -> [batch, seq_len, emb_dim]
        x = x.squeeze(1)
        # Permute for Conv1d: [batch, emb_dim, seq_len]
        x = x.permute(0, 2, 1)
        
        features = self.conv1(x)
        features = self.mx1(features)
        
        features = self.mx2(self.conv2(features))
        
        features = self.conv3(features)
        features = self.mx3(features)
        
        # Squeeze back: [batch, 128]
        features = features.squeeze(2)
        return features

# --- 主模型定义 (GATPPI) ---
class GATPPI(torch.nn.Module):
    def __init__(self, args):
        super(GATPPI, self).__init__()
        # 禁用 cudnn 可能是为了避免某些 RNN/LSTM 的问题，保留原设置
        torch.backends.cudnn.enabled = False
        
        self.type = args['task_type']
        self.embedding_size = args['emb_dim']
        self.output_dim = args['output_dim']
        
        # --- 图神经网络部分 (TAGConv) ---
        # k=2 表示 2-hop 邻居
        self.gcn1 = TAGConv(self.embedding_size, self.embedding_size, k=2) 
        
        self.relu = nn.ReLU()
        self.fc_g1 = torch.nn.Linear(self.embedding_size, self.output_dim)  

        self.maxpooling = MaxPooling()
        self.avgpooling = AvgPooling()

        # --- 序列处理部分 (TextCNN) ---
        self.textcnn = ConvsLayer(self.embedding_size)
        self.textflatten = nn.Linear(128, self.output_dim)
        
        # --- 融合与分类层 ---
        # 学习权重参数，用于平衡图特征和序列特征
        self.w1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        
        self.fc1 = nn.Linear(self.output_dim * 2, 512)
        self.BatchNorm1d_1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.BatchNorm1d_2 = nn.BatchNorm1d(256)
        
        self.out = nn.Linear(256, 1)

    def forward(self, G1, pad_dmap1, G2, pad_dmap2):
        # --- Protein 1 处理 --- 这里是P1 同样是小肽
        # Graph path
        g1 = self.relu(self.gcn1(G1, G1.ndata['feat']))
        g1 = g1.reshape(-1, self.embedding_size)  
        G1.ndata['feat'] = g1
        g1_maxpooling = self.maxpooling(G1, G1.ndata['feat'])
        g1_feat = self.relu(self.fc_g1(g1_maxpooling))

        # Sequence path
        seq1 = self.textcnn(pad_dmap1)
        seq1 = self.relu(self.textflatten(seq1))
        
        # Weighted combination for Protein 1
        w1 = torch.sigmoid(self.w1) # 原代码用 F.sigmoid
        gc1 = torch.add((1 - w1) * g1_feat, w1 * seq1)

        # --- Protein 2 处理 ---
        # Graph path
        g2 = F.relu(self.gcn1(G2, G2.ndata['feat']))
        g2 = g2.reshape(-1, self.embedding_size)  
        G2.ndata['feat'] = g2
        g2_maxpooling = self.maxpooling(G2, G2.ndata['feat'])
        g2_feat = self.relu(self.fc_g1(g2_maxpooling))

        # Sequence path
        seq2 = self.textcnn(pad_dmap2)
        seq2 = self.relu(self.textflatten(seq2))
        
        # Weighted combination for Protein 2
        gc2 = torch.add((1 - w1) * g2_feat, w1 * seq2)

        # --- 融合两个蛋白的特征 ---
        gc = torch.cat([gc1, gc2], dim=1)
        
        # Dense layers
        gc = self.fc1(gc)
        gc = self.BatchNorm1d_1(gc)
        gc = self.relu(gc)
        
        gc = self.fc2(gc)
        gc = self.BatchNorm1d_2(gc)
        gc = self.relu(gc)
        
        # Output layer
        out = self.out(gc)
        return torch.sigmoid(out)
