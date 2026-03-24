# -*- coding: utf-8 -*-
import torch
import dgl
import scipy.sparse as spp
import os
import numpy as np
from torch.utils.data import Dataset
import config

# --- 设备配置 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 加载 Embedding 数据 ---
# 注意: 这会占用较大内存，确保机器有足够 RAM
print(f"正在加载 Embedding 数据: {config.EMBEDDING_FILE} ...", flush=True)
if os.path.exists(config.EMBEDDING_FILE):
    _embed_npz = np.load(config.EMBEDDING_FILE)
    EMBED_DATA = {k: _embed_npz[k] for k in _embed_npz.files}
    _embed_npz.close()
    print("Embedding 数据加载完成。", flush=True)
else:
    print(f"错误: 未找到 Embedding 文件 {config.EMBEDDING_FILE}。请先运行 gen_emd.py。", flush=True)
    EMBED_DATA = {} 

EMBED_KEY_MAP = {k.lower(): k for k in EMBED_DATA.keys()}

def _build_cmap_map():
    cmap_map = {}
    if not os.path.isdir(config.CMAP_DIR):
        return cmap_map
    for fname in os.listdir(config.CMAP_DIR):
        if not fname.endswith(".npz"):
            continue
        base = fname[:-4]
        lower = base.lower()
        if lower not in cmap_map:
            cmap_map[lower] = fname
    return cmap_map

CMAP_FILE_MAP = _build_cmap_map()

def pad_dmap(dmaplist):
    """
    对序列 Embedding 进行 Padding 处理，统一长度为 1200
    
    参数:
        dmaplist (list): 包含多个 Tensor 的列表，每个 Tensor 是一条序列的 Embedding。
                         Tensor 形状通常为 [seq_len, embedding_dim]。
    
    返回:
        pad_dmap_tensors (Tensor): Padding 后的 4D Tensor，形状为 [batch_size, 1, 1200, 1024]。
                                   已移动到配置的设备 (GPU/CPU)。
    """
    # [batch, 1200, 1024]
    pad_dmap_tensors = torch.zeros((len(dmaplist), 1200, 1024)).float()
    for idx, d in enumerate(dmaplist):
        d = d.float().cpu() # 确保在 CPU 上操作
        # 兼容不同长度的 tensor (截断或填充)
        seq_len = d.shape[0]
        if seq_len > 1200:
            d = d[:1200]
            seq_len = 1200
        pad_dmap_tensors[idx, :seq_len, :] = torch.FloatTensor(d)
    
    # 增加通道维度 [batch, 1, 1200, 1024] 并移至 GPU
    pad_dmap_tensors = pad_dmap_tensors.unsqueeze(1).to(device)
    return pad_dmap_tensors

def default_loader(cpath, pid):
    """
    加载单个样本的 Contact Map 和 Embedding
    
    参数:
        cpath (str): Contact Map (.npz) 文件的绝对路径。
        pid (str): 蛋白质 ID (例如 '1A1M_A')，用于从全局 EMBED_DATA 中查找对应的 Embedding。
    
    返回:
        G (DGLGraph): 构建好的图结构数据，节点包含特征。
        textembed (Tensor): 序列 Embedding，已处理长度 (截断或填充到 1200)。
    """
    if not os.path.exists(cpath):
        raise FileNotFoundError(f"Contact Map not found: {cpath}")
        
    cmap_data = np.load(cpath)
    # 获取序列长度
    nodenum = len(str(cmap_data['seq']))
    cmap = cmap_data['contact']
    
    # --- 构建图 (DGLGraph) ---
    # 处理 Embedding 键名 (兼容 pid 和 pid.npy)
    if pid in EMBED_DATA:
        key = pid
    elif f"{pid}.npy" in EMBED_DATA:
        key = f"{pid}.npy"
    else:
        pid_lower = pid.lower()
        key = EMBED_KEY_MAP.get(pid_lower) or EMBED_KEY_MAP.get(f"{pid_lower}.npy")
        if key is None:
            keys_sample = list(EMBED_DATA.keys())[:5]
            raise KeyError(f"ID {pid} (or {pid}.npy) not found in embeddings. Available keys sample: {keys_sample}")
         
    # 获取原始 embedding [seq_len, 1024]
    raw_embed = EMBED_DATA[key]
    
    # 按照原始脚本逻辑: 直接截取 nodenum 长度
    # 如果 raw_embed 长度不够，这里切片不会报错 (numpy)，但后续赋值给 G.ndata['feat'] 时 DGL 会报错
    g_embed = torch.tensor(raw_embed[:nodenum]).float()

    adj = spp.coo_matrix(cmap)
    src = torch.from_numpy(adj.row).long()
    dst = torch.from_numpy(adj.col).long()
    G = dgl.graph((src, dst), num_nodes=adj.shape[0])
    G.ndata['feat'] = g_embed

    # --- 处理序列 Embedding (Padding/Truncating to 1200) ---
    if nodenum > 1200:
        textembed = raw_embed[:1200]
    elif nodenum < 1200:
        # 补零
        textembed = np.concatenate((raw_embed, np.zeros((1200 - len(raw_embed), 1024))))
    else:
        textembed = raw_embed

    textembed = torch.tensor(textembed).float()
    
    return G, textembed

class MyDataset(Dataset):
    def __init__(self, list1, list2, list3, transform=None, target_transform=None, loader=default_loader):
        """
        三个传过来的数据,注意这里的顺序与原作者的顺序不同，我的训练数据是pep-Pro，而原作者的训练数据是Pro-pep
        26/3/11  把顺序改过来重新训练一次试试
        list1: Peptide IDs
        list2: Receptor IDs
        list3: Labels
        """
        self.list1 = list1
        self.list2 = list2
        self.list3 = list3
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        #按照train的逻辑，分别获取小肽的 ID、蛋白的 ID、标签（注意顺序）
        p1 = self.list1[index]#获取小肽的 ID
        p2 = self.list2[index]#获取蛋白的 ID
        label = self.list3[index]
        
        # 构造 Contact Map 路径
        # 辅助函数：根据命名规范生成文件名
        def _get_cmap_fname(p):
            # 尝试解析 PDB_CHAIN 格式 (包含 '_' 且后缀为单字符)
            if '_' in p:
                chain = p.split('_')[-1]
                if len(chain) == 1:
                    return f"{p}_{ord(chain)}.npz"
            # 默认直接拼接 .npz
            return f"{p}.npz"

        fname1 = _get_cmap_fname(p1)
        fname2 = _get_cmap_fname(p2)

        path1 = os.path.join(config.CMAP_DIR, fname1)#构造路径
        path2 = os.path.join(config.CMAP_DIR, fname2)
        
        # 检查文件是否存在，如果不存在尝试原始命名 (兼容性)
        if not os.path.exists(path1) and os.path.exists(os.path.join(config.CMAP_DIR, p1 + '.npz')):
             path1 = os.path.join(config.CMAP_DIR, p1 + '.npz')
        if not os.path.exists(path2) and os.path.exists(os.path.join(config.CMAP_DIR, p2 + '.npz')):
             path2 = os.path.join(config.CMAP_DIR, p2 + '.npz')

        #再次兼容test数据集的大小写问题
        if not os.path.exists(path1):
            alt1 = CMAP_FILE_MAP.get(os.path.splitext(fname1)[0].lower()) or CMAP_FILE_MAP.get(p1.lower())
            if alt1 is not None:
                path1 = os.path.join(config.CMAP_DIR, alt1)
        if not os.path.exists(path2):
            alt2 = CMAP_FILE_MAP.get(os.path.splitext(fname2)[0].lower()) or CMAP_FILE_MAP.get(p2.lower())
            if alt2 is not None:
                path2 = os.path.join(config.CMAP_DIR, alt2)
        
        # 加载数据
        G1, embed1 = self.loader(path1, p1)
        G2, embed2 = self.loader(path2, p2)
        
        return p1, p2, G1, embed1, G2, embed2, label
        #返回p1（小肽的 ID）、p2（蛋白的 ID）、G1（小肽的图结构数据）、embed1（小肽的序列 Embedding）、G2（蛋白的图结构数据）、embed2（蛋白的序列 Embedding）、label（标签）

    def __len__(self):
        return len(self.list1)

def collate(samples):
    """
    DataLoader 的 collate_fn，用于将样本打包成 batch
    """
    p1, p2, graphs1, dmaps1, graphs2, dmaps2, labels = map(list, zip(*samples))
    return p1, p2, graphs1, dmaps1, graphs2, dmaps2, torch.tensor(labels)
