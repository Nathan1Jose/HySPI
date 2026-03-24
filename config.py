# -*- coding: utf-8 -*-
import os

# --- 基础路径 ---
PROJECT_ROOT = "/mnt/e/zlc/Project/HySPI"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

# --- 输入文件路径 ---
ACTIONS_FILE = os.path.join(DATA_ROOT, "processed/tpEppro_aux/actions_clean_reordered.tsv")

SEQ_FILE = os.path.join(DATA_ROOT, "processed/tpEppro_aux/data_seq_clean.fasta")
ID_FILE = os.path.join(DATA_ROOT, "processed/tpEppro_aux/data_id_clean.txt")
CHAIN_FILE = os.path.join(DATA_ROOT, "processed/tpEppro_aux/data_chain_clean.txt")
MAPPING_FILE = os.path.join(DATA_ROOT, "processed/tpEppro_aux/pdb_mapping_clean.tsv")

EMBEDDING_FILE = os.path.join(DATA_ROOT, "embeddings.npz")
CMAP_DIR = os.path.join(DATA_ROOT, "contact_map")

# --- 输出路径 ---
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "saved_model")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

RESULT_FILE = os.path.join(PROJECT_ROOT, "logs/training_results.tsv")

# --- 训练参数 ---
MODEL_ARGS = {
    'emb_dim': 1024,
    'output_dim': 128,
    'dense_hid': 64,
    'task_type': 0,
    'n_classes': 1
}

TRAIN_ARGS = {
    'epochs': 50,
    'lr': 0.001,
    'batch_size': 64,
    'do_test': True,
    'do_save': True,
    'k_folds': 5
}
