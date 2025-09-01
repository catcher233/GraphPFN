from csbm.csbm_dataset import _parameterized_lambda_and_mu
from csbm.csbm_dataset import _context_sbm
from csbm.csbm_dataset import _save_data_to_pickle
from utils import *
from csbm.csbm_dataset import *
import os.path as osp
import numpy as np


# --- 1. 设置生成图的参数 ---
n_nodes = 10000  # 节点数
n_features = 500  # 特征维度
avg_degree = 10  # 平均度
n_samples = 5  # 采样次数
theta_values = np.random.uniform(-1, 1, n_samples)  # 从[-1,1]均匀分布采样theta值

print(f"Sampled theta values: {theta_values}")

# --- 2. 为每个theta值生成图数据 ---
root_path = './my_datasets'  # 你可以改成任何你想要的路径
datasets = []

for i, theta in enumerate(theta_values):
    print(f"\n--> Step {i+1}: Generating data for theta = {theta:.4f}...")
    
    # 使用 theta 计算 Lambda 和 mu
    Lambda, mu = _parameterized_lambda_and_mu(theta, p=n_features, n=n_nodes)
    
    # 生成 cSBM 图数据对象
    raw_data_object = _context_sbm(n=n_nodes, d=avg_degree, Lambda=Lambda, p=n_features, mu=mu)
    
    # 设置保存的根目录和文件名
    raw_file_name = f"raw_csbm_theta_{theta:.4f}"
    
    # 保存为 pickle 文件
    pickle_file_path = _save_data_to_pickle(raw_data_object, p2root=osp.join(root_path, raw_file_name, 'raw'),file_name=raw_file_name)
    print(f"Raw data saved to: {pickle_file_path}")
    
    # 实例化 PyG 数据集类，它会自动处理数据
    print(f"Creating PyTorch Geometric dataset for theta = {theta:.4f}...")
    dataset = SynCSBM(root=osp.join(root_path, raw_file_name), name=raw_file_name)
    datasets.append(dataset)
    
    # 验证生成的数据
    graph_data = dataset.data
    print(f"Dataset {i+1} - Nodes: {graph_data.num_nodes}, Edges: {graph_data.num_edges}, Features: {graph_data.num_features}")
    print(f"Dataset {i+1} - Train: {graph_data.train_mask.sum().item()}, Val: {graph_data.val_mask.sum().item()}, Test: {graph_data.test_mask.sum().item()}")

print(f"\n=== Summary ===")
print(f"Successfully generated {len(datasets)} datasets with theta values: {theta_values}")
print(f"All datasets saved in: {root_path}")