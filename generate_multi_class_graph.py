from csbm.multi_class_csbm import _multi_class_context_sbm_lpm, _parameterized_multi_class
from utils import *
import os.path as osp
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

# # Google Drive 上传相关导入
# import os
# import json
# from datetime import datetime
# from google.oauth2.credentials import Credentials
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from googleapiclient.discovery import build
# from googleapiclient.http import MediaFileUpload
# import pickle

# # Google Drive API权限范围
# SCOPES = ['https://www.googleapis.com/auth/drive.file']

# def authenticate_google_drive():
#     """认证Google Drive API"""
#     creds = None
    
#     # 检查是否存在token文件
#     if os.path.exists('token.pickle'):
#         with open('token.pickle', 'rb') as token:
#             creds = pickle.load(token)
    
#     # 如果没有有效凭据，进行OAuth流程
#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             # 查找OAuth JSON文件
#             oauth_files = [f for f in os.listdir('.') if f.startswith('client_secret_') and f.endswith('.json')]
#             if not oauth_files:
#                 raise FileNotFoundError("未找到OAuth JSON文件！请确保文件在当前目录下。")
            
#             oauth_file = oauth_files[0]
#             print(f"使用OAuth文件: {oauth_file}")
            
#             flow = InstalledAppFlow.from_client_secrets_file(oauth_file, SCOPES)
#             creds = flow.run_local_server(port=0)
        
#         # 保存凭据以供下次使用
#         with open('token.pickle', 'wb') as token:
#             pickle.dump(creds, token)
    
#     return build('drive', 'v3', credentials=creds)

# def upload_single_file_to_drive(service, file_path, folder_name=None):
#     """上传单个文件到Google Drive"""
    
#     # 创建文件夹（如果指定且不存在）
#     folder_id = None
#     if folder_name:
#         # 检查文件夹是否已存在
#         results = service.files().list(
#             q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'",
#             fields="files(id, name)"
#         ).execute()
        
#         folders = results.get('files', [])
#         if folders:
#             folder_id = folders[0]['id']
#             print(f"使用现有文件夹: {folder_name} (ID: {folder_id})")
#         else:
#             folder_metadata = {
#                 'name': folder_name,
#                 'mimeType': 'application/vnd.google-apps.folder'
#             }
#             folder = service.files().create(body=folder_metadata, fields='id').execute()
#             folder_id = folder.get('id')
#             print(f"创建新文件夹: {folder_name} (ID: {folder_id})")
    
#     # 上传文件
#     file_name = os.path.basename(file_path)
#     file_metadata = {
#         'name': file_name
#     }
    
#     if folder_id:
#         file_metadata['parents'] = [folder_id]
    
#     media = MediaFileUpload(file_path, resumable=True)
    
#     print(f"正在上传文件: {file_name}")
#     file = service.files().create(
#         body=file_metadata,
#         media_body=media,
#         fields='id,name,webViewLink'
#     ).execute()
    
#     print(f"文件上传成功!")
#     print(f"文件ID: {file.get('id')}")
#     print(f"查看链接: {file.get('webViewLink')}")
    
#     return file

# --- 1. 设置生成图的参数 ---
# === 批次采样配置 ===
# 用户可根据需要调整以下参数：
n_nodes = 100  # 节点数
n_features = 50  # 特征维度
n_samples = 2000  # 总采样次数 - 可设置为更大的值如10000, 50000等
batch_size = 50  # 批次大小 - 建议根据内存情况调整(10-100)
connection_density = 0.8  # 连接密度
inter_class_boost = 0.3  # 类间连接增强参数

# 批次采样优势：
# 1. 支持大数量采样（如10000+样本）而不会内存溢出
# 2. 提供实时进度显示和统计信息
# 3. 自动内存管理和垃圾回收
# 4. 错误处理和恢复机制
# 5. 可重现的随机种子设置

# 计算批次数量
n_batches = (n_samples + batch_size - 1) // batch_size  # 向上取整
print(f"批次采样配置:")
print(f"  - 总样本数: {n_samples}")
print(f"  - 批次大小: {batch_size}")
print(f"  - 批次数量: {n_batches}")
print(f"  - 最后一批样本数: {n_samples - (n_batches - 1) * batch_size}")
# 从[-1,1]均匀分布采样theta值
# === 批次采样函数 ===
def generate_batch_parameters(batch_start, batch_end):
    """为指定批次生成参数"""
    batch_size_actual = batch_end - batch_start
    
    # 设置随机种子以确保可重现性
    np.random.seed(42 + batch_start)
    random.seed(42 + batch_start)
    
    theta_values = np.random.uniform(-0.6, 0.5, batch_size_actual)
    connection_density_values = np.random.uniform(0.5, 0.7, batch_size_actual)
    inter_class_boost_values = np.random.uniform(0.1, 0.2, batch_size_actual)
    num_classes_values = [random.randint(3, 8) for _ in range(batch_size_actual)]
    
    return theta_values, connection_density_values, inter_class_boost_values, num_classes_values

def process_single_sample(sample_idx, theta, num_classes, connection_density, inter_class_boost, 
                         n_nodes, n_features, root_path):
    """处理单个样本的生成和保存"""
    print(f"\n--> Sample {sample_idx + 1}: Generating data for theta = {theta:.4f}, num_classes = {num_classes}...")
    print(f"    connection_density = {connection_density:.4f}, inter_class_boost = {inter_class_boost:.4f}")
    
    try:
        # 生成参数
        Lambda, mu = _parameterized_multi_class(theta, p=n_features, n=n_nodes, num_classes=num_classes)
        print(f"    Lambda = {Lambda:.4f}, mu = {mu:.4f}")
        
        # 生成图数据
        raw_data_object = _multi_class_context_sbm_lpm(
            n=n_nodes, 
            Lambda=Lambda, 
            p=n_features, 
            mu=mu, 
            num_classes=num_classes,
            connection_density=connection_density,
            inter_class_boost=inter_class_boost,
        )
        
        if raw_data_object is None:
            print(f"    ERROR: Function returned None for sample {sample_idx + 1}!")
            return None
        
        # 保存数据
        raw_file_name = f"multi_class_csbm_theta_{theta:.4f}_classes_{num_classes}_sample_{sample_idx + 1}"
        pt_file_path = osp.join(root_path, f"{raw_file_name}.pt")
        torch.save(raw_data_object, pt_file_path)
        print(f"    Raw data saved to: {pt_file_path}")
        
        # 打印统计信息
        graph_data = raw_data_object
        print(f"    Sample {sample_idx + 1} - Nodes: {graph_data.num_nodes}, Edges: {graph_data.num_edges}, Features: {graph_data.x.shape[1]}")
        print(f"    Sample {sample_idx + 1} - Number of classes: {len(torch.unique(graph_data.y))}")
        
        # 计算同质性
        edge_index = graph_data.edge_index.numpy()
        total_edges = edge_index.shape[1] // 2
        same_class_edges = 0
        for e in range(0, edge_index.shape[1], 2):
            src, dst = edge_index[0, e], edge_index[1, e]
            if graph_data.y[src] == graph_data.y[dst]:
                same_class_edges += 1
        
        diff_class_edges = total_edges - same_class_edges
        homophily = same_class_edges / total_edges if total_edges > 0 else 0
        print(f"    Sample {sample_idx + 1} - Homophily: {homophily:.3f} (Same: {same_class_edges}, Diff: {diff_class_edges})")
        
        return {
            'sample_idx': sample_idx,
            'theta': theta,
            'num_classes': num_classes,
            'connection_density': connection_density,
            'inter_class_boost': inter_class_boost,
            'file_path': pt_file_path,
            'homophily': homophily,
            'num_nodes': graph_data.num_nodes,
            'num_edges': graph_data.num_edges
        }
        
    except Exception as e:
        print(f"    ERROR: Exception occurred for sample {sample_idx + 1}: {e}")
        import traceback
        traceback.print_exc()
        return None

# # --- 2. 初始化Google Drive服务 ---
# print("\n初始化Google Drive服务...")
# try:
#     drive_service = authenticate_google_drive()
#     print("Google Drive认证成功!")
    
#     # 创建时间戳用于文件夹命名
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     folder_name = f"Multi_Class_Graphs_{timestamp}"
#     print(f"将使用文件夹: {folder_name}")
# except Exception as e:
#     print(f"Google Drive认证失败: {e}")
#     print("将跳过上传功能，仅保存到本地")
#     drive_service = None
#     folder_name = None

# --- 3. 批次生成多类别图数据 ---
root_path = './my_multi_class_datasets'  # 保存路径

# 确保保存目录存在
import os
if not osp.exists(root_path):
    os.makedirs(root_path)

# 初始化统计信息
total_generated = 0
total_failed = 0
batch_results = []

print(f"\n{'='*60}")
print(f"开始批次采样生成")
print(f"{'='*60}")

# 批次处理主循环
for batch_idx in range(n_batches):
    batch_start = batch_idx * batch_size
    batch_end = min((batch_idx + 1) * batch_size, n_samples)
    current_batch_size = batch_end - batch_start
    
    print(f"\n{'='*40}")
    print(f"批次 {batch_idx + 1}/{n_batches}")
    print(f"样本范围: {batch_start + 1} - {batch_end}")
    print(f"批次大小: {current_batch_size}")
    print(f"{'='*40}")
    
    # 生成当前批次的参数
    theta_values, connection_density_values, inter_class_boost_values, num_classes_values = generate_batch_parameters(batch_start, batch_end)
    
    # 处理当前批次的每个样本
    batch_success = 0
    batch_failed = 0
    
    for i, (theta, num_classes, connection_density, inter_class_boost) in enumerate(zip(
        theta_values, num_classes_values, connection_density_values, inter_class_boost_values)):
        
        sample_idx = batch_start + i
        result = process_single_sample(
            sample_idx, theta, num_classes, connection_density, inter_class_boost,
            n_nodes, n_features, root_path
        )
        
        if result is not None:
            batch_results.append(result)
            batch_success += 1
            total_generated += 1
        else:
            batch_failed += 1
            total_failed += 1
    
    # 批次完成统计
    print(f"\n批次 {batch_idx + 1} 完成:")
    print(f"  - 成功生成: {batch_success}/{current_batch_size}")
    print(f"  - 失败: {batch_failed}/{current_batch_size}")
    print(f"  - 成功率: {batch_success/current_batch_size:.1%}")
    
    # 内存清理（Python垃圾回收）
    import gc
    gc.collect()
    
    # 显示总体进度
    progress = (batch_idx + 1) / n_batches
    print(f"\n总体进度: {progress:.1%} ({batch_idx + 1}/{n_batches} 批次完成)")
    print(f"累计生成: {total_generated} 个样本")
    if total_failed > 0:
        print(f"累计失败: {total_failed} 个样本")
    
    # # 上传到Google Drive并删除本地文件
    # if drive_service is not None:
    #     try:
    #         print(f"\n正在上传文件到Google Drive...")
    #         uploaded_file = upload_single_file_to_drive(drive_service, pt_file_path, folder_name)
    #         uploaded_files.append({
    #             'local_path': pt_file_path,
    #             'file_info': uploaded_file,
    #             'dataset_info': f"theta_{theta:.4f}_classes_{num_classes}"
    #         })
            
    #         # 上传成功后删除本地文件
    #         os.remove(pt_file_path)
    #         print(f"本地文件已删除: {pt_file_path}")
    #         print(f"✓ 数据集 {i+1} 已成功上传并清理")
            
    #     except Exception as e:
    #         print(f"✗ 上传文件时出错: {e}")
    #         print(f"本地文件保留: {pt_file_path}")
    # else:
    #     print(f"跳过上传，文件保存在本地: {pt_file_path}")
    
# --- 4. 生成完成统计和分析 ---
print(f"\n{'='*60}")
print(f"批次生成完成统计")
print(f"{'='*60}")

print(f"\n总体统计:")
print(f"  - 目标样本数: {n_samples}")
print(f"  - 成功生成: {total_generated}")
print(f"  - 失败数量: {total_failed}")
print(f"  - 总体成功率: {total_generated/n_samples:.1%}")

if batch_results:
    # 分析生成的数据集
    print(f"\n数据集分析:")
    
    # 统计theta分布
    theta_list = [r['theta'] for r in batch_results]
    print(f"  - Theta范围: [{min(theta_list):.3f}, {max(theta_list):.3f}]")
    print(f"  - Theta均值: {np.mean(theta_list):.3f}")
    
    # 统计类别数分布
    classes_list = [r['num_classes'] for r in batch_results]
    unique_classes = sorted(set(classes_list))
    print(f"  - 类别数范围: {unique_classes}")
    for nc in unique_classes:
        count = classes_list.count(nc)
        print(f"    * {nc}类: {count} 个数据集 ({count/len(batch_results):.1%})")
    
    # 统计同质性分布
    homophily_list = [r['homophily'] for r in batch_results]
    print(f"  - 同质性范围: [{min(homophily_list):.3f}, {max(homophily_list):.3f}]")
    print(f"  - 同质性均值: {np.mean(homophily_list):.3f}")
    
    # 统计图大小
    nodes_list = [r['num_nodes'] for r in batch_results]
    edges_list = [r['num_edges'] for r in batch_results]
    print(f"  - 节点数: {nodes_list[0]} (固定)")
    print(f"  - 边数范围: [{min(edges_list)}, {max(edges_list)}]")
    print(f"  - 平均边数: {np.mean(edges_list):.1f}")
    
    print(f"\n所有数据集已保存到: {root_path}")
else:
    print(f"\n警告: 没有成功生成任何数据集!")
    


print(f"\n{'='*60}")
print(f"批次采样数据生成任务完成!")
print(f"{'='*60}")