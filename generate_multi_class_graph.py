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
n_nodes = 100  # 节点数
n_features = 50  # 特征维度
n_samples = 5  # 采样次数
connection_density = 0.8  # 连接密度
inter_class_boost = 0.3  # 类间连接增强参数
# 从[-1,1]均匀分布采样theta值
theta_values = np.random.uniform(-0.6, 0.5, n_samples)
connection_density_values = np.random.uniform(0.5, 0.7, n_samples)  # 连接密度
inter_class_boost_values = np.random.uniform(0.1, 0.2, n_samples)  # 类间连接增强参数
n_node = np.random.randint(80,300)  # 节点数
n_features = np.random.randint(60,256)
# 从[3,8]均匀分布采样类别数
num_classes_values = [random.randint(3, 8) for _ in range(n_samples)]

print(f"Sampled theta values: {theta_values}")
print(f"Sampled num_classes values: {num_classes_values}")

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

# --- 3. 为每个theta值和类别数生成图数据 ---
root_path = './my_multi_class_datasets'  # 保存路径
datasets = []
uploaded_files = []  # 记录上传的文件

for i, (theta, num_classes,connection_density,inter_class_boost) in enumerate(zip(theta_values, num_classes_values,
                                                                                  connection_density_values,inter_class_boost_values)):
    print(f"\n--> Step {i+1}: Generating data for theta = {theta:.4f}, num_classes = {num_classes}...")
    print(f"    connection_density = {connection_density:.4f}, inter_class_boost = {inter_class_boost:.4f}")
    
    try:
        # 使用 theta 计算 Lambda 和 mu，考虑类别数
        Lambda, mu = _parameterized_multi_class(theta, p=n_features, n=n_nodes, num_classes=num_classes,
                                                )
        
        print(f"    Lambda = {Lambda:.4f}, mu = {mu:.4f}")
        
        # 生成多类别 cSBM 图数据对象
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
            print("    ERROR: Function returned None!")
            continue
            
    except Exception as e:
        print(f"    ERROR: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        continue
    
    # 设置保存的根目录和文件名
    raw_file_name = f"multi_class_csbm_theta_{theta:.4f}_classes_{num_classes}"
    
    # 创建保存目录
    save_dir = osp.join(root_path)
    #os.makedirs(save_dir, exist_ok=True)
    
    # 保存为 PyTorch Geometric .pt 文件
    pt_file_path = osp.join(save_dir, f"{raw_file_name}.pt")
    torch.save(raw_data_object, pt_file_path)
    print(f"Raw data saved to: {pt_file_path}")
    
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
    
    # 验证生成的数据
    graph_data = raw_data_object
    print(f"Dataset {i+1} - Nodes: {graph_data.num_nodes}, Edges: {graph_data.num_edges}, Features: {graph_data.x.shape[1]}")
    print(f"Dataset {i+1} - Number of classes: {len(torch.unique(graph_data.y))}")
    
    # 打印每个类别的节点数量
    for c in range(num_classes):
        class_count = (graph_data.y == c).sum().item()
        print(f"  - Class {c}: {class_count} nodes")
    
    
    # 检查边的分布
    edge_index = graph_data.edge_index.numpy()
    total_edges = edge_index.shape[1] // 2  # 因为是无向图，所以除以2
    
    # 计算同类边和异类边的数量
    same_class_edges = 0
    for e in range(0, edge_index.shape[1], 2):  # 每两条边（因为是无向图）
        src, dst = edge_index[0, e], edge_index[1, e]
        if graph_data.y[src] == graph_data.y[dst]:
            same_class_edges += 1
    
    diff_class_edges = total_edges - same_class_edges
    print(f"Dataset {i+1} - Same class edges: {same_class_edges} ({same_class_edges/total_edges:.2%}), "
          f"Different class edges: {diff_class_edges} ({diff_class_edges/total_edges:.2%})")
    
    # 将数据添加到数据集列表
    datasets.append(raw_data_object)
    
    # 可视化图结构（仅对小图或采样的子图进行可视化）
    if n_nodes <= 1000:  # 对于较大的图，可以考虑采样一个子图
        try:
            # 将PyG数据转换为NetworkX图
            G = to_networkx(raw_data_object, to_undirected=True)
            
            # 处理节点采样和颜色
            sampled_nodes = None
            if n_nodes > 100:
                nodes = list(G.nodes())
                sampled_nodes = random.sample(nodes, 100)  # 采样100个节点
                G = G.subgraph(sampled_nodes)
                node_colors = [int(raw_data_object.y[n].item()) for n in sampled_nodes]
                print(f"    采样了 {len(sampled_nodes)} 个节点用于可视化")
            else:
                node_colors = [int(raw_data_object.y[n].item()) for n in range(raw_data_object.num_nodes)]
            
            # 创建图形 - 左右对比：潜在坐标 vs Spring布局
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # === 左图：使用潜在坐标布局 ===
            pos_latent = {}
            
            # 获取和处理潜在坐标
            if hasattr(raw_data_object, 'latent_pos') and raw_data_object.latent_pos is not None:
                latent_coords = raw_data_object.latent_pos.numpy() if hasattr(raw_data_object.latent_pos, 'numpy') else raw_data_object.latent_pos
                
                print(f"    潜在空间维度: {latent_coords.shape[1]}D")
                
                # 根据潜在空间维度处理坐标
                if latent_coords.shape[1] == 2:
                    # 2D潜在空间，直接使用
                    if sampled_nodes is not None:
                        for i, node in enumerate(sampled_nodes):
                            pos_latent[i] = latent_coords[node]
                    else:
                        for i in range(len(latent_coords)):
                            pos_latent[i] = latent_coords[i]
                    print(f"    直接使用2D潜在坐标")
                else:
                    # 高维潜在空间，使用PCA降维到2D
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    
                    if sampled_nodes is not None:
                        # 对采样节点的潜在坐标进行PCA
                        sampled_coords = latent_coords[sampled_nodes]
                        coords_2d = pca.fit_transform(sampled_coords)
                        for i in range(len(coords_2d)):
                            pos_latent[i] = coords_2d[i]
                    else:
                        # 对所有节点进行PCA
                        coords_2d = pca.fit_transform(latent_coords)
                        for i in range(len(coords_2d)):
                            pos_latent[i] = coords_2d[i]
                    
                    print(f"    PCA降维: {latent_coords.shape[1]}D -> 2D, 解释方差比: {pca.explained_variance_ratio_.sum():.3f}")
            else:
                # 如果没有潜在坐标，使用spring布局作为备选
                pos_latent = nx.spring_layout(G, seed=42)
                print("    警告: 没有找到潜在坐标，使用Spring布局代替")
            
            # === 右图：使用传统spring布局 ===
            pos_spring = nx.spring_layout(G, seed=42)
            
            # 绘制左图：潜在坐标布局
            nodes1 = nx.draw_networkx_nodes(G, pos_latent, node_size=40, 
                                  node_color=node_colors, 
                                  cmap=plt.cm.Set3, 
                                  vmin=0, 
                                  vmax=num_classes-1,
                                  ax=ax1, alpha=0.8)
            
            nx.draw_networkx_edges(G, pos_latent, alpha=0.2, ax=ax1, width=0.5)
            
            # 计算同质性用于显示
            edge_index = raw_data_object.edge_index.numpy()
            total_edges = edge_index.shape[1] // 2
            same_class_edges = 0
            for e in range(0, edge_index.shape[1], 2):
                src, dst = edge_index[0, e], edge_index[1, e]
                if raw_data_object.y[src] == raw_data_object.y[dst]:
                    same_class_edges += 1
            homophily = same_class_edges / total_edges if total_edges > 0 else 0
            
            ax1.set_title(f"潜在坐标布局\n(theta={theta:.3f}, classes={num_classes}, homophily={homophily:.2f})", 
                         fontsize=12)
            ax1.set_xlabel("Latent Dimension 1")
            ax1.set_ylabel("Latent Dimension 2")
            ax1.grid(True, alpha=0.3)
            
            # 绘制右图：Spring布局
            nodes2 = nx.draw_networkx_nodes(G, pos_spring, node_size=40, 
                                  node_color=node_colors, 
                                  cmap=plt.cm.Set3, 
                                  vmin=0, 
                                  vmax=num_classes-1,
                                  ax=ax2, alpha=0.8)
            
            nx.draw_networkx_edges(G, pos_spring, alpha=0.2, ax=ax2, width=0.5)
            
            ax2.set_title(f"Spring布局\n(theta={theta:.3f}, classes={num_classes}, homophily={homophily:.2f})", 
                         fontsize=12)
            ax2.set_xlabel("Spring Layout X")
            ax2.set_ylabel("Spring Layout Y")
            ax2.grid(True, alpha=0.3)
            
            # 添加颜色条
            #cbar = plt.colorbar(nodes1, ax=[ax1, ax2], label='Class', shrink=0.8)
            #cbar.set_ticks(range(num_classes))
            
            # 如果有类别中心信息，在潜在坐标图上标出
            if hasattr(raw_data_object, 'class_centers') and raw_data_object.class_centers is not None:
                class_centers = raw_data_object.class_centers
                
                # 处理类别中心的维度
                if class_centers.shape[1] == 2:
                    centers_2d = class_centers
                elif 'pca' in locals():
                    # 使用与节点相同的PCA变换
                    centers_2d = pca.transform(class_centers)
                else:
                    centers_2d = class_centers[:, :2]  # 取前两维
                
                # 在潜在坐标图上绘制类别中心
                ax1.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                           c='red', marker='X', s=200, linewidths=3, 
                           edgecolors='black', label='Class Centers', alpha=1.0, zorder=10)
                ax1.legend()
                
                print(f"    类别中心已标注在潜在坐标图上")
            
            plt.tight_layout()
            
            # 保存图像
            viz_path = osp.join(root_path, f"latent_viz_theta_{theta:.4f}_classes_{num_classes}.png")
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"潜在坐标对比可视化已保存到: {viz_path}")
            
        except Exception as e:
            print(f"Warning: Could not visualize graph: {e}")
            import traceback
            traceback.print_exc()

# # --- 4. 生成完成报告 ---
# print(f"\n" + "=" * 60)
# print(f"数据生成和上传完成报告")
# print(f"=" * 60)

# print(f"\n成功生成 {len(datasets)} 个多类别数据集:")
# for i, (theta, num_classes) in enumerate(zip(theta_values, num_classes_values)):
#     print(f"  - 数据集 {i+1}: theta = {theta:.4f}, classes = {num_classes}")

# if drive_service is not None and uploaded_files:
#     print(f"\n✓ Google Drive上传统计:")
#     print(f"  - 成功上传文件数: {len(uploaded_files)}")
#     print(f"  - Google Drive文件夹: {folder_name}")
#     print(f"\n上传的文件详情:")
#     for item in uploaded_files:
#         print(f"  • {item['dataset_info']}")
#         print(f"    文件ID: {item['file_info'].get('id')}")
#         print(f"    查看链接: {item['file_info'].get('webViewLink')}")
    
#     # 保存上传记录
#     if 'timestamp' in locals():
#         upload_record = {
#             'timestamp': timestamp,
#             'folder_name': folder_name,
#             'uploaded_files': uploaded_files,
#             'generation_params': {
#                 'n_samples': n_samples,
#                 'n_nodes': n_nodes,
#                 'n_features': n_features,
#                 'theta_values': theta_values.tolist(),
#                 'num_classes_values': num_classes_values
#             },
#             'upload_date': datetime.now().isoformat()
#         }
        
#         record_file = f'upload_record_{timestamp}.json'
#         with open(record_file, 'w', encoding='utf-8') as f:
#             json.dump(upload_record, f, ensure_ascii=False, indent=2)
        
#         print(f"\n上传记录已保存到: {record_file}")
    
#     print(f"\n✓ 所有本地文件已清理，数据已安全上传到Google Drive!")
# else:
#     print(f"\n本地文件保存位置: {root_path}")
#     if drive_service is None:
#         print(f"注意: 由于Google Drive认证失败，文件仅保存在本地")

# print(f"\n数据生成任务完成!")