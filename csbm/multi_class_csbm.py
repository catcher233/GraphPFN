import numpy as np
from torch_geometric.data import Data
import torch

import numpy as np
import torch
from torch_geometric.data import Data

def _multi_class_context_sbm_lpm(n, Lambda, p, mu, num_classes=2, latent_dim=2, sigma=1.2, within_class_scale='auto',
                                 connection_density=0.1, inter_class_boost=0.3):
    """
    多类别连续潜在位置模型 + cSBM 风格特征生成函数
    
    该函数结合了连续潜在位置模型(Latent Position Model, LPM)和上下文随机块模型(cSBM)的特点，
    生成具有多个类别的图数据。边的生成采用混合策略：50%概率使用Bernoulli分布，50%使用Poisson分布。
    
    参数说明:
    n (int): 图中节点的总数
    d (float): 图的平均度数，用于控制图的稠密程度
    Lambda (float): 结构参数，控制社群内外连接的紧密程度
    p (int): 每个节点特征向量的维度
    mu (float): 特征信号强度，控制类别特征的区分度
    num_classes (int): 类别数量，默认为2，范围限制在[2, 10]
    train_percent (float): 训练集所占比例，默认为0.6
    latent_dim (int): 潜在空间的维度，默认为2
    sigma (float): 高斯核函数的标准差参数，控制距离衰减速度，默认为1.0
    inter_class_boost (float): 跨类别连接增强参数，范围[0.0, 1.0]，默认为0.0
                              值越大，不同类别节点间的连接概率越高
    
    返回值:
    torch_geometric.data.Data: 包含节点特征、边索引、标签和数据划分的图数据对象
    """
# 根据类别数量自适应调整潜在空间维度
    # 类别数越多，需要更高维的潜在空间来避免类别重叠
    if num_classes <= 4:
        latent_dim = 2  # 少数类别使用2维空间，便于可视化
    elif num_classes <= 8:
        latent_dim = 4  # 中等类别数使用4维空间，平衡表达能力和计算效率
    else:
        latent_dim = 5  # 多类别使用5维空间，提供足够的分离能力
    # ===== 步骤1: 节点标签分配 =====
    # 随机将n个节点分配到num_classes个类别中
    # 使用多项式分布确保每个类别至少有一个节点
    
    # 首先为每个类别分配至少一个节点
    y = np.zeros(n, dtype=int)          # 初始化节点标签数组
    
    # 为每个类别分配一个节点（确保所有类别都有代表）
    for class_idx in range(num_classes):
        y[class_idx] = class_idx
    
    # 对剩余的节点进行随机分配
    remaining_nodes = n - num_classes
    if remaining_nodes > 0:
        # 使用多项式分布随机分配剩余节点
        # 每个类别被选中的概率相等
        probabilities = np.ones(num_classes) / num_classes
        random_assignments = np.random.multinomial(remaining_nodes, probabilities)
        
        # 将随机分配的节点添加到对应类别
        current_idx = num_classes  # 从第num_classes个位置开始分配
        for class_idx in range(num_classes):
            class_size = random_assignments[class_idx]
            y[current_idx:current_idx + class_size] = class_idx
            current_idx += class_size
    
    # 打乱节点顺序，避免类别标签的顺序性偏差
    shuffle_indices = np.random.permutation(n)
    y = y[shuffle_indices]

    # ===== 步骤2: 生成连续潜在位置 =====
    # 在潜在空间中为每个节点分配位置坐标，同类节点在空间中聚集
    latent_pos = np.zeros((n, latent_dim))  # 初始化潜在位置矩阵 (n × latent_dim)
    
    # 为每个类别在潜在空间中随机选择一个中心点
    # 中心点坐标在[-5, 5]范围内均匀分布
    class_centers = np.random.uniform(-4, 4, size=(num_classes, latent_dim))
 # 自适应调整scale的函数
    def adaptive_scale(n, num_classes, latent_dim):
        base_scale = 1.0
        size_factor = min(1.5, np.log(n) / np.log(1000))
        class_factor = max(0.7, 2.0 / num_classes)
        dim_factor = max(0.8, latent_dim / 3.0)
        final_scale = base_scale * size_factor * class_factor * dim_factor
        return np.clip(final_scale, 1.2, 2)
    
    
    # 自适应调整scale
    if within_class_scale == 'auto':
        scale = adaptive_scale(n, num_classes, latent_dim)
    else:
        scale = within_class_scale
    # 为每个节点生成潜在位置
    for i in range(n):
        # 节点i的位置围绕其所属类别的中心点进行高斯分布
        # scale=1.0控制同类节点间的散布程度，值越小聚集越紧密
        latent_pos[i] = np.random.normal(class_centers[y[i]], scale=scale)
    print(f"Adaptive within-class scale: {scale:.3f}")

    # ===== 步骤3: 生成边 =====
    # 使用混合策略生成图的边：基于潜在位置距离的概率模型
    edge_index = [[], []]  # 存储边的起点和终点索引

    def adaptive_sigma(latent_dim, num_classes, class_centers=None):
        """根据潜在空间特性自适应调整sigma"""
        
        # 基础sigma值
        base_sigma = 1.2
        
        # 根据潜在空间维度调整
        # 高维空间中距离概念发生变化，需要更大的sigma
        dim_factor = np.sqrt(latent_dim / 2.0)
        
        # 根据类别数调整
        # 类别越多，需要更精确的距离控制
        class_factor = max(0.7, 2.0 / np.sqrt(num_classes))
        
        # 如果提供了类别中心，根据中心间距离调整
        if class_centers is not None:
            # 计算类别中心间的平均距离
            center_distances = []
            for i in range(len(class_centers)):
                for j in range(i+1, len(class_centers)):
                    dist = np.linalg.norm(class_centers[i] - class_centers[j])
                    center_distances.append(dist)
            
            avg_center_dist = np.mean(center_distances)
            # 根据中心间距离调整：距离越远，需要更大的sigma
            distance_factor = min(2.0, avg_center_dist / 5.0)
        else:
            distance_factor = 1.0
        
        final_sigma = base_sigma * dim_factor * class_factor * distance_factor
        
        # 限制在合理范围内
        return np.clip(final_sigma, 0.5, 3.0)
    
    # 自适应调整sigma
    sigma = adaptive_sigma(latent_dim, num_classes, class_centers)
    print(f"Adaptive sigma: {sigma:.3f}")
    # 遍历所有可能的节点对(i,j)，其中i < j
    for i in range(n-1):
        for j in range(i+1, n):
            # 计算节点i和j在潜在空间中的欧几里得距离
            dist = np.linalg.norm(latent_pos[i] - latent_pos[j])
            
            # 使用高斯核函数计算基础连接强度/概率
            # sigma控制距离衰减速度：sigma越小，距离对连接概率的影响越大
            base_prob = np.exp(-dist**2 / (2 * sigma**2))
            
            # 检查节点i和j是否属于不同类别
            if y[i] == y[j]:
                final_prob = base_prob * (1.0+Lambda)*connection_density  #
            else:
                final_prob = base_prob * (1.0-Lambda)*connection_density*(1+inter_class_boost) #
            final_prob = np.clip(final_prob, 0.0, 0.8)  # 确保概率在[0,1]范围内
            Flip = np.random.binomial(1, final_prob)

            # 如果生成了边（Flip > 0），则添加到边列表中
            if Flip > 0:
                # 添加无向边：i->j 和 j->i
                edge_index[0].append(i)
                edge_index[1].append(j)
                edge_index[0].append(j)
                edge_index[1].append(i)

    # ===== 步骤4: 从潜在位置生成节点特征 (向量化版本) =====
    # 基于潜在位置坐标生成节点特征，保持空间连续性和类别区分性
    
    # 初始化节点特征矩阵 (n × p)
    x = np.zeros((n, p))
    
    # 创建从潜在空间到特征空间的随机映射矩阵 W (latent_dim × p)
    # 使用标准差为 1/sqrt(latent_dim) 的高斯分布，确保映射的数值稳定性
    W = np.random.normal(0, 1/np.sqrt(latent_dim), size=(latent_dim, p))

    # 通过矩阵乘法一次性计算所有节点的信号部分
    # latent_pos @ W: (n × latent_dim) @ (latent_dim × p) -> (n × p)
    # 这种向量化操作比逐个节点计算效率更高
    all_signals = latent_pos @ W

    # 一次性生成所有节点的随机噪声矩阵 (n × p)
    # 噪声服从标准正态分布，用于增加特征的随机性
    all_Z = np.random.normal(0, 1, size=(n, p))

    # 计算最终的节点特征：信号部分 + 噪声部分
    # sqrt(mu/n) * all_signals: 缩放的信号部分，mu控制信号强度
    # all_Z / sqrt(p): 标准化的噪声部分，确保噪声不会随特征维度增长
    x = np.sqrt(mu / n) * all_signals + all_Z / np.sqrt(p)
    
    # 添加类别特定的偏置，增强类别间的区分度
    # 每个类别有独特的偏置向量，进一步强化类别特征
    class_bias = np.random.normal(0, 0.5, size=(num_classes, p))
    
    # 为每个节点添加其所属类别的偏置
    for i in range(n):
        # 偏置强度与信号强度成比例，保持特征的相对重要性
        x[i] += class_bias[y[i]] * np.sqrt(mu / n)
    # ===== 步骤5: 创建PyTorch Geometric图对象 =====
    # 将numpy数组转换为PyTorch张量，创建符合PyG标准的图数据结构
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),        # 节点特征矩阵 (n × p)
        edge_index=torch.tensor(edge_index, dtype=torch.long),  # 边索引矩阵 (2 × num_edges)
        y=torch.tensor(y, dtype=torch.int64)           # 节点标签向量 (n,)
    )
    
    # 合并重复边并按索引排序，优化图数据结构的存储和访问效率
    # coalesce()函数会自动处理重复边并确保边索引的有序性
    data.coalesce()


    # ===== 步骤6: 存储生成参数到图对象中 =====
    # 将所有生成参数保存到data对象中，便于后续分析、调试和实验复现
    data.Lambda = Lambda                    # 结构参数，控制社群内外连接的紧密程度
    data.mu = mu                           # 特征信号强度参数，控制类别特征的区分度
    data.n = n                             # 节点总数
    data.p = p                             # 节点特征向量的维度
    data.num_classes = num_classes         # 图中的类别数量
    data.latent_pos = latent_pos          # 节点在潜在空间中的坐标位置 (n × latent_dim)
    data.latent_dim = latent_dim          # 潜在空间的维度
    data.sigma = sigma                     # 高斯核函数的标准差参数
    data.class_centers = class_centers     # 各类别在潜在空间中的中心点坐标

    return data  # 返回包含完整图结构、特征和元数据的图对象


def _parameterized_multi_class(theta, p, n, num_classes, epsilon=0.1):
    """
    通过单一参数 theta 来生成 Lambda 和 mu，支持多类别设置
    
    参数:
    theta: 一个在 [-1, 1] 范围内的控制参数
    p: 节点特征的维度
    n: 节点总数
    num_classes: 社群/类别数量
    epsilon: 一个小的正常数，用于确保根号内的值为正
    """
    from math import pi

    # 计算节点数 n 与特征维度 p 的比率 gamma
    gamma = n / p

    # 断言语句，确保输入的 theta 值在合法的 [-1, 1] 区间内
    assert (theta >= -1) and (theta <= 1)
    
    # 确保类别数在有效范围内
    num_classes = max(2, min(10, num_classes))

    # 使用 sin 函数，根据 theta 计算 Lambda (同质性/结构强度)
    Lambda = np.sqrt(1 + epsilon) * np.sin(theta * pi / 2)
    if Lambda > 1:
        Lambda = 1.0

    # 使用 cos 函数，根据 theta 计算 mu (社群信号/特征强度)
    # 对于多类别情况，可能需要调整 mu 的计算
    mu = np.sqrt(gamma * (1 + epsilon)) * np.cos(theta * pi / 2)
    
    # 对于多类别情况，可以根据类别数调整 mu
    # 类别越多，区分难度越大，可能需要更强的信号
    mu_adjustment = 1.0 + 0.1 * (num_classes - 2)  # 简单的线性调整
    mu = mu * mu_adjustment

    return Lambda, mu