import numpy as np
from torch_geometric.data import Data
import pickle
from datetime import datetime
import os.path as osp
import os
import argparse
from torch_geometric.data import InMemoryDataset
import torch

DATASET_ROOT = os.path.split(__file__)[0] + '/dataset/csbm'


def _index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


# 定义一个函数，用于对图数据进行分层随机划分，创建训练、验证和测试集
def _random_splits(data, num_classes, train_per_class, val_per_class):
    # 初始化一个空列表，用于存储每个类别下所有节点的索引
    class_indexes = []

    # 遍历每一个类别
    for i in range(num_classes):
        # 找到所有标签为当前类别 i 的节点的索引
        index = (data.y == i).nonzero().view(-1)

        # 将当前类别下的节点索引进行随机打乱，以确保抽样的随机性
        index = index[torch.randperm(index.size(0))]

        # 将打乱后的索引列表添加到 class_indexes 中
        class_indexes.append(index)

    # 创建训练集索引：从每个类别打乱后的索引列表中，取出前 train_per_class 个索引，然后将它们合并
    train_index = torch.cat([i[0:train_per_class] for i in class_indexes], dim=0)

    # 创建验证集索引：从每个类别中，取出训练集之后、紧接着的 val_per_class 个索引，然后合并
    validate_index = torch.cat([i[train_per_class: train_per_class + val_per_class] for i in class_indexes], dim=0)

    # 创建测试集索引：从每个类别中，取出训练集和验证集之外所有剩余的索引，然后合并
    test_index = torch.cat([i[train_per_class + val_per_class:] for i in class_indexes], dim=0)
    #对拼接后的索引进行一次全局打乱
    train_index = train_index[torch.randperm(train_index.size(0))]
    validate_index = validate_index[torch.randperm(validate_index.size(0))]
    test_index = test_index[torch.randperm(test_index.size(0))]


    # 将训练集索引列表转换为一个布尔掩码（mask），并将其赋值给 data 对象的 train_mask 属性
    data.train_mask = _index_to_mask(train_index, size=data.num_nodes)

    # 将验证集索引列表转换为布尔掩码，并赋值给 val_mask 属性
    data.val_mask = _index_to_mask(validate_index, size=data.num_nodes)

    # 将测试集索引列表转换为布尔掩码，并赋值给 test_mask 属性
    data.test_mask = _index_to_mask(test_index, size=data.num_nodes)

    # 返回更新了三个掩码属性的 data 对象
    return data


# 定义一个函数，用于生成 cSBM 图数据
# n: 节点总数, d: 节点的平均度数, Lambda: 控制社群内外连接紧密度的参数（>0时同质性更高）
# p: 节点特征的维度, mu: 节点特征中包含的社群信号强度, train_percent: 训练集所占的比例
def _context_sbm(n, d, Lambda, p, mu, train_percent=0.6):
    # 计算社群内部节点之间存在边的概率参数
    c_in = d + np.sqrt(d) * Lambda

    # 计算社群之间节点存在边的概率参数
    c_out = d - np.sqrt(d) * Lambda

    # 初始化一个长度为 n 的标签数组，初始值全为 1
    y = np.ones(n)

    # 将后半部分的节点的标签设置为 -1，从而将所有节点平分为两个社群
    y[int(n / 2) + 1:] = -1

    # 确保标签数组的数据类型为整数
    y = np.asarray(y, dtype=int)

    # 初始化一个空的列表，用于以 COO 格式存储图的边（[所有边的源节点], [所有边的目标节点]）
    edge_index = [[], []]

    # 使用两层循环遍历所有可能的节点对 (i, j)
    for i in range(n - 1):
        for j in range(i + 1, n):

            # 检查节点 i 和节点 j 是否属于同一个社群
            if y[i] * y[j] > 0:
                # 如果在同一社群，使用 c_in 概率进行一次伯努利试验（抛硬币），决定是否添加边
                Flip = np.random.binomial(1, c_in / n)
            else:
                # 如果不在同一社群，使用 c_out 概率进行伯努利试验
                Flip = np.random.binomial(1, c_out / n)

            # 如果试验结果为1，则添加边
            if Flip > 0.5:
                # 添加从 i 到 j 的边
                edge_index[0].append(i)
                edge_index[1].append(j)
                # 同时添加从 j 到 i 的边，以构建一个无向图
                edge_index[0].append(j)
                edge_index[1].append(i)

    # 初始化一个 n x p 大小的零矩阵，用于存放所有节点的特征
    x = np.zeros([n, p])

    # 创建一个基础的“信号”向量 u，它将嵌入到所有节点的特征中
    u = np.random.normal(0, 1 / np.sqrt(p), [1, p])

    # 循环遍历每个节点，为其生成特征向量
    for i in range(n):
        # 为当前节点生成一个随机的“噪声”向量 Z
        Z = np.random.normal(0, 1, [1, p])

        # 生成节点 i 的特征：它由“信号”部分和“噪声”部分组成
        # “信号”部分与节点的标签 y[i] 相关，其强度由 mu 控制
        x[i] = np.sqrt(mu / n) * y[i] * u + Z / np.sqrt(p)

    # 使用 PyTorch Geometric 的 Data 类来创建一个图数据对象
    # x: 节点特征, edge_index: 边的连接信息
    # y: 节点标签 (将 {-1, 1} 转换为 {0, 1} 格式)
    data = Data(x=torch.tensor(x, dtype=torch.float32),
                edge_index=torch.tensor(edge_index),
                y=torch.tensor((y + 1) // 2, dtype=torch.int64))

    # 对边索引进行合并操作，去除重复的边并进行排序，这是一个标准的预处理步骤
    data.coalesce()

    # 计算数据中的类别总数
    num_class = len(np.unique(y))

    # 根据总训练比例，计算出每个类别应该分配到的训练样本数量
    percls_trn = int(round(train_percent * n / num_class))
    
    # 计算每个类别的验证集样本数量（假设验证集占总数的20%）
    val_percent = 0.2
    percls_val = int(round(val_percent * n / num_class))
    
    # 确保训练集+验证集不会超过每个类别的总数，为测试集留出空间
    nodes_per_class = n // num_class
    if percls_trn + percls_val >= nodes_per_class:
        # 重新调整比例，确保有测试集
        percls_trn = int(nodes_per_class * 0.6)  # 60%用于训练
        percls_val = int(nodes_per_class * 0.2)  # 20%用于验证，剩余20%用于测试

    # 调用之前定义的 _random_splits 函数，为数据对象创建训练、验证和测试集的掩码
    data = _random_splits(data, num_class, percls_trn, percls_val)

    # 将生成该图所用的超参数也存入 data 对象中，方便后续记录和分析
    data.Lambda = Lambda
    data.mu = mu
    data.n = n
    data.p = p
    data.d = d
    data.train_percent = train_percent

    # 返回最终创建好的、包含所有信息的图数据对象
    return data


# 定义一个函数，用于通过单一参数 theta 来生成 Lambda 和 mu
# theta: 一个在 [-1, 1] 范围内的控制参数
# p: 节点特征的维度
# n: 节点总数
# epsilon: 一个小的正常数，用于确保根号内的值为正
def _parameterized_lambda_and_mu(theta, p, n, epsilon=0.1):
    # 从数学库中导入 pi (π)
    from math import pi

    # 计算节点数 n 与特征维度 p 的比率 gamma
    gamma = n / p

    # 断言语句，确保输入的 theta 值在合法的 [-1, 1] 区间内
    assert (theta >= -1) and (theta <= 1)

    # 使用 sin 函数，根据 theta 计算 Lambda (同质性/结构强度)
    # 当 theta 接近 1 或 -1 时, sin 值绝对值接近 1, Lambda 绝对值最大, 图的同质/异质性结构最明显
    # 当 theta 接近 0 时, sin 值接近 0, Lambda 接近 0, 图的社群结构不明显
    Lambda = np.sqrt(1 + epsilon) * np.sin(theta * pi / 2)

    # 使用 cos 函数，根据 theta 计算 mu (社群信号/特征强度)
    # 当 theta 接近 0 时, cos 值接近 1, mu 最大, 节点特征包含的社群信号最强
    # 当 theta 接近 1 或 -1 时, cos 值接近 0, mu 接近 0, 节点特征包含的社群信号最弱
    mu = np.sqrt(gamma * (1 + epsilon)) * np.cos(theta * pi / 2)

    # 返回计算出的 Lambda 和 mu
    return Lambda, mu

# 定义一个函数，用于将数据对象保存为 pickle 文件
# data: 需要被保存的 Python 对象
# p2root: 保存文件的根目录路径
# file_name: (可选) 自定义的文件名
def _save_data_to_pickle(data, p2root, file_name=None):
    # 获取当前的日期和时间
    now = datetime.now()

    # 将当前时间格式化成一个字符串，例如 "Jul_17_2025-18:51"
    surfix = now.strftime('%b_%d_%Y-%H:%M')

    # 检查用户是否提供了自定义文件名
    if file_name is None:
        # 如果没有提供文件名，则创建一个默认文件名，格式为 'cSBM_data_日期-时间'
        tmp_data_name = '_'.join(['cSBM_data', surfix])
    else:
        # 如果提供了文件名，则直接使用该文件名
        tmp_data_name = file_name

    # 使用 os.path.join 构造出最终文件的完整路径
    p2cSBM_data = osp.join(p2root, tmp_data_name)

    # 检查指定的根目录是否存在
    if not osp.isdir(p2root):
        # 如果目录不存在，则创建该目录
        os.makedirs(p2root)

    # 使用 'with' 语句打开文件，'bw' 表示以二进制(binary)写入(write)模式打开
    # 'with' 能确保文件在使用后被自动关闭
    with open(p2cSBM_data, 'bw') as f:
        # 使用 pickle.dump 将 data 对象序列化并写入到文件 f 中
        pickle.dump(data, f)

    # 函数返回最终保存文件的完整路径
    return p2cSBM_data

# 定义一个用于 cSBM 人造图的数据集类，它继承自 PyTorch Geometric 的 InMemoryDataset
# InMemoryDataset 是一个基类，适用于那些可以一次性完全加载到内存中的小数据集
class SynCSBM(InMemoryDataset):
    """用于 cSBM 人造图的数据集类。"""
    def __init__(self, root, name=None,
                 transform=None, pre_transform=None):
        # 直接使用传入的参数，不进行多余的路径拼接
        self.name = name
        super(SynCSBM, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # 将核心参数提取为顶层属性
        self.Lambda = self.data.Lambda.item()
        self.mu = self.data.mu.item()

    @property
    def raw_file_names(self):
        return [self.name]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # 这个方法理论上不应该被调用，因为我们会先生成原始文件
        raise Exception("Raw file not found. Please generate it first.")

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)