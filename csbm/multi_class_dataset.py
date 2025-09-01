import os.path as osp
import os
import pickle
import torch
from torch_geometric.data import InMemoryDataset
from csbm.multi_class_csbm import _multi_class_context_sbm, _parameterized_multi_class

class MultiClassSynCSBM(InMemoryDataset):
    """
    多类别合成上下文随机块模型(cSBM)数据集
    
    参数:
    root: 数据集的根目录
    name: 数据集名称
    transform: 数据转换函数
    pre_transform: 预处理转换函数
    """
    def __init__(self, root, name=None,
                 transform=None, pre_transform=None):
        self.name = name
        super(MultiClassSynCSBM, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.pickle')]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # 不需要下载，数据是合成的
        pass

    def process(self):
        # 读取原始数据
        raw_data_path = osp.join(self.raw_dir, self.raw_file_names[0])
        with open(raw_data_path, 'rb') as f:
            data = pickle.load(f)
        
        # 如果需要预处理
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # 保存处理后的数据
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


def generate_multi_class_csbm(n=1000, p=50, theta=0.5, num_classes=3):
    """
    生成一个多类别cSBM数据集并返回
    
    参数:
    n: 节点数
    d: 平均度
    p: 特征维度
    theta: 控制参数 [-1, 1]
    num_classes: 类别数 [2, 10]
    train_percent: 训练集比例
    
    返回:
    data: PyTorch Geometric数据对象
    """
    # 确保类别数在有效范围内
    num_classes = max(2, min(10, int(num_classes)))
    
    # 计算Lambda和mu
    Lambda, mu = _parameterized_multi_class(theta, p, n, num_classes)
    
    # 生成数据
    data = _multi_class_context_sbm(n, Lambda, p, mu, num_classes)
    
    return data