"""
网络动力学系统训练数据生成器
主要功能：根据方程字符串，生成大量不同拓扑结构和初始条件下的动力学仿真数据
用途：为神经网络模型提供训练数据，学习网络动力学的规律
"""

import os
import time
from copy import copy

import hydra              # 配置管理工具
import numpy as np        # 数值计算
import pandas as pd       # 数据处理
from sklearn.utils import shuffle  # 数据打乱
import pickle             # 数据序列化

import torch              # PyTorch深度学习框架
import torch.nn as nn     # 神经网络模块
from torch_scatter import scatter_sum  # 图神经网络相关操作

import adamod             # 优化器

# 导入自定义模块
from data_create.NetworkSystemInstances_new import GeneralDynamics  # 动力学系统仿真
from data_create.lib.Topo import Topo                              # 网络拓扑生成
from data_create.lib.InitCondition import InitCondition            # 初始条件生成
from string_create.Creation import Creation                         # 字符串创建工具
import nn_models                                                   # 神经网络模型

# ============== 设备配置 ==============
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # 使用GPU加速
else:
    device = torch.device("cpu")     # 使用CPU
# device = torch.device("cpu")  # 强制使用CPU（可选）
print("using device : ", device)


# ============== 数据生成器主类 ==============
class GenerateData:
    """
    网络动力学数据生成器
    
    功能：
    1. 从CSV文件加载方程字符串库
    2. 为每个方程生成多种网络拓扑和初始条件
    3. 通过数值仿真生成动力学轨迹数据
    4. 保存生成的训练数据
    """
    def __init__(
            self,
            cfg,                # Hydra配置对象
            eq_filename=None,   # 方程字符串文件路径
    ):
        """
        初始化数据生成器
        
        参数:
            cfg: 配置参数（包含时间、拓扑等设置）
            eq_filename: 包含方程字符串的CSV文件路径
        """
        
        # ========== 基本参数设置 ==========
        self.max_dim = 5  # 最大系统维度
        
        self.eq_filename = eq_filename  # 方程文件路径
        
        # ========== 加载方程数据集 ==========
        # 从CSV文件读取方程字符串
        # 文件格式：[编号, f方程字符串, g方程字符串, 状态标志]
        self.dataset = pd.read_csv(self.eq_filename, header=None,
                                   names=['no', 'f_eq_str_0', 'g_eq_str_0', 'state'])
        
        # ========== 数据扩增策略 ==========
        self.repeat_num = 1000  # 每个方程重复生成的次数
        
        # 将每个方程重复repeat_num次，增加数据多样性
        self.dataset = self.dataset.loc[self.dataset.index.repeat(self.repeat_num)]
        
        # 打乱数据集顺序，避免训练时的偏差
        self.dataset = shuffle(self.dataset)
        
        # ========== 拓扑控制参数 ==========
        self.same_topo_num = 100  # 使用相同拓扑的连续样本数量
                                 # 这样可以在相同网络结构下测试不同方程的表现
        
        # ========== 存储容器 ==========
        self.generated_data = []  # 存储生成的所有仿真数据

        # ========== 配置参数 ==========
        self.cfg = cfg            # 保存配置信息
        
        self.state_dim = 1        # 系统状态维度（每个节点的状态变量数）
        self.norm_state = True    # 是否对状态进行归一化处理


    def generate_data(self, ):
        """
        核心数据生成方法：为每个方程生成多样化的仿真数据
        
        工作流程：
        1. 遍历所有方程字符串
        2. 为每组方程生成不同的网络拓扑
        3. 创建随机初始条件
        4. 执行动力学仿真
        5. 收集成功的仿真结果
        """
        
        # ========== 初始化变量 ==========
        state_dim = self.state_dim     # 系统状态维度
        lineno = 0                     # 当前处理的数据集行号
        simu_ok_num = 0               # 成功仿真的数量
        start_time = time.time()       # 开始时间（用于计算总耗时）
        
        # ========== 主循环：遍历所有方程 ==========
        while lineno < len(self.dataset):
            
            # 提取当前方程的f函数和g函数字符串（支持多维系统）
            f_eq_str = [self.dataset.iloc[lineno]['f_eq_str_%s' % i] for i in range(state_dim)]
            g_eq_str = [self.dataset.iloc[lineno]['g_eq_str_%s' % i] for i in range(state_dim)]
            # f_eq_str = self.dataset.iloc[lineno]['f_eq_str']  # 单维度版本（已注释）
            # g_eq_str = self.dataset.iloc[lineno]['g_eq_str']  # 单维度版本（已注释）
            
            # 获取方程状态标志（用于过滤无效方程）
            state = self.dataset.iloc[lineno]['state']
            
            lineno += 1  # 移动到下一行

            # ========== 过滤无效方程 ==========
            if not bool(state):
                continue  # 跳过状态为False的方程
                
            # ========== 拓扑管理策略 ==========
            if simu_ok_num % self.same_topo_num == 0:
                # 每100个成功样本重新采样网络拓扑
                print('**resample topo**[ succeed number = %s (dataset lineno %s)]' % (simu_ok_num,lineno))
                
                # 随机采样网络规模：在[max_num/5, max_num]范围内
                N_sampled = np.random.randint(int(self.cfg.topo.max_num / 5), self.cfg.topo.max_num)
                
                # 随机选择网络拓扑类型（如小世界、无标度、随机等）
                topo_type_sampled = np.random.choice(a=self.cfg.topo.type_list, size=1)[0]
                
                # 生成网络拓扑对象
                topo = Topo(N_sampled, topo_type_sampled)
                N_sampled = topo.N  # 获取实际的网络节点数
            else:
                # 保持相同拓扑，测试不同方程在同一网络结构下的表现
                print('**keep same topo**[ succeed number = %s (dataset lineno %s)]' % (simu_ok_num,lineno))
            
            # print('lineno=', lineno)  # 调试信息（已注释）

            # ========== 生成初始条件 ==========
            # 为当前网络创建随机初始条件
            # lbs: 下界 [-5, -5, ...], ubs: 上界 [5, 5, ...]
            # num_sampling=1: 只生成一组初始条件
            init_cond = InitCondition(
                N_sampled,                            # 网络节点数
                state_dim,                           # 状态维度  
                lbs=torch.ones(state_dim) * (-5.),   # 初始状态下界
                ubs=torch.ones(state_dim) * 5.,      # 初始状态上界
                num_sampling=1                       # 采样次数
            )

            # ========== 创建动力学系统并进行仿真 ==========
            # 构建通用动力学系统对象
            s_ns = GeneralDynamics(
                N_sampled,          # 网络节点数
                state_dim,          # 状态维度
                topo_type_sampled,  # 拓扑类型
                f_eq_str,           # f函数字符串列表
                g_eq_str,           # g函数字符串列表
                topo,               # 拓扑对象
                init_cond           # 初始条件对象
            )
            
            # 执行数值仿真，生成动力学轨迹数据
            s_data = s_ns.simulating_data(
                self.cfg.t.start,                    # 仿真开始时间
                self.cfg.t.inc,                      # 时间步长
                self.cfg.t.end,                      # 仿真结束时间
                self.cfg.resample_init_condition,    # 是否重采样初始条件
                norm_state=self.norm_state           # 是否归一化状态
            )
            
            # 打印当前处理的方程（用于调试和监控）
            print(f_eq_str, g_eq_str)
            
            # ========== 处理仿真结果 ==========
            # 检查仿真是否成功
            if s_data is None:
                print('**Simulation failed!!!**')
                continue  # 仿真失败，跳过当前方程
            
            # 调试信息：打印数据范围（已注释）
            # print('s_data[self_diff][1].min(), max()',s_data['self_diff'][1].min(),s_data['self_diff'][1].max())
            # print('s_data[interact_diff][1].min(), max()',s_data['interact_diff'][1].min(),s_data['interact_diff'][1].max())

            # 保存成功的仿真数据
            self.generated_data.append(s_data)
            
            simu_ok_num += 1  # 增加成功计数
            
            # 输出成功信息和进度统计
            print('**Simulation succeed!!!**[ succeed number = %s (dataset lineno %s)]**, cost=%s' % 
                  (simu_ok_num, lineno, time.time()-start_time))
            

    
    # ========== 数据存储和加载方法 ==========         
    def saved_data_to_file(self, filename=None):
        """
        将生成的数据保存到文件
        
        参数:
            filename: 保存文件名（可选，默认根据配置自动生成）
        """
        if filename is None:
            # 自动生成文件名：原文件名 + 归一化标志
            filename = self.eq_filename + '_norm_state%s.pkl' % str(self.norm_state)
        
        # 使用pickle序列化保存数据
        f = open(filename, 'wb')
        pickle.dump(self.generated_data, f)
        f.close()
        
        print(f'Data saved to: {filename}')
        
    def load_data_from_file(self, filename):
        """
        从文件加载已生成的数据
        
        参数:
            filename: 数据文件路径
        """
        if os.path.isfile(filename):
            print('file exists, loading ...')
            with open(filename, 'rb') as f:
                self.generated_data = pickle.load(f)
                print('--ok')
        else:
            print('no file [%s] exists'%filename)
        

# ============== 主函数和程序入口 ==============

# 支持的配置文件类型：
# - Simulation_config: 通用仿真配置
# - MutualisticInteraction_config: 互利相互作用系统配置  
# - GeneRegulatory_config: 基因调控网络配置
@hydra.main(config_name="Simulation_config", version_base='1.2', config_path='configs')
def main(cfg):
    """
    主函数：执行完整的数据生成流程
    
    参数:
        cfg: Hydra配置对象，包含所有仿真参数
    
    执行流程：
    1. 创建数据生成器实例
    2. 执行数据生成过程
    3. 保存生成的数据到文件
    """
    
    # 创建数据生成器，指定方程文件路径
    generate_data = GenerateData(cfg, eq_filename='data/Simulation_dataset_100.csv')
    
    # 执行数据生成：遍历所有方程，进行仿真
    generate_data.generate_data()
    
    # 保存生成的数据到pickle文件
    generate_data.saved_data_to_file()
    
    # 以下为测试和调试代码（已注释）：
    # generate_data.load_data_from_file('data/dataset_5000.csv_norm_stateFalse.pkl')
    # print(generate_data.generated_data[0])
    # 
    # from sklearn.utils import shuffle
    # print(shuffle(generate_data.generated_data)[0])


if __name__ == "__main__":
    """
    程序入口：使用Hydra配置管理运行主函数
    
    使用方法：
    python generate_data.py
    
    可以通过命令行参数覆盖配置：
    python generate_data.py topo.max_num=50 t.end=2.0
    """
    main()
