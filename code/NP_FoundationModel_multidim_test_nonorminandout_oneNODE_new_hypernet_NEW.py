# ==============================================================================
# CNP基础模型测试与微调系统
# Neural Process Foundation Model Testing and Fine-tuning System
# 
# 功能：基于预训练的CNP模型进行few-shot学习和动力学系统预测
# 核心流程：
# 1. 加载预训练的CNP基础模型
# 2. 使用少量观测数据进行微调(fine-tuning)
# 3. 通过ODE积分器进行长期动力学预测
# 4. 执行符号回归发现可解释的动力学方程
# ==============================================================================

from symbolic_regression_tools import perform_symbolic_regression  # 符号回归工具

from NP_FoundationModel_multidim_trainonexist_new_hypernet_multidim import *
import torchdiffeq as ode  # 微分方程求解器

import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

import random
import numpy as np
import csv
import copy
import math
import time

from itertools import product  # 笛卡尔积生成

from data_create.lib.DynamicsEquation_new import DynamicsEquation  # 动力学方程类

# ==============================================================================
# 设备配置
# ==============================================================================
if torch.cuda.is_available():
    device = torch.device("cuda:1")  # 使用第二块GPU
else:
    device = torch.device("cpu")
print("using device : ", device)


# 全局变量：存储发现的动力学方程
F_EQ_STR = None  # 自身动力学方程字符串
G_EQ_STR = None  # 相互作用动力学方程字符串


def set_rand_seed(rseed):
    """
    设置随机种子以确保结果可重现
    
    参数:
        rseed: 随机种子值
    """
    torch.manual_seed(rseed)  
    torch.cuda.manual_seed(rseed)  
    torch.cuda.manual_seed_all(rseed)  
    random.seed(rseed)
    np.random.seed(rseed)


# ==============================================================================
# 微调网络类 (FinetuneNet)
# 
# 功能：基于预训练CNP模型的权重进行微调，实现few-shot学习
# 
# 核心思想：
# 1. 从预训练模型中提取"冻结权重"作为先验知识
# 2. 添加可训练的"增量权重"进行微调
# 3. 最终权重 = 冻结权重 + 增量权重
# 
# 优势：
# - 快速适应：利用预训练知识，只需少量数据即可适应新任务
# - 稳定性强：冻结权重提供稳定的基础，避免灾难性遗忘
# - 可解释性：权重分解清晰展示了预训练知识和新学习的贡献
# ==============================================================================

class FinetuneNet(nn.Module):
    def __init__(
            self,
            state_dim=1,          # 状态空间维度
            hidden_dim=512,       # 隐藏层维度
            max_dim=5,            # 最大支持维度
            pretrained_model=None, # 预训练模型（CNP_self或CNP_interaction）
            type='self',          # 网络类型：'self'或'interaction'
    ):
        super().__init__()

        self.state_dim = state_dim
        
        # 标记是否使用预训练模型
        if pretrained_model is None:
            self.pretrained_model_flag = False  # 从零开始训练
        else:
            self.pretrained_model_flag = True   # 使用预训练权重

        if pretrained_model is None:
            if type == 'self':
                self.decode_encode_x_multidim = nn.Sequential(
                            nn.Linear(int(self.state_dim), hidden_dim),
                            # nn.LayerNorm(hidden_dim),
                            # nn.LeakyReLU(inplace=True),
                            # nn.Linear(hidden_dim, hidden_dim),
                        ).to(device)
            else:
                self.decode_encode_x_multidim = nn.Sequential(
                            nn.Linear(int(self.state_dim*2), hidden_dim),
                            # nn.LayerNorm(hidden_dim),
                            # nn.LeakyReLU(inplace=True),
                            # nn.Linear(hidden_dim, hidden_dim),
                        ).to(device)
        else:
            self.decode_encode_x_multidim = copy.deepcopy(pretrained_model.decode_encode_x_multidim[self.state_dim-1]).to(device)
            for name, param in self.decode_encode_x_multidim.named_parameters():
                param.requires_grad = False

        # used to train
        self.MLP_nonlinear_1_w = nn.Parameter(torch.zeros(hidden_dim, hidden_dim).to(device))
        self.MLP_nonlinear_1_b = nn.Parameter(torch.zeros(1, hidden_dim).to(device))
        self.MLP_nonlinear_2_w = nn.Parameter(torch.zeros(hidden_dim, hidden_dim).to(device))
        self.MLP_linear_w = nn.Parameter(torch.zeros(hidden_dim, hidden_dim).to(device))
        self.MLP_linear_b = nn.Parameter(torch.zeros(1, hidden_dim).to(device))

        if pretrained_model is None:
            # used to train
            # nonlinear_1 = nn.Linear(hidden_dim, hidden_dim).to(device)
            # nonlinear_2 = nn.Linear(hidden_dim, hidden_dim, bias=False).to(device)
            # linear_1 = nn.Linear(hidden_dim, hidden_dim).to(device)

            # self.MLP_nonlinear_1_w = nonlinear_1.weight.t()
            # self.MLP_nonlinear_1_b = nonlinear_1.bias
            # self.MLP_nonlinear_2_w = nonlinear_2.weight.t()
            # self.MLP_linear_w = linear_1.weight.t()
            # self.MLP_linear_b = linear_1.bias
            nn.init.kaiming_uniform_(self.MLP_nonlinear_1_w, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.MLP_nonlinear_2_w, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.MLP_linear_w, a=math.sqrt(5))


            #
            # [d,d]
            self.MLP_nonlinear_1_w_freezed = torch.zeros_like(self.MLP_nonlinear_1_w)
            # [1, d]
            self.MLP_nonlinear_1_b_freezed = torch.zeros_like(self.MLP_nonlinear_1_b)
            # [d,d]
            self.MLP_nonlinear_2_w_freezed = torch.zeros_like(self.MLP_nonlinear_2_w)
            # [d,d]
            self.MLP_linear_w_freezed = torch.zeros_like(self.MLP_linear_w)
            # [1, d]
            self.MLP_linear_b_freezed = torch.zeros_like(self.MLP_linear_b)
        else:

            #
            # [num_sampling, b, d, d] -> [d,d]
            self.MLP_nonlinear_1_w_freezed = copy.deepcopy(torch.mean(torch.mean(pretrained_model.saved_weights[0], dim=0), dim=0).detach().view(hidden_dim, hidden_dim)).to(device)
            # [num_sampling, b, 1, d] -> [1, d]
            self.MLP_nonlinear_1_b_freezed = copy.deepcopy(torch.mean(torch.mean(pretrained_model.saved_weights[1], dim=0), dim=0).detach().view(1, hidden_dim)).to(device)
            # [num_sampling, b, d, d] -> [d,d]
            self.MLP_nonlinear_2_w_freezed = copy.deepcopy(torch.mean(torch.mean(pretrained_model.saved_weights[2], dim=0), dim=0).detach().view(hidden_dim, hidden_dim)).to(device)
            # [num_sampling, b, d, d] -> [d,d]
            self.MLP_linear_w_freezed = copy.deepcopy(torch.mean(torch.mean(pretrained_model.saved_weights[3], dim=0), dim=0).detach().view(hidden_dim, hidden_dim)).to(device)
            # [num_sampling, b, 1, d] -> [1, d]
            self.MLP_linear_b_freezed = copy.deepcopy(torch.mean(torch.mean(pretrained_model.saved_weights[4], dim=0), dim=0).detach().view(1, hidden_dim)).to(device)


        self.MLP_nonlinear_act = nn.ReLU()

        if pretrained_model is None:
            # decode_x_linear_1 = nn.Linear(hidden_dim, self.state_dim).to(device)
            # # [d,state]
            # self.decode_x_mean_w = decode_x_linear_1.weight.t()
            # # [1,state]
            # self.decode_x_mean_b = decode_x_linear_1.bias

            # [d,state]
            self.decode_x_mean_w = nn.Parameter(torch.zeros(hidden_dim, self.state_dim).to(device))
            # [1,state]
            self.decode_x_mean_b = nn.Parameter(torch.zeros(1, self.state_dim).to(device))

            nn.init.kaiming_uniform_(self.decode_x_mean_w, a=math.sqrt(5))

        else:
            # [num_sampling, b, d, state] -> [d,state]
            self.decode_x_mean_w = copy.deepcopy(torch.mean(torch.mean(pretrained_model.saved_weights[5], dim=0), dim=0).detach().view(hidden_dim, self.state_dim)).to(device)
            # [num_sampling, b, 1, state] -> [1,state]
            self.decode_x_mean_b = copy.deepcopy(torch.mean(torch.mean(pretrained_model.saved_weights[6], dim=0),
                                                      dim=0).detach().view(1, self.state_dim)).to(device)

    def trainable_weights(self,):
        if self.pretrained_model_flag:
            weights = [self.MLP_nonlinear_1_w, self.MLP_nonlinear_1_b, self.MLP_nonlinear_2_w,
                                           self.MLP_linear_w, self.MLP_linear_b,]
            #weights = [self.MLP_nonlinear_1_w, self.MLP_nonlinear_1_b,]
        else:
            weights = list(self.decode_encode_x_multidim.parameters()) +\
                      [self.MLP_nonlinear_1_w, self.MLP_nonlinear_1_b, self.MLP_nonlinear_2_w,
                       self.MLP_linear_w, self.MLP_linear_b,
                       self.decode_x_mean_w, self.decode_x_mean_b]
        return weights

       
    def forward(self, x):
        """
        前向传播：组合预训练权重和微调权重进行预测
        
        参数:
            x: 输入状态 [batch_size, state_dim] 或 [batch_size, state_dim*2]
            
        返回:
            out: 预测的状态导数 [batch_size, state_dim]
        """
        # ==============================================================================
        # 阶段1：状态编码
        # ==============================================================================
        x_encoded = self.decode_encode_x_multidim(x)  # 将输入状态编码到隐藏空间

        # ==============================================================================
        # 阶段2：非线性分支 - 捕捉复杂的非线性动力学
        # 权重组合策略：预训练权重 + 微调增量权重
        # ==============================================================================
        h0 = torch.matmul(
            self.MLP_nonlinear_act(  # ReLU激活
                torch.matmul(x_encoded, 
                           self.MLP_nonlinear_1_w_freezed + self.MLP_nonlinear_1_w) +  # 权重相加
                self.MLP_nonlinear_1_b_freezed + self.MLP_nonlinear_1_b   # 偏置相加
            ),
            self.MLP_nonlinear_2_w_freezed + self.MLP_nonlinear_2_w  # 第二层非线性权重
        )
        
        # ==============================================================================
        # 阶段3：线性分支 - 建模线性动力学组件
        # ==============================================================================
        h1 = torch.matmul(x_encoded, 
                         self.MLP_linear_w_freezed + self.MLP_linear_w) + \
             self.MLP_linear_b_freezed + self.MLP_linear_b

        # ==============================================================================
        # 阶段4：特征融合和输出映射
        # ==============================================================================
        hh = h0 + h1  # 非线性和线性分支的残差连接
        
        # 映射到状态空间维度
        out = torch.matmul(hh, self.decode_x_mean_w) + self.decode_x_mean_b

        return out


# ==============================================================================
# ODE函数类 (ODEFunc)
# 
# 功能：定义动力学系统的微分算子，用于ODE积分器
# 
# 核心思想：
# dx_i/dt = f(x_i) + Σ_j g(x_i, x_j)
# 
# 组件：
# - scaling_nn_1: 自身动力学网络，建模f(x_i)  
# - scaling_nn_2: 相互作用网络，建模g(x_i, x_j)
# - adj: 网络邻接矩阵，定义节点连接关系
# 
# 作用：为torchdiffeq提供可微分的动力学函数，实现长期预测
# ==============================================================================

class ODEFunc(nn.Module):
    def __init__(self, state_dim, hidden_dim=128, max_dim=5, pretrained_model=None):
        """
        初始化ODE函数
        
        参数:
            state_dim: 状态空间维度
            hidden_dim: 隐藏层维度
            max_dim: 最大支持维度
            pretrained_model: 预训练的CNPFoundationModel（可选）
        """
        super(ODEFunc, self).__init__()
        
        # ==============================================================================
        # 构建微调网络：基于预训练模型或从零开始
        # ==============================================================================
        if pretrained_model is None:
            # 从零开始训练（用于对比实验）
            self.scaling_nn_1 = FinetuneNet(state_dim=state_dim, hidden_dim=hidden_dim, max_dim=max_dim,
                                            pretrained_model=None, type='self')
            self.scaling_nn_2 = FinetuneNet(state_dim=state_dim, hidden_dim=hidden_dim, max_dim=max_dim,
                                            pretrained_model=None, type='interaction')
        else:
            # 基于预训练CNP模型进行微调（推荐方式）
            self.scaling_nn_1 = FinetuneNet(state_dim=state_dim, hidden_dim=hidden_dim, max_dim=max_dim,
                                            pretrained_model=pretrained_model.CNP_self, type='self')
            self.scaling_nn_2 = FinetuneNet(state_dim=state_dim, hidden_dim=hidden_dim, max_dim=max_dim,
                                            pretrained_model=pretrained_model.CNP_interaction, type='interaction')
        
        # 网络拓扑结构（运行时更新）
        self.adj = None

    def update(self, adj):
        """更新网络拓扑结构"""
        self.adj = adj

    def predict_diff_self(self, x):
        """
        预测自身动力学：dx_i/dt = f(x_i)
        
        参数:
            x: 节点状态 [batch_size, state_dim]
        返回:
            自身动力学贡献
        """
        pre_diff_self = self.scaling_nn_1(x)
        return pre_diff_self

    def predict_diff_interaction(self, x):
        """
        预测相互作用动力学：dx_i/dt = g(x_i, x_j)
        
        参数:
            x: 节点对状态 [batch_size, state_dim*2] - 拼接了x_i和x_j
        返回:
            相互作用动力学贡献
        """
        pre_diff_interaction = self.scaling_nn_2(x)
        return pre_diff_interaction

    def forward(self, t, x):
        """
        微分算子：计算系统在时刻t的状态导数
        
        核心公式：dx_i/dt = f(x_i) + Σ_j g(x_i, x_j)
        
        参数:
            t: 当前时间（未使用，但ODE求解器要求）
            x: 当前状态 [batch_size, #nodes, state_dim]
            
        返回:
            状态导数 [batch_size, #nodes, state_dim]
        """
        # ==============================================================================
        # 阶段1：计算自身动力学 f(x_i)
        # ==============================================================================
        pre_diff_self = self.scaling_nn_1(x)  # [batch_size, #nodes, state_dim]

        # ==============================================================================
        # 阶段2：计算相互作用动力学 g(x_i, x_j)
        # ==============================================================================
        row, col = self.adj  # 边的起点和终点
        # 构造节点对输入：[x_j, x_i] - 注意顺序！
        x_i_j_in = torch.cat([x[:, col.long(), :], x[:, row.long(), :]], dim=-1)
        pre_diff_interaction = self.scaling_nn_2(x_i_j_in)  # [batch_size, #edges, state_dim]

        # ==============================================================================
        # 阶段3：聚合相互作用到目标节点
        # 使用scatter_sum将所有指向节点i的边的贡献求和
        # ==============================================================================
        interaction_sum = scatter_sum(pre_diff_interaction, col.long(), 
                                    dim=1, dim_size=pre_diff_self.size(1))

        # ==============================================================================
        # 阶段4：组合自身动力学和相互作用动力学
        # ==============================================================================
        out_dynamics = pre_diff_self + interaction_sum

        #返回系统总动力学
        return out_dynamics


# ==============================================================================
# CNP基础模型测试类 (TestCNPFoundationModel)
# 
# 功能：基于预训练CNP模型进行few-shot学习和动力学系统发现
# 
# 完整工作流程：
# 1. 【模型加载】：加载预训练的CNP基础模型
# 2. 【上下文学习】：从少量观测数据中学习系统特征  
# 3. 【权重提取】：提取CNP生成的动态权重作为先验知识
# 4. 【微调优化】：使用符号约束进行针对性微调
# 5. 【长期预测】：通过ODE积分器进行长时间序列预测
# 6. 【符号发现】：执行符号回归发现可解释的动力学方程
# 
# 优势：
# - Few-shot能力：仅需少量数据即可适应新的动力学系统
# - 物理一致性：通过ODE约束确保长期预测的稳定性
# - 可解释性：最终输出符号形式的动力学方程
# ==============================================================================

class TestCNPFoundationModel(TrainCNPFoundationModel):
    def __init__(
            self,
            cfg,         # 配置参数
            model_path,  # 预训练模型路径
    ):
        # 继承训练类的基础配置
        super(TestCNPFoundationModel, self).__init__(cfg)

        # ==============================================================================
        # 加载预训练的CNP基础模型
        # ==============================================================================
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(device)

        self.cfg = cfg

        self.params = {'lr': 1e-2,
                       'weight_decay': 1e-5}

        self.num_epoch_fine_tune = 5000
        
        self.state_dim = 1

        self.ode_func = None

        # params for odeint
        self.adjoint = True
        self.rtol = 1e-2  
        self.atol = 1e-3
        #self.rtol = 1e-3  
        #self.atol = 1e-4
        #self.rtol = 1e-9 
        #self.atol = 1e-9

        self.method = 'dopri5'  # 'dopri5'

        # save
        self.adj = None

        # datasets
        dataset_1 = pd.read_csv('data/dataset_5000_test_dim=1.csv', header=None,
                                names=['no', 'f_eq_str_0', 'g_eq_str_0', 'eq_type', 'state'])
        # dataset_2 = pd.read_csv('data/dataset_5000_test_dim=2.csv', header=None,
        #                         names=['no', 'f_eq_str_0', 'f_eq_str_1', 'g_eq_str_0', 'g_eq_str_1', 'eq_type', 'state'])
        # self.dataset = {1: dataset_1, 2: dataset_2}
        self.dataset = {1: dataset_1,}

        # self.dataset_size = {1: len(dataset_1), 2: len(dataset_2)}
        self.dataset_size = {1: len(dataset_1), }

        # self.dataset = self.dataset[:300]
        # self.max_state_dim = 2
        self.max_state_dim = 1

    def make_batch(self, flag_IEEE754=True, state_dim=1):
        batch_size = 1
        lineno = [0] * self.max_state_dim
        while sum(lineno) < sum(list(self.dataset_size.values())):

            start_time = time.time()
            # keep same dim in one batch
            #state_dim = np.random.randint(1, self.max_dim + 1)
            # state_dim = np.random.choice(
            #     a=[key for key in list(self.dataset_size.keys()) if lineno[key - 1] < self.dataset_size[key]],
            #     size=1)[0]
            print(' -- state_dim = %s (lineno = %s, dataset_size = %s)' % (state_dim, lineno, self.dataset_size))
            # selected_dim = np.random.randint(1, state_dim + 1)  # selected dim for target's output

            # keep same topo in one batch
            #N_sampled = np.random.randint(int(self.cfg.topo.max_num / 5), self.cfg.topo.max_num)
            # 网络规模设置
            N_sampled = 400
            #topo_type_sampled = np.random.choice(a=self.cfg.topo.type_list, size=1)[0]
            # 拓扑结构设置
            topo_type_sampled = 'grid'
            #topo_type_sampled = 'power_law'
            topo = Topo(N_sampled, topo_type_sampled)
            # 网络规模设置
            N_sampled = topo.N

            # same sampling x_i and same sampling t
            #num_sampled_x_i = 1
            #num_sampled_x_i = 5
            #num_sampled_x_i = 10
            #num_sampled_x_i = 20
            # 观测节点数量设置
            num_sampled_x_i = np.random.randint(1, int(N_sampled / 10))
            #num_sampled_x_i = 5
            # num_sampled_x_i = 1
            # num_sampled_x_i = np.random.randint(1, 10)
            # 随机选择观测节点
            sampled_x_i_idxs_ = np.random.choice(a=list(range(N_sampled)), size=num_sampled_x_i,
                                                 replace=False)  # [num_sampled_x_i, ]

            # tal_time_steps = int((self.cfg.t.end - self.cfg.t.start) / self.cfg.t.inc)
            # 时间步长设置
            total_time_steps = int((self.cfg.t.end - self.cfg.t.start) / self.cfg.t.inc / 2)
            num_sampled_t_idxs_for_one_x_i = np.random.randint(int(total_time_steps * 0.6), int(total_time_steps * 1))
            # num_sampled_t_idxs_for_one_x_i = total_time_steps
            sampled_t_idxs_for_one_x_i = torch.from_numpy(
                np.random.choice(a=list(range(total_time_steps)), size=num_sampled_t_idxs_for_one_x_i,
                                 replace=False)).long()

            sampled_t_idxs = []
            sampled_x_i_idxs = []
            for ii in range(num_sampled_x_i):
                # num_sampled_t_idxs_for_one_x_i = np.random.randint(10, 50)
                sampled_x_i_idxs.append(torch.Tensor([sampled_x_i_idxs_[ii]] * num_sampled_t_idxs_for_one_x_i))
                sampled_t_idxs.append(sampled_t_idxs_for_one_x_i)

            # [#points, ]
            sampled_t_idxs = torch.cat(sampled_t_idxs, dim=-1).long().view(-1)
            sampled_x_i_idxs = torch.cat(sampled_x_i_idxs, dim=-1).long().view(-1)

            assert sampled_t_idxs.size(0) == sampled_x_i_idxs.size(0)

            print('#points in context = %s' % len(sampled_t_idxs))

            t_context_batch = []
            i_context_batch = []
            x_i_context_batch = []
            x_j_set_context_batch = []
            points_info_context_batch = []

            x_target_self_in_batch = []
            x_target_self_out_batch = []
            x_target_interaction_in_batch = []
            x_target_interaction_out_batch = []
            x_target_total_batch = []
            points_info_batch = []

            num_in_batch = 0
            while num_in_batch < batch_size:

                if lineno[state_dim - 1] >= self.dataset_size[state_dim]:
                    break

                # print('lineno=',lineno)
                f_eq_str = [self.dataset[state_dim].iloc[lineno[state_dim - 1]]['f_eq_str_%s' % i] for i in range(state_dim)]
                g_eq_str = [self.dataset[state_dim].iloc[lineno[state_dim - 1]]['g_eq_str_%s' % i] for i in range(state_dim)]
                eq_type = self.dataset[state_dim].iloc[lineno[state_dim - 1]]['eq_type']
                state = self.dataset[state_dim].iloc[lineno[state_dim - 1]]['state']

                lineno[state_dim - 1] += 1

                if not bool(state):
                    continue
                
                print(eq_type)

                # generate data
                if eq_type == 'Heat' or eq_type == 'Gene' or eq_type == 'Mutualistic':
                    init_cond = InitCondition(N_sampled, state_dim, lbs=torch.ones(state_dim) * (0.),
                                                  ubs=torch.ones(state_dim) * 25., num_sampling=1,
                                                  constraint=None)
                    t_start = self.cfg.t.start
                    t_inc = self.cfg.t.inc
                    t_end = self.cfg.t.end
                elif eq_type == 'Neural':
                    init_cond = InitCondition(N_sampled, state_dim, lbs=torch.ones(state_dim) * (-3.),
                                                  ubs=torch.ones(state_dim) * 3., num_sampling=1,
                                                  constraint=None)
                    t_start = self.cfg.t.start
                    t_inc = self.cfg.t.inc
                    t_end = self.cfg.t.end
                elif eq_type == 'Ecosystems':
                    init_cond = InitCondition(N_sampled, state_dim, lbs=torch.ones(state_dim) * (0.),
                                                  ubs=torch.ones(state_dim) * 10., num_sampling=1,
                                                  constraint=None)
                    t_start = self.cfg.t.start
                    t_inc = self.cfg.t.inc
                    t_end = self.cfg.t.end
                elif eq_type == 'Epidemic' or eq_type == 'Population' or eq_type == 'Lotka-Volterra':
                    init_cond = InitCondition(N_sampled, state_dim, lbs=torch.ones(state_dim) * (0.),
                                                  ubs=torch.ones(state_dim) * 1., num_sampling=1,
                                                  constraint=None)
                    t_start = self.cfg.t.start
                    t_inc = self.cfg.t.inc
                    t_end = self.cfg.t.end
                elif eq_type == 'SI' or eq_type == 'SIS':
                    init_cond = InitCondition(N_sampled, state_dim, lbs=torch.ones(state_dim) * 1e-6,
                                                  ubs=torch.ones(state_dim) * 1e-3, num_sampling=1,
                                                  constraint=(torch.Tensor([0, 1]), "sum_is_one"))
                    t_start = 0.
                    t_inc = 0.5
                    t_end = 50.
                else:
                    print("unknown state_dim [%s]" % state_dim)
                    exit(1)

                s_ns = GeneralDynamics(N_sampled, state_dim, topo_type_sampled, f_eq_str, g_eq_str, topo, init_cond)
                s_data = s_ns.simulating_data(t_start,
                                              t_inc,
                                              t_end,
                                              self.cfg.resample_init_condition,
                                              norm_state=True)
                if s_data is None:
                    print('**Simulation failed!!!**')
                    continue
                
                print(f_eq_str, g_eq_str)

                num_in_batch += 1

                print('**Simulation succeed!!!**[ %s (dataset lineno %s)]**' % (num_in_batch, lineno))

                # s_ns.display(s_data)

                # s_data : {'t': torch.from_numpy(t_range.reshape(-1, 1)),  # [len(t_range),]
                #         'state_data': torch.from_numpy(New_X.reshape(len(t_range), self.N, self.dim)),
                #         # [len(t_range), N, dim]
                #         'total_diff': total_diff_signal,
                #         'self_diff': (self_diff_in, self_diff_out),
                #         'interact_diff': (interact_diff_in, interact_diff_out), # [len(t_range), #edges, dim + dim]
                #         }

                obs_mask = torch.zeros_like(s_data['state_data'])
                obs_mask[0, :, :] = 1. # obs init state

                t_context_one = s_data['t'][sampled_t_idxs].view(-1, 1)  # [#points, 1]
                i_context_one = sampled_x_i_idxs.view(-1, 1)  # [#points, 1]
                x_i_context_one = s_data['state_data'][sampled_t_idxs, sampled_x_i_idxs, :].view(-1, state_dim)  # [#points, 1]

                obs_mask[sampled_t_idxs, sampled_x_i_idxs, :] = 1.  # obs x_i

                x_j_set_context_one = []
                points_info_context_one = []
                # row, col = topo.sparse_adj
                row, col = s_data['sparse_adj']
                # add self loop
                row = torch.cat([row, torch.arange(N_sampled)], dim=0)
                col = torch.cat([col, torch.arange(N_sampled)], dim=0)

                for iiii in range(len(sampled_x_i_idxs)):
                    sampled_x_i_idx = sampled_x_i_idxs[iiii]
                    neibors = s_data['state_data'][sampled_t_idxs[iiii], row, :][col == sampled_x_i_idx].view(
                        -1, state_dim)

                    obs_mask[sampled_t_idxs[iiii], row[col == sampled_x_i_idx], :] = 1.  # obs x_j

                    x_j_set_context_one.append(neibors)
                    points_info_context_one.append(torch.ones_like(neibors)[:, 0].view(-1) * iiii)
                x_j_set_context_one = torch.cat(x_j_set_context_one, dim=0)
                points_info_context_one = torch.cat(points_info_context_one, dim=0)
                
                # normalizing for context
                max_ = torch.cat([x_i_context_one.view(-1), x_j_set_context_one.view(-1)], dim=-1).max()
                min_ = torch.cat([x_i_context_one.view(-1), x_j_set_context_one.view(-1)], dim=-1).min()
                x_i_context_one = (x_i_context_one - min_ + 1.) / (max_ - min_ + 1.)
                x_j_set_context_one = (x_j_set_context_one - min_ + 1.) / (max_ - min_ + 1.)

                t_context_batch.append(t_context_one.unsqueeze(0))  # [1, #points, 1]
                i_context_batch.append(i_context_one.unsqueeze(0))
                x_i_context_batch.append(x_i_context_one.unsqueeze(0))
                x_j_set_context_batch.append(x_j_set_context_one.unsqueeze(0))
                points_info_context_batch.append(points_info_context_one.unsqueeze(0))

                # target
                # sampled_t_idxs_target = []
                # sampled_x_i_idxs_target = []
                # for ii in range(num_sampled_x_i):
                #     total_time_steps = int((self.cfg.t.end - self.cfg.t.start) / self.cfg.t.inc)
                #     num_sampled_t_idxs_for_one_x_i = total_time_steps
                #     sampled_x_i_idxs_target.append(
                #         torch.Tensor([sampled_x_i_idxs_[ii]] * num_sampled_t_idxs_for_one_x_i))
                #     sampled_t_idxs_target.append(torch.from_numpy(
                #         np.random.choice(a=list(range(total_time_steps)), size=num_sampled_t_idxs_for_one_x_i,
                #                          replace=False)).long())
                # # [#points, ]
                # sampled_t_idxs_target = torch.cat(sampled_t_idxs_target, dim=-1).long().view(-1)
                # sampled_x_i_idxs_target = torch.cat(sampled_x_i_idxs_target, dim=-1).long().view(-1)
                #
                # x_target_interaction_in_one = []
                # x_target_interaction_out_one = []
                # points_info_target_one = []
                # for iiii in range(len(sampled_x_i_idxs_target)):
                #     sampled_x_i_idx_target = sampled_x_i_idxs_target[iiii]
                #     neibors_in = s_data['interact_diff'][0][sampled_t_idxs_target[iiii], col == sampled_x_i_idx_target,
                #                  :].view(
                #         -1, state_dim * 2)
                #     x_target_interaction_in_one.append(neibors_in)
                #     neibors_out = s_data['interact_diff'][1][sampled_t_idxs_target[iiii], col == sampled_x_i_idx_target,
                #                   selected_dim - 1].view(
                #         -1, 1)
                #     x_target_interaction_out_one.append(neibors_out)
                #     points_info_target_one.append(torch.ones_like(neibors_in)[:, 0].view(-1) * iiii)
                # x_target_interaction_in_one = torch.cat(x_target_interaction_in_one, dim=0)
                # x_target_interaction_out_one = torch.cat(x_target_interaction_out_one, dim=0)
                # points_info_target_one = torch.cat(points_info_target_one, dim=0)
                #
                # x_target_self_in_batch.append(
                #     s_data['self_diff'][0][sampled_t_idxs_target, sampled_x_i_idxs_target, :].view(-1,
                #                                                                                    state_dim).unsqueeze(
                #         0))
                # x_target_self_out_batch.append(
                #     s_data['self_diff'][1][sampled_t_idxs_target, sampled_x_i_idxs_target, selected_dim - 1].view(-1, 1).unsqueeze(0))
                # x_target_interaction_in_batch.append(x_target_interaction_in_one.unsqueeze(0))
                # x_target_interaction_out_batch.append(x_target_interaction_out_one.unsqueeze(0))
                # x_target_total_batch.append(
                #     s_data['total_diff'][sampled_t_idxs_target, sampled_x_i_idxs_target, selected_dim - 1].view(-1, 1).unsqueeze(
                #         0))
                # points_info_batch.append(points_info_target_one.unsqueeze(0))
                x_target_self_in_batch.append(
                    s_data['self_diff'][0].view(-1, state_dim).unsqueeze(
                        0))
                x_target_self_out_batch.append(
                    s_data['self_diff'][1].view(-1, state_dim).unsqueeze(0))
                x_target_interaction_in_batch.append(
                    s_data['interact_diff'][0].view(-1, state_dim + state_dim).unsqueeze(
                        0))
                x_target_interaction_out_batch.append(
                    s_data['interact_diff'][1].view(-1, state_dim).unsqueeze(
                        0))
                x_target_total_batch.append(
                    s_data['total_diff'].view(-1, state_dim).unsqueeze(
                        0))

            if num_in_batch == 0:  # when no more vaild equations in dataset
                continue

            # # t_context: [b, #points, 1]
            # # i_context: [b, #points, 1]
            # # x_i_context: [b, #points, 1]
            # # x_j_set_context: [b, m, state_dim], where m = 1 + #neighbors of node i
            # # points_info_context: [m, ], where m = 1 + #neighbors of node i
            # t_context, i_context, x_i_context, x_j_set_context, points_info_context = x_context
            t_context_batch = torch.cat(t_context_batch, dim=0)
            i_context_batch = torch.cat(i_context_batch, dim=0)
            x_i_context_batch = torch.cat(x_i_context_batch, dim=0)
            x_j_set_context_batch = torch.cat(x_j_set_context_batch, dim=0)
            points_info_context_batch = torch.cat(points_info_context_batch, dim=0)  # [b, m]

            print("points_info_context_batch.size()=", points_info_context_batch.size())
            # Filter out incorrect data, including no neigbors, wrong topo and Node is or its neighbors in the same batch are different
            if points_info_context_batch.size(1) == 0 or torch.max(
                    points_info_context_batch[0]) + 1 != x_i_context_batch.size(1) or (
                    torch.mean(points_info_context_batch, dim=0) - points_info_context_batch[0]).sum() != 0:
                continue
            
            mu_in_state = torch.mean(x_i_context_batch, dim=1).view(-1, state_dim)
            std_in_state = torch.std(x_i_context_batch, dim=1).view(-1, state_dim)

            if flag_IEEE754:

                b, num_points, _ = t_context_batch.size()

                print('in flag_IEEE754', t_context_batch.size(), x_i_context_batch.size(), x_j_set_context_batch.size())

                t_context_batch = float2bit(t_context_batch).view(b, num_points, -1)
                x_i_context_batch = float2bit(x_i_context_batch).view(b, num_points, -1)
                m = x_j_set_context_batch.size(1)
                x_j_set_context_batch = float2bit(x_j_set_context_batch).view(b, m, -1)

                # norm trick
                t_context_batch = (t_context_batch - 0.5) * 2.
                x_i_context_batch = (x_i_context_batch - 0.5) * 2.
                x_j_set_context_batch = (x_j_set_context_batch - 0.5) * 2.

            # x_target_in: [b, ?, state_dim]
            # x_target_out: [b, ?, 1]
            # x_target_in, x_target_out = x_target
            x_target_self_in_batch = torch.cat(x_target_self_in_batch, dim=0)
            x_target_self_out_batch = torch.cat(x_target_self_out_batch, dim=0)
            x_target_interaction_in_batch = torch.cat(x_target_interaction_in_batch, dim=0)
            x_target_interaction_out_batch = torch.cat(x_target_interaction_out_batch, dim=0)
            x_target_total_batch = torch.cat(x_target_total_batch, dim=0)
            # points_info_batch = torch.cat(points_info_batch, dim=0)
            points_info_batch = torch.zeros(10)

            # print(points_info_batch.size(), x_target_interaction_out_batch.size(), x_target_total_batch.size())

            batch_data = {'x_context': (t_context_batch.float(),
                                        i_context_batch.float(),
                                        x_i_context_batch.float(),
                                        x_j_set_context_batch.float(),
                                        points_info_context_batch[0].long()),
                          'x_target_self': (x_target_self_in_batch.float(), x_target_self_out_batch.float()),
                          'x_target_interaction': (
                              x_target_interaction_in_batch.float(), x_target_interaction_out_batch.float()),
                          'x_target_total': x_target_total_batch.float(),
                          'points_info': points_info_batch[0].long(),
                          'num_in_batch': num_in_batch,
                          'state_dim': state_dim,
                          'mu_x_target_self': torch.mean(x_target_self_out_batch.float().view(x_target_self_out_batch.size(0),-1, state_dim),dim=1).view(-1, 1, state_dim),
                          'std_x_target_self': torch.std(x_target_self_out_batch.float().view(x_target_self_out_batch.size(0),-1, state_dim),dim=1).view(-1, 1, state_dim) + 1.0, # 1e-5
                          'mu_x_target_interaction': torch.mean(x_target_interaction_out_batch.float().view(x_target_interaction_out_batch.size(0),-1, state_dim),dim=1).view(-1, 1, state_dim),
                          'std_x_target_interaction': torch.std(x_target_interaction_out_batch.float().view(x_target_interaction_out_batch.size(0),-1, state_dim),dim=1).view(-1, 1, state_dim) + 1.0, # 1e-5
                          }

            batch_data['adj'] = s_data['sparse_adj'].float()
            batch_data['X0'] = s_data['state_data'][0].float()
            batch_data['mu_in_state'] = mu_in_state.float()
            batch_data['std_in_state'] = std_in_state.float()
            batch_data['obs_mask'] = obs_mask.float()
            batch_data['obs_state'] = s_data['state_data'].float()
            
            
            # handle t
            t_target = torch.cat([s_data['t'][sampled_t_idxs].view(-1).clone().detach(), torch.Tensor([0.])], dim=-1)  # add init state's time
            t_target_remove_duplicates_and_sort_increasing = torch.unique(t_target)
            batch_data['t_target'] = t_target_remove_duplicates_and_sort_increasing.view(-1).float()
            batch_data['t_target_for_test'] = s_data['t'].view(-1).float()
            
            print("make a batch cost = %.2f" % (time.time() - start_time))
            yield batch_data

    # 积分算子：积分求解过程，通过微分方程与系统初始状态来预测系统任意时刻的状态
    def ode_integration(self, vt, x):

        integration_time_vector = vt.type_as(x)
        self.ode_func.update(self.adj)

        if self.adjoint:
            # 使用伴随方法进行内存高效积分
            out = ode.odeint_adjoint(self.ode_func,
                                     x, integration_time_vector,
                                     rtol=self.rtol, atol=self.atol, method=self.method)
        else:
            # 标准ODE积分
            out = ode.odeint(self.ode_func,
                             x, integration_time_vector,
                             rtol=self.rtol, atol=self.atol, method=self.method)
        # the size of out should be confirmed later
        return out  ## [#steps, num_sampling, #nodes, d]

    def model_predict(self, adj, X0=None, t_target=None):
        # 预测系统在指定时间点的状态
        self.adj = adj

        # [#steps, num_sampling, #nodes, d]
        pre_states = self.ode_integration(t_target, X0.unsqueeze(0))

        return pre_states
    # 画图
    def eval_model(self, adj, x_context, points_info, state_dim,
                    mu_std_target, mu_in_state, std_in_state,
                    X0, t_target, obs_state, obs_mask,
                    add_str):
        self.ode_func.eval()
                
        pre_states = self.model_predict(adj, X0, t_target)

        pre_states = pre_states.mean(1)  #[t, num_sampling, nodes, 1]
        # 生成真实值与预测值的对比图
        fig,axs=plt.subplots(2,2,figsize=(10,10))
        for ii in range(obs_state.size(1)):
            axs[0][0].plot(obs_state[:,ii,0].cpu(), 'k:')
            axs[0][0].plot(pre_states[:, ii, 0].detach().cpu(), 'r', alpha=0.5) 
        c=axs[0][1].matshow(obs_mask[:,:,0].cpu())
        plt.colorbar(c)
        a=axs[1][0].matshow(obs_state[:,:,0].cpu())
        plt.colorbar(a)
        b=axs[1][1].matshow(pre_states[:, :, 0].detach().cpu())
        plt.colorbar(b)
        plt.savefig('nonorminandout_oneNODE_epoch_%s.png'%add_str)
        plt.close()

    def eval_model_diff(self, x_context, state_dim, x_target_self, x_target_interaction, mu_std_target, add_str):
        self.model.eval()
        
        #mu_x_target_self, std_x_target_self = mu_std_target[0][0], mu_std_target[1][0]
        #mu_x_target_interaction, std_x_target_interaction = mu_std_target[0][1], mu_std_target[1][1]
        
        num_sampling = 20
        
        res_self = self.model.CNP_self(x_context, x_target_self, state_dim, None, num_sampling=num_sampling)
        res_interaction = self.model.CNP_interaction(x_context, x_target_interaction, state_dim, None, num_sampling=num_sampling, weights_emb=res_self['weights_emb'])
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        # x_target_out_ = (x_target_self[1] - mu_x_target_self) / std_x_target_self  # normalize x_target_out for training
        x_target_out_ = x_target_self[1]
        axs[0].plot([torch.min(x_target_out_.cpu().view(-1)), torch.max(x_target_out_.cpu().view(-1))],
                    [torch.min(x_target_out_.cpu().view(-1)), torch.max(x_target_out_.cpu().view(-1))],
                    'k:')
        axs[0].scatter(x_target_out_.cpu().view(-1),
                       res_self['pre_dist'].loc.detach().cpu().mean(0).view(-1), alpha=0.5)
                       
        axs[0].set_title('F: prediction v.s. groundtruth')
        
        # x_target_out_ = (x_target_interaction[1] - mu_x_target_interaction) / std_x_target_interaction  # normalize x_target_out for training
        x_target_out_ = x_target_interaction[1]
        axs[1].plot([torch.min(x_target_out_.cpu().view(-1)),
                     torch.max(x_target_out_.cpu().view(-1))],
                    [torch.min(x_target_out_.cpu().view(-1)),
                     torch.max(x_target_out_.cpu().view(-1))], 'k:')
        axs[1].scatter(x_target_out_.cpu().view(-1),
                       res_interaction['pre_dist'].loc.detach().cpu().mean(0).view(-1), alpha=0.5)
        
        axs[1].set_title('G: prediction v.s. groundtruth')
        plt.savefig('eval_nonorminandout_oneNODE_test_diff_%s.png' % add_str)
        plt.close()
        
        pre_f = res_self['pre_dist'].loc.detach().cpu().mean(0).view(-1)
        ground_f = x_target_self[1].cpu().view(-1)
        pre_g = res_interaction['pre_dist'].loc.detach().cpu().mean(0).view(-1)
        ground_g = x_target_interaction[1].cpu().view(-1)
        
        Relative_MAE1 = torch.mean(torch.abs(pre_f - ground_f))/torch.mean(torch.abs(ground_f))
        Relative_MAE2 = torch.mean(torch.abs(pre_g - ground_g))/torch.mean(torch.abs(ground_g))
        
        print('diff\'s Relative MAE1: %s, Relative MAE2: %s'%(Relative_MAE1, Relative_MAE2))
        
        #return
        
        # for sampling_i in range(num_sampling):
        #     print("*******************************************************")
        #     print("*******    perform symbolic regression [%s/%s] ********"%(sampling_i+1,num_sampling))
        #     print("*******************************************************")
        #     for _ii in range(state_dim):
        #         eq_str1, _ = perform_symbolic_regression(x_target_self[0].cpu().view(-1, state_dim), res_self['pre_dist'].loc.detach().cpu()[sampling_i].view(-1, state_dim)[:,_ii].view(-1, 1))
        #         eq_str2, _ = perform_symbolic_regression(x_target_interaction[0].cpu().view(-1,state_dim*2), res_interaction['pre_dist'].loc.detach().cpu()[sampling_i].view(-1, state_dim)[:,_ii].view(-1, 1))
        #
        #         eq_list = list(product(eq_str1, eq_str2))
        #
        #         with open('search_2nd_phase/eq_str_%s_dim=%s.csv'%(add_str, _ii),'a+') as f:
        #             f_csv = csv.writer(f)
        #             f_csv.writerows(eq_list)

        if True: # perform symbolic regression on mean of predictive results

            print("*******************************************************")
            print("*******    perform symbolic regression on mean ********")
            print("*******************************************************")
            for _ii in range(state_dim):
                eq_str1, _ = perform_symbolic_regression(x_target_self[0].cpu().view(-1,state_dim), res_self['pre_dist'].loc.detach().cpu().mean(0).view(-1,state_dim)[:,_ii].view(-1, 1))
                eq_str2, _ = perform_symbolic_regression(x_target_interaction[0].cpu().view(-1,state_dim*2), res_interaction['pre_dist'].loc.detach().cpu().mean(0).view(-1,state_dim)[:,_ii].view(-1, 1))
                
                eq_list = list(product(eq_str1, eq_str2))
                
                with open('search_2nd_phase/eq_str_%s_dim=%s.csv'%(add_str, _ii),'w') as f:
                    f_csv = csv.writer(f)
                    f_csv.writerows(eq_list)
            
        
    def save_and_read_batch_data(self, batch_data=None, add_str=""):
        if batch_data is not None: 
            # save data...
            fname = 'data/test_data_on_%s.pickle' % (add_str)
            f = open(fname, 'wb')
            pickle.dump(batch_data, f)
            f.close()
        else:
            # load data
            fname = 'data/test_data_on_%s.pickle' % (add_str)
            with open(fname, 'rb') as f:
                batch_data = pickle.load(f)
            
        return batch_data
    
    def fine_tune(self, batch_data, ode_func=None, add_str=""):
        """
        核心方法：基于预训练CNP模型进行微调和动力学系统发现
        
        完整流程：
        1. 【特征提取】：CNP模型从上下文数据中提取潜在表示
        2. 【符号发现】：通过符号回归发现候选动力学方程  
        3. 【微调优化】：使用发现的方程约束微调神经网络
        4. 【长期预测】：通过ODE积分验证模型性能
        
        参数:
            batch_data: 包含观测数据和系统信息的批次数据
            ode_func: 预训练的ODE函数（可选，用于继续训练）
            add_str: 系统类型标识符（如'Heat', 'Neural'等）
        """
        start_time = time.time() 
        
        # ==============================================================================
        # 数据准备：将批次数据移动到GPU并提取关键信息
        # ==============================================================================
        x_context = move_list_to_device(batch_data['x_context'], device)  # 上下文观测数据
        points_info = batch_data['points_info'].to(device)
        state_dim = batch_data['state_dim']  # 状态空间维度
        
        # 系统信息和测试数据
        adj = batch_data['adj'].to(device)              # 网络邻接矩阵
        X0 = batch_data['X0'].to(device)                # 系统初始状态
        mu_in_state = batch_data['mu_in_state'].to(device)
        std_in_state = batch_data['std_in_state'].to(device)
        t_target = batch_data['t_target'].to(device)    # 目标时间点
        obs_mask = batch_data['obs_mask'].to(device)    # 观测掩码
        obs_state = batch_data['obs_state'].to(device)  # 观测到的真实状态

        # 目标动力学数据
        x_target_self = move_list_to_device(batch_data['x_target_self'], device)        # 自身动力学
        x_target_interaction = move_list_to_device(batch_data['x_target_interaction'], device)  # 相互作用动力学
        
        t_target_for_test = batch_data['t_target_for_test'].to(device)  # 完整时间序列
        
        indices_t_target = []
        for tt in t_target:
            for t_idx in range(t_target_for_test.size(0)):
                if tt == t_target_for_test[t_idx]:
                    indices_t_target.append(t_idx)
                    break
        indices_t_target = torch.Tensor(indices_t_target).to(device)
        
        #print(t_target_for_test.size(), t_target.size(), indices_t_target.size())
        
        self.model.eval()
        
        #mu_x_target_self = batch_data['mu_x_target_self'].to(device)
        #std_x_target_self = batch_data['std_x_target_self'].to(device)
        #mu_x_target_interaction = batch_data['mu_x_target_interaction'].to(device)
        #std_x_target_interaction = batch_data['std_x_target_interaction'].to(device)
                
        #mu_std_target = ((mu_x_target_self, mu_x_target_interaction), (std_x_target_self, std_x_target_interaction))
        
        if ode_func is None:
            # ==============================================================================
            # 阶段1：基于CNP模型进行符号回归发现动力学方程
            # ==============================================================================
            case_name_ = add_str     # 系统类型名称
            dim_ = 1                 # 当前处理1维系统
            opt_flag_ = 0            # 优化标志
            add_str_ = ''            # 附加字符串
            
            # 步骤1.1：评估CNP模型的预测性能，生成训练数据用于符号回归
            self.eval_model_diff(x_context, state_dim, x_target_self, x_target_interaction, None, add_str)
            
            # 步骤1.2：调用符号回归模块发现候选动力学方程
            import subprocess
            
            cmd="python screen_pretrain_knowledge_eq.py --case_name='%s' --dim=%s --opt_flag=%s --add_str=%s"%(case_name_, dim_, opt_flag_, add_str_)
            p=subprocess.Popen(cmd,shell=True)
            return_code=p.wait()  # 等待符号回归完成

            print('符号回归完成，返回代码: %s'%return_code)
            
            time.sleep(1)
            return  # 符号回归阶段完成，退出
            
            knowledge_fname = 'pretrain_knowledge_eq_%s_dim%s_optflag%s_addstr%s.csv'%(case_name_, dim_, opt_flag_, add_str_)
            knowledge_dataset_1 = pd.read_csv(knowledge_fname, header=0,
                                names=['case_name', 'f_eq', 'g_eq', 'fitness', 'complex'])
            #print(knowledge_dataset_1)
            
            #exit(1)
            
            _f_eq_str__ = knowledge_dataset_1.iloc[0]['f_eq'].replace('x0', 'x_1_0')
            _g_eq_str__ = knowledge_dataset_1.iloc[0]['g_eq'].replace('x0', 'x_1_0').replace('x1', 'x_2_0')
            
            global F_EQ_STR
            global G_EQ_STR
            F_EQ_STR = [_f_eq_str__]
            G_EQ_STR = [_g_eq_str__]
            
            print('\n\npretrained knowledge: ', F_EQ_STR, G_EQ_STR)
        
        #
        #
        # self.eval_model(adj, x_context, points_info, state_dim,
        #             None, mu_in_state, std_in_state,
        #             X0, t_target_for_test, obs_state, obs_mask, add_str)
        #
        # return

        if ode_func is None:
            # 基于预训练CNP模型创建ODE函数
            #self.ode_func = ODEFunc(state_dim=self.state_dim, hidden_dim=128, max_dim=5, pretrained_model=None)
            self.ode_func = ODEFunc(state_dim=self.state_dim, hidden_dim=128, max_dim=5, pretrained_model=copy.deepcopy(self.model))
        else:
            self.ode_func = ode_func
            
        #del self.model
        #del x_context
        #del x_target_self
        #del x_target_interaction
        #del mu_std_target
        #if torch.cuda.is_available():
        #    torch.cuda.empty_cache()
            
            
        ## ----function----
        #f_eq_str = ['-2.52*x_1_0']
        #g_eq_str = ['x_2_0*(1 - x_1_0)']
        f_eq_str = F_EQ_STR
        g_eq_str = G_EQ_STR
        
        #f_eq_str = ['0.0053933477 - 3.2227287*x_1_0']
        #g_eq_str = ['-0.984457968271707*x_1_0*x_2_0 + x_2_0*exp(-0.0034366425*exp(x_1_0))']

        
        self.constrainted_dynamics_eq = DynamicsEquation(self.state_dim, f_eq_str, g_eq_str)
        
        #low_bound = 0  ##############
        #up_bound = 1  ###############
        
        
        if add_str == 'Heat' or add_str == 'Gene' or add_str == 'Mutualistic':
            low_bound = 0 
            up_bound = 25  
        elif add_str == 'Neural':        
            low_bound = -3 
            up_bound = 3 
        elif add_str == 'Ecosystems':
            low_bound = 0 
            up_bound = 10 
        elif add_str == 'Epidemic' or add_str == 'Population' or add_str == 'Lotka-Volterra':
            low_bound = 0 
            up_bound = 1
        else:
            print("unknown add_str [%s]" % add_str)
            exit(1)
        
        # 自身动力学数据
        input_state_1 = torch.randn(200 * self.state_dim, self.state_dim)
        input_state_1.uniform_(low_bound, up_bound)
        self_diff = self.constrainted_dynamics_eq.generate_diff_data_1(input_state_1)
        self_diff_in = self_diff[0].view(-1, self.state_dim).to(device)
        self_diff_out = self_diff[1].view(-1, self.state_dim).to(device)

        # 交互动力学数据
        input_state_2 = torch.randn(200 * self.state_dim, self.state_dim + self.state_dim)
        input_state_2.uniform_(low_bound, up_bound)
        interact_diff = self.constrainted_dynamics_eq.generate_diff_data_2(input_state_2)
        interact_diff_in = interact_diff[0].view(-1, self.state_dim + self.state_dim).to(device)
        interact_diff_out = interact_diff[1].view(-1, self.state_dim).to(device)

        ##----

        #print(self.ode_func.scaling_nn_1.trainable_weights())
        #print(self.ode_func.scaling_nn_2.trainable_weights())
        # 微调优化器和学习率调度器
        if ode_func is None:

            optim = torch.optim.Adam(self.ode_func.scaling_nn_1.trainable_weights() + self.ode_func.scaling_nn_2.trainable_weights(),
                                     1e-2,
                                     weight_decay=self.params['weight_decay'])
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=int(self.num_epoch_fine_tune/5), gamma=0.35)
        else:                     
            optim = torch.optim.Adam(self.ode_func.scaling_nn_1.trainable_weights() + self.ode_func.scaling_nn_2.trainable_weights(), 1e-3)
                                 
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=int(self.num_epoch_fine_tune/2), gamma=0.8)
        
        
        
        # optim = adamod.AdaMod(self.ode_func.scaling_nn_1.trainable_weights() + self.ode_func.scaling_nn_2.trainable_weights(),
        #                          self.params['lr'], weight_decay=self.params['weight_decay'])
        
        for epoch in tqdm(range(self.num_epoch_fine_tune)):
            
            self.ode_func.train()
            
            optim.zero_grad()
            
            aa=time.time()

            if ode_func is None:
                # 纯方程约束训练
                pre_diff_self = self.ode_func.predict_diff_self(self_diff_in)
                pre_diff_interaction = self.ode_func.predict_diff_interaction(interact_diff_in)

                #loss = torch.sqrt(torch.mean((pre_diff_self - self_diff_out)**2)/torch.mean(self_diff_out**2)) + \
                #       torch.sqrt(torch.mean((pre_diff_interaction - interact_diff_out)**2)/torch.mean(interact_diff_out**2))
                
                loss = torch.mean(torch.abs(pre_diff_self - self_diff_out))/torch.mean(torch.abs(self_diff_out)) + \
                       torch.mean(torch.abs(pre_diff_interaction - interact_diff_out))/torch.mean(torch.abs(interact_diff_out))


            else:
                # 方程约束+观测数据约束训练
                pre_diff_self = self.ode_func.predict_diff_self(self_diff_in)
                pre_diff_interaction = self.ode_func.predict_diff_interaction(interact_diff_in)

                #loss = torch.sqrt(torch.mean((pre_diff_self - self_diff_out)**2)/torch.mean(self_diff_out**2)) + \
                #       torch.sqrt(torch.mean((pre_diff_interaction - interact_diff_out)**2)/torch.mean(interact_diff_out**2))
                
                loss1 = torch.mean(torch.abs(pre_diff_self - self_diff_out))/torch.mean(torch.abs(self_diff_out)) + \
                       torch.mean(torch.abs(pre_diff_interaction - interact_diff_out))/torch.mean(torch.abs(interact_diff_out))
                ##
                                 
                pre_states = self.model_predict(adj, X0=X0, t_target=t_target)

                print('pre_states.size=', pre_states.size())

                pre_states = pre_states.mean(1)  #[t, num_sampling, nodes, 1]

                # print(pre_states.size())

                # loss = torch.mean(torch.abs(pre_states[obs_mask.long()[indices_t_target.long()]==1] - obs_state[indices_t_target.long()][obs_mask.long()[indices_t_target.long()]==1]))  # l1 loss
                # loss = torch.sqrt(torch.mean((pre_states[obs_mask.long()[indices_t_target.long()] == 1]- obs_state[indices_t_target.long()][obs_mask.long()[indices_t_target.long()] == 1])**2)/torch.mean((obs_state[indices_t_target.long()][obs_mask.long()[indices_t_target.long()] == 1])**2))
                #loss = torch.sqrt(torch.mean((pre_states[obs_mask.long()[indices_t_target.long()] == 1]- obs_state[indices_t_target.long()][obs_mask.long()[indices_t_target.long()] == 1])**2)/torch.mean((obs_state[indices_t_target.long()][obs_mask.long()[indices_t_target.long()] == 1])**2))
                loss2 = torch.mean(torch.abs(pre_states[obs_mask.long()[indices_t_target.long()] == 1]- obs_state[indices_t_target.long()][obs_mask.long()[indices_t_target.long()] == 1]))/torch.mean(torch.abs(obs_state[indices_t_target.long()][obs_mask.long()[indices_t_target.long()] == 1]))
                
                loss = loss2 # + loss1


                # print('forward cost=',time.time()-aa)

                print('obs_mask.sum()=',obs_mask.sum())

            # aa=time.time()

            loss.backward()
            
            if ode_func is not None:
                torch.nn.utils.clip_grad_norm_(self.ode_func.scaling_nn_1.trainable_weights() + self.ode_func.scaling_nn_2.trainable_weights(), 1.)
            optim.step()  ## update paramters
            
            lr_scheduler.step()
            
            # print('backward cost=',time.time()-aa)

            print('epoch %s, loss = %s, '
                      'cost_time = %.2f' \
                      % (epoch + 1,
                         loss.item(),
                         time.time() - start_time), end='\r\n')
                         
                         
            if (epoch + 1) == self.num_epoch_fine_tune:
                torch.save(self.ode_func.state_dict(),
                               "saved_model_finetuned_%s.pkl"%add_str)

            # if (epoch + 1) % 100 == 0 or (epoch + 1) == self.num_epoch_fine_tune:
            #     self.eval_model(adj, x_context, points_info, state_dim,
            #         None, mu_in_state, std_in_state,
            #         X0, t_target_for_test, obs_state, obs_mask, str(epoch+1))
        
        if ode_func is not None:
            pre_diff_self = self.ode_func.predict_diff_self(x_target_self[0]).detach()
            pre_diff_interaction = self.ode_func.predict_diff_interaction(x_target_interaction[0]).detach()
            
            
            # eval diff
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
            # x_target_out_ = (x_target_self[1] - mu_x_target_self) / std_x_target_self  # normalize x_target_out for training
            x_target_out_ = x_target_self[1]
            axs[0].plot([torch.min(x_target_out_.cpu().view(-1)), torch.max(x_target_out_.cpu().view(-1))],
                        [torch.min(x_target_out_.cpu().view(-1)), torch.max(x_target_out_.cpu().view(-1))],
                        'k:')
            axs[0].scatter(x_target_out_.cpu().view(-1),
                           pre_diff_self.cpu().view(-1), alpha=0.5)
                           
            axs[0].set_title('F: prediction v.s. groundtruth')
            
            # x_target_out_ = (x_target_interaction[1] - mu_x_target_interaction) / std_x_target_interaction  # normalize x_target_out for training
            x_target_out_ = x_target_interaction[1]
            axs[1].plot([torch.min(x_target_out_.cpu().view(-1)),
                         torch.max(x_target_out_.cpu().view(-1))],
                        [torch.min(x_target_out_.cpu().view(-1)),
                         torch.max(x_target_out_.cpu().view(-1))], 'k:')
            axs[1].scatter(x_target_out_.cpu().view(-1),
                           pre_diff_interaction.cpu().view(-1), alpha=0.5)
            
            axs[1].set_title('G: prediction v.s. groundtruth')
            plt.savefig('eval_nonorminandout_oneNODE_test_diff_%s_afterfinetuning.png' % add_str)
            plt.close()
            
            pre_f = pre_diff_self.cpu().view(-1)
            ground_f = x_target_self[1].cpu().view(-1)
            pre_g = pre_diff_interaction.cpu().view(-1)
            ground_g = x_target_interaction[1].cpu().view(-1)
            
            Relative_MAE1 = torch.mean(torch.abs(pre_f - ground_f))/torch.mean(torch.abs(ground_f))
            Relative_MAE2 = torch.mean(torch.abs(pre_g - ground_g))/torch.mean(torch.abs(ground_g))
            
            print('After finetuning || diff\'s Relative MAE1: %s, Relative MAE2: %s'%(Relative_MAE1, Relative_MAE2))
            
            
            print("****************************************************************")
            print("*******    perform symbolic regression after finetuning ********")
            print("****************************************************************")
            
                    
            for _ii in range(state_dim):
                eq_str1, _ = perform_symbolic_regression(x_target_self[0].cpu().view(-1,state_dim), pre_diff_self.cpu().view(-1,state_dim)[:,_ii].view(-1, 1))
                eq_str2, _ = perform_symbolic_regression(x_target_interaction[0].cpu().view(-1,state_dim*2), pre_diff_interaction.cpu().view(-1,state_dim)[:,_ii].view(-1, 1))
                    
                eq_list = list(product(eq_str1, eq_str2))
                    
                with open('search_2nd_phase/eq_str_%s_dim=%s_after_finetuning.csv'%(add_str, _ii),'w') as f:
                    f_csv = csv.writer(f)
                    f_csv.writerows(eq_list)       

        #exit(1)


# Simulation_config,MutualisticInteraction_config, GeneRegulatory_config
@hydra.main(config_name="Simulation_config", version_base='1.2', config_path='configs')
def main(cfg):
    #model_path = 'saved_model_train_on_exist_nonorminandout_oneNODE_new_hypernet.pkl'
    #model_path = 'saved_model_train_on_exist_oneNODE_new_hypernet.pkl'
    # model_path = 'saved_model_train_on_exist_oneNODE_new_hypernet_NP.pkl'
    #model_path = 'saved_model_train_on_exist_nonorminandout.pkl'
    #model_path = 'saved_model_train_on_exist_oneNODE_new_hypernet_NP_multidim.pkl'
    model_path = 'saved_model_train_on_exist_oneNODE_new_hypernet_NP_multidim_7sys+2sys.pkl'
    testmodel = TestCNPFoundationModel(cfg, model_path)
    
    set_rand_seed(1)
    
    gen_batch = testmodel.make_batch(flag_IEEE754=True, state_dim=1)
    
    def test_(batch_data, add_str):
        batch_data = testmodel.save_and_read_batch_data(batch_data, add_str)
        testmodel.num_epoch_fine_tune = 10000
        testmodel.fine_tune(batch_data, ode_func=None, add_str=add_str)
        #testmodel.num_epoch_fine_tune = 500
        #ode_func = ODEFunc(state_dim=testmodel.state_dim, hidden_dim=128, max_dim=5, pretrained_model=copy.deepcopy(testmodel.model))
        #ode_func.load_state_dict(torch.load("saved_model_finetuned_%s.pkl"%add_str, map_location=device))
        #testmodel.fine_tune(batch_data, ode_func=ode_func, add_str=add_str)
        
    

            
    batch_data = next(gen_batch)
    add_str = 'Epidemic'
    test_(batch_data, add_str)
    batch_data = next(gen_batch)
    add_str = 'Neural'
    test_(batch_data, add_str)
    
    batch_data = next(gen_batch)
    add_str = 'Population'
    test_(batch_data, add_str)
    batch_data = next(gen_batch)
    add_str = 'Lotka-Volterra'
    test_(batch_data, add_str)
    batch_data = next(gen_batch)
    add_str = 'Heat'
    test_(batch_data, add_str)
    batch_data = next(gen_batch)
    add_str = 'Mutualistic'
    test_(batch_data, add_str)
    batch_data = next(gen_batch)
    add_str = 'Gene'
    test_(batch_data, add_str)
    
    
    
    
    """
    testmodel.fine_tune(gen_batch, 'Epidemic')
    testmodel.fine_tune(gen_batch, 'Neural')
    testmodel.fine_tune(gen_batch, 'Population')
    testmodel.fine_tune(gen_batch, 'Lotka-Volterra')

    testmodel.fine_tune(gen_batch, 'heat')
    testmodel.fine_tune(gen_batch, 'mutu')
    testmodel.fine_tune(gen_batch, 'gene')
    
    
    gen_batch = testmodel.make_batch(flag_IEEE754=True, state_dim=1)
    
    testmodel.fine_tune(gen_batch, 'heat')
    testmodel.fine_tune(gen_batch, 'mutu')
    testmodel.fine_tune(gen_batch, 'gene')
    

    gen_batch = testmodel.make_batch(flag_IEEE754=True, state_dim=2)

    testmodel.fine_tune(gen_batch, 'SI')
    testmodel.fine_tune(gen_batch, 'SIS')
    """


if __name__ == "__main__":

    main()
