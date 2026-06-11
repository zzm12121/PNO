"""
基于遗传编程的符号回归搜索系统
主要功能：从预训练的方程知识库中筛选最佳的网络动力学方程
适用于多维度动力学系统的方程发现和优化
"""

import copy           # 深拷贝操作
import multiprocessing # 多进程处理
import threading      # 线程控制

lock = threading.Lock()  # 线程锁，用于保护scipy优化器的线程安全
import warnings

warnings.filterwarnings("ignore")  # 忽略警告信息

import math                      # 数学函数
import random                    # 随机数生成
import time                      # 时间处理
from functools import partial    # 函数工具
import numpy as np               # 数值计算
import sympy as sp              # 符号数学计算
import scipy.optimize as opt    # 科学计算优化算法

import matplotlib.pyplot as plt  # 绘图库
from tqdm import tqdm           # 进度条显示
import func_timeout             # 函数超时控制
from func_timeout import func_set_timeout  # 设置函数超时装饰器
import pandas as pd             # 数据处理
import csv                      # CSV文件处理
import pickle                   # 数据序列化
import search_2nd_phase.gp_tools as gp  # 遗传编程工具模块


# ============== 命令行参数配置 ==============
import argparse

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Search_for_2ndPhase')
parser.add_argument('--case_name', help='案例名称，支持："heat"(热扩散)、"mutu"(互利作用)、"gene"(基因调控), 默认:"heat"', default="heat")
parser.add_argument('--dim', help='系统维度，默认:1', default=1)
parser.add_argument('--opt_flag', help='是否启用常数优化，0:不优化，1:优化，默认:0', default=0)
parser.add_argument('--add_str', help='文件名附加字符串，默认:""', default="")

args = parser.parse_args()

# 支持的案例类型:
# 'heat' - 热扩散方程
# 'mutu' - 互利相互作用方程  
# 'gene' - 基因调控网络方程
case_name = args.case_name        # 案例名称
dim = int(args.dim)              # 系统维度
opt_flag = int(args.opt_flag)    # 常数优化标志
add_str = args.add_str           # 附加字符串

# ============== 数据加载 ==============
# 构建测试数据集路径
test_data_set_path = 'data/test_data_on_%s%s.pickle' % (case_name, add_str)
with open(test_data_set_path, 'rb') as f:
    batch_data = pickle.load(f)  # 加载预处理的批量数据

# 生成不同维度的案例名称列表
case_name_list = ["%s_dim=%s"%(case_name, i) for i in range(dim)]

# 提取关键数据结构
sparse_A = np.array(batch_data['adj'].numpy(), dtype=int)    # 稀疏邻接矩阵（网络拓扑结构）
X0 = batch_data['X0'].numpy()                              # 初始状态条件

t_range = batch_data['t_target'].numpy()                   # 目标时间序列
Mask = batch_data['obs_mask'].numpy()                      # 观测掩码（标记哪些数据点被观测）
Y = batch_data['obs_state'].numpy()                        # 观测到的真实状态值

# 获取系统维度信息
Dim = X0.shape[1]    # 系统状态维度（每个节点的状态变量数）
N = X0.shape[0]      # 网络节点数量

# print('*Load data OK!')  # 数据加载完成提示

# ============== 算法参数设置 ==============
# 遗传编程树的深度范围控制
min_ = 1    # 最小树深度：控制表达式的最小复杂度
max_ = 5    # 最大树深度：防止表达式过于复杂

# 误差容忍阈值设置
tol_err = 5e-3   # 可接受的相对误差阈值（0.5%）
                 # 适用于mutu(互利作用)、heat(热扩散)、gene(基因调控)等案例

# 优化超时控制
timeout_second = 10.0*60    # 单次常数优化的超时时间：10分钟
                            # 防止某些复杂表达式的优化过程无限期运行

# 种群选择数量控制
# 该值与待优化方程的最大复杂度相关，值越大，允许的方程复杂度越高
num_choose = multiprocessing.cpu_count()    # 基于CPU核心数设置并行处理的方程数量

# ============== 常数采样分布设置 ==============
min_const = -3.0    # 随机常数的最小值
max_const = 3.0     # 随机常数的最大值
# 创建常数采样函数：在[-3.0, 3.0]区间内均匀采样随机常数
sampling_const = partial(random.uniform, min_const, max_const)


# ============== 原语集合构建 ==============
# 构建遗传编程的原语集合，定义可用于构建数学表达式的基本操作符
def build_psets(dim=1):
    """
    构建用于遗传编程的原语集合
    
    参数:
        dim: 系统维度，决定输入变量的数量
    
    返回:
        pset_f: f函数的原语集合（用于节点内部动力学）
        pset_g: g函数的原语集合（用于节点间相互作用）
        pset: 包含两个集合的元组
    """
    
    # ========== f函数原语集合 ==========
    # f函数描述节点的内部动力学行为
    pset_f = gp.PrimitiveSet("F", dim, prefix='x')
    # 添加二元运算符（需要两个操作数）
    pset_f.addPrimitive(lambda x, y: np.nan_to_num(np.add(x, y), nan=1e30),
                        2, name='Add')
    pset_f.addPrimitive(lambda x, y: np.nan_to_num(np.subtract(x, y), nan=1e30),
                        2, name='Sub')
    pset_f.addPrimitive(lambda x, y: np.nan_to_num(np.multiply(x, y), nan=1e30),
                        2, name='Mul')
    pset_f.addPrimitive(lambda x, y: np.nan_to_num(np.divide(x, y), nan=1e30),
                        2, name='Div')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.exp(x), nan=1e30),
                        1, name='exp')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.sin(x), nan=1e30),
                        1, name='sin')
    pset_f.addPrimitive(lambda x: np.nan_to_num(np.cos(x), nan=1e30),
                        1, name='cos')
    pset_f.addPrimitive(lambda x, y: np.nan_to_num(np.power(np.abs(x), np.abs(y))),
                       2, name='Pow')
    pset_f.addEphemeralConstant('C',
                                sampling_const)

    # ---------- g -----------------
    # g函数描述节点间的相互作用机制，输入维度为dim*2（包含两个节点的状态）
    pset_g = gp.PrimitiveSet("G", int(dim * 2), prefix='x')

    pset_g.addPrimitive(lambda x, y: np.nan_to_num(np.add(x, y), nan=1e30),
                        2, name='Add')
    pset_g.addPrimitive(lambda x, y: np.nan_to_num(np.subtract(x, y), nan=1e30),
                        2, name='Sub')
    pset_g.addPrimitive(lambda x, y: np.nan_to_num(np.multiply(x, y), nan=1e30),
                        2, name='Mul')
    pset_g.addPrimitive(lambda x, y: np.nan_to_num(np.divide(x, y), nan=1e30),
                        2, name='Div')
    pset_g.addPrimitive(lambda x: np.nan_to_num(np.exp(x), nan=1e30),
                        1, name='exp')
    pset_g.addPrimitive(lambda x: np.nan_to_num(np.sin(x), nan=1e30),
                        1, name='sin')
    pset_g.addPrimitive(lambda x: np.nan_to_num(np.cos(x), nan=1e30),
                        1, name='cos')
    pset_g.addPrimitive(lambda x, y: np.nan_to_num(np.power(np.abs(x), np.abs(y))),
                       2, name='Pow')
    pset_g.addEphemeralConstant('C',
                                sampling_const)

    # ========== 原语集合组合 ==========
    # pset是一个元组，包含两个原语集合：
    # - pset_f: 用于构建f函数（节点内部动力学）的原语集合
    # - pset_g: 用于构建g函数（节点间相互作用）的原语集合
    pset = (pset_f, pset_g)

    return pset_f, pset_g, pset


# 初始化原语集合
pset_f, pset_g, pset = build_psets(Dim)

# ============== 表达式转换器 ==============
# 用于将遗传编程的函数名转换为sympy符号表达式的映射字典
converter = {
    'Sub': lambda x, y: x - y,          # 减法：x - y
    'Div': lambda x, y: x / y,          # 除法：x / y
    'Mul': lambda x, y: x * y,          # 乘法：x * y
    'Add': lambda x, y: x + y,          # 加法：x + y
    'Neg': lambda x: -x,                # 负号：-x
    'Pow': lambda x, y: x ** y,         # 幂运算：x^y
    'Inv': lambda x: x ** (-1),         # 倒数：1/x
    'Sqrt': lambda x: sp.sqrt(x),       # 平方根：√x
    'exp': lambda x: sp.exp(x),         # 指数函数：e^x
    'sin': lambda x: sp.sin(x),         # 正弦函数：sin(x)
    'cos': lambda x: sp.cos(x),         # 余弦函数：cos(x)
}


# ============== 辅助统计函数 ==============
def stat_complex(pop):
    """
    统计种群中搜索到的方程的复杂度分布
    
    参数:
        pop: 种群列表，每个个体包含[(f_func, g_func), fitness]
    
    返回:
        Stat_Complex: 字典，键为复杂度（表达式节点总数），值为该复杂度的方程数量
    """
    Stat_Complex = {}
    for p_i in pop:
        # 计算方程复杂度：f函数和g函数的节点数之和
        complex_of_eq = len(p_i[0][0]) + len(p_i[0][1])
        
        # 统计不同复杂度的方程数量
        if complex_of_eq not in Stat_Complex.keys():
            Stat_Complex[complex_of_eq] = 1
        else:
            Stat_Complex[complex_of_eq] += 1
    return Stat_Complex


# ============== 核心函数 ==============
def eval_func(ind_i_1_2, pset, X0, sparse_A, Y, Mask, Stat_Complex=None):
    """
    评估个体（方程对）的适应度函数
    
    参数:
        ind_i_1_2: 个体，包含(f_func, g_func)的方程对
        pset: 原语集合元组(pset_f, pset_g)
        X0: 初始状态条件
        sparse_A: 稀疏邻接矩阵（网络拓扑）
        Y: 真实观测状态值
        Mask: 观测掩码
        Stat_Complex: 复杂度统计字典（可选，用于复杂度惩罚）
    
    返回:
        loss: 个体的适应度值（损失值，越小越好）
    """
    # 解包原语集合
    pset_f_, pset_g_ = pset

    # 编译遗传编程树为可执行函数
    eval_func_f = gp.compile(ind_i_1_2[0], pset_f_)  # 编译f函数（节点内部动力学）
    eval_func_g = gp.compile(ind_i_1_2[1], pset_g_)  # 编译g函数（节点间相互作用）

    # 求解常微分方程初值问题(一个积分算子的过程，通过微分方程与初识状态，得到目标时间序列)
    with gp.stdout_redirected():  # 屏蔽求解过程中的警告信息
        # lock.acquire()  # 可选的线程锁（当前注释掉）
        # 使用f和g函数求解网络动力学系统
        soluation_Y, t_range = gp.solve_ivp(
            eval_func_f, eval_func_g,    # f和g函数
            X0, sparse_A,                # 初始条件和网络拓扑
            t_start=0, t_end=1, t_inc=0.01  # 时间参数：从0到1，步长0.01
        )
        # lock.release()

    # 计算预测误差
    # 备选的损失函数计算方式：
    # loss = np.mean(np.abs(soluation_Y[Mask == 1] - Y[Mask == 1]))           # 平均绝对误差
    # loss = np.sqrt(np.mean((soluation_Y[Mask == 1] - Y[Mask == 1])**2)/np.mean(Y[Mask == 1]**2))  # 归一化RMSE
    
    # 当前使用的损失函数：相对平均绝对误差
    loss = np.mean(np.abs(soluation_Y[Mask == 1] - Y[Mask == 1])) / np.mean(np.abs(Y[Mask == 1]))

    # 复杂度惩罚机制（可选）
    if Stat_Complex is not None:
        # 计算当前个体的复杂度
        complex_of_eq = len(ind_i_1_2[0]) + len(ind_i_1_2[1])
        # 根据复杂度在种群中的稀缺程度调整损失值
        # 越稀少的复杂度惩罚越小，鼓励多样性
        complexity_penalty = np.exp(float(Stat_Complex[complex_of_eq] / np.sum(list(Stat_Complex.values()))))
        loss = loss * complexity_penalty
    
    # 处理数值异常，将NaN替换为极大值
    return np.nan_to_num(loss, nan=1e30)


# ============== 常数优化函数 ==============

@func_set_timeout(timeout_second)  # 设置超时装饰器，防止优化过程无限期运行
def optimize_in_scipy(on_core_i, fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol, callback, options):
    """
    使用scipy进行数值优化的核心函数
    
    参数:
        on_core_i: 当前运行的CPU核心编号（用于调试）
        fun: 要优化的目标函数
        x0: 初始参数猜测值
        args: 传递给目标函数的额外参数
        method: 优化算法（如'BFGS', 'L-BFGS-B'等）
        jac, hess, hessp: 梯度、海塞矩阵相关参数
        bounds: 参数边界约束
        constraints: 其他约束条件
        tol: 收敛容忍度
        callback: 回调函数
        options: 优化器选项
    
    返回:
        res: scipy优化结果对象
    """
    # if on_core_i == 1:
    #     time.sleep(5)  # 测试用的延时（已注释）
    
    # 调用scipy的minimize函数进行数值优化
    res = opt.minimize(fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol, callback, options)
    return res


def optimize_in_scipy_with_timeout_check(on_core_i, fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol,
                                         callback, options):
    """
    带超时检查和线程安全的scipy优化包装函数
    
    该函数解决了两个关键问题：
    1. 线程安全：scipy的optimize函数在多线程环境下不安全，需要加锁
    2. 超时处理：捕获超时异常，避免程序崩溃
    
    参数: 与optimize_in_scipy相同
    
    返回:
        res: 优化结果对象，超时时返回None
    """
    # 获取线程锁，确保scipy优化器的线程安全
    # scipy.optimize.minimize在多线程环境下不是线程安全的
    lock.acquire()  
    
    try:
        # 尝试执行优化过程
        res = optimize_in_scipy(on_core_i, fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol, callback,
                                options)
        print('*ok! on core [%s]' % on_core_i, end='\n')  # 优化成功提示
        
    except func_timeout.exceptions.FunctionTimedOut:
        # 捕获超时异常
        print('Warning: optimize_in_scipy time out! on core [%s]' % on_core_i, end='\n')
        res = None  # 超时时返回None
        
    finally:
        # 无论成功还是失败，都要释放锁
        lock.release()
    
    return res


def multistart(on_core_i, fun, x0, nrestart=1, full_output=False, args=(), method=None, jac=None, hess=None, hessp=None,
               bounds=None, constraints=(), tol=None, callback=None, options=None):
    """
    多重启动优化函数：通过多个不同初始点进行优化，寻找全局最优解
    
    背景：数值优化往往容易陷入局部最优解，通过多次从不同起点开始优化，
         可以大大提高找到全局最优解的概率
    
    参数:
        on_core_i: CPU核心编号
        fun: 目标函数
        x0: 初始参数猜测值
        nrestart: 重启次数（总共进行nrestart次优化）
        full_output: 是否返回所有结果（False时只返回最佳结果）
        其他参数: scipy优化器参数
    
    返回:
        最佳优化结果，或所有结果的排序列表
    """
    
    # 创建结果存储数组
    res_list = np.empty(nrestart, dtype=object)
    
    # ========== 第一次优化：使用原始初始点 ==========
    res = optimize_in_scipy_with_timeout_check(on_core_i, fun, x0, args, method, jac, hess, hessp, bounds, constraints,
                                               tol, callback, options)
    # print(res)  # 调试输出（已注释）
    res_list[0] = res
    
    # ========== 多重启动优化：使用随机扰动的初始点 ==========
    for i in range(nrestart - 1):
        # 生成随机扰动的新初始点
        # 在原始初始点基础上添加高斯噪声（均值=0，标准差=1）
        new_x0 = x0 + np.array([random.gauss(0, 1) * 1. for _ in range(x0.shape[0])])
        
        # 从新初始点开始优化
        res = optimize_in_scipy_with_timeout_check(on_core_i, fun, new_x0, args, method, jac, hess, hessp, bounds,
                                                   constraints, tol, callback, options)
        res_list[i + 1] = res
        # print(res)  # 调试输出（已注释）

    # ========== 结果评估和排序 ==========
    res_fun_list = []
    for res in res_list:
        if res is None:  # 处理优化失败（超时）的情况
            res_fun_list.append(1e30)  # 赋予极大的函数值
        else:
            res_fun_list.append(res.fun)  # 获取目标函数值
    
    # 按目标函数值排序（从小到大，越小越好）
    sort_res_list = res_list[np.argsort(res_fun_list)]
    
    # 返回结果
    if full_output:
        return sort_res_list[0], sort_res_list  # 返回最佳结果和所有结果
    else:
        return sort_res_list[0]  # 只返回最佳结果


# ============== 常数优化核心函数 ==============
def optimizeConstants(individual, X0, sparse_A, Y, Mask, on_core_i, opt_const_dim=-1):
    """
    优化遗传编程个体中的常数参数
    
    功能：从遗传编程生成的表达式中提取常数，使用数值优化方法找到最优常数值，
         然后将优化后的常数重新设置到表达式中，以提高方程的拟合精度
    
    参数:
        individual: 遗传编程个体，包含(f_func, g_func)
        X0: 初始状态条件
        sparse_A: 稀疏邻接矩阵
        Y: 真实观测数据
        Mask: 观测掩码
        on_core_i: CPU核心编号
        opt_const_dim: 要优化的常数维度限制（-1表示优化所有常数）
    
    返回:
        优化后的个体(individual_f, individual_g)
    """
    # s_time = time.time()  # 计时用（已注释）
    
    # ========== 步骤1：提取f函数中的所有常数 ==========
    # 从f函数的表达式树中筛选出所有数值常数（排除变量名）
    constants_f = np.array(list(map(lambda n: n.value,
                                    filter(lambda n: isinstance(n, gp.Terminal) and not isinstance(n.value, str),
                                           individual[0]))))
    
    # 如果指定了常数维度限制，则只优化前opt_const_dim个常数
    if opt_const_dim > 0:
        constants_f = constants_f[:opt_const_dim] 
    num_constants_f = len(constants_f)  # f函数中的常数个数
    
    # ========== 步骤2：提取g函数中的所有常数 ==========
    # 从g函数的表达式树中筛选出所有数值常数
    constants_g = np.array(list(map(lambda n: n.value,
                                    filter(lambda n: isinstance(n, gp.Terminal) and not isinstance(n.value, str),
                                           individual[1]))))
    
    # 如果指定了常数维度限制，则只优化前opt_const_dim个常数
    if opt_const_dim > 0:
        constants_g = constants_g[:opt_const_dim]
    num_constants_g = len(constants_g)  # g函数中的常数个数
    
    # ========== 步骤3：合并所有待优化的常数 ==========
    # 将f函数和g函数的常数合并为一个优化向量
    constants = np.concatenate([constants_f, constants_g], axis=-1)

    # ========== 步骤4：如果存在常数，则进行优化 ==========
    if constants.size > 0:
        
        def setConstants(individual, constants):
            """
            将优化后的常数值设置回表达式树中
            
            参数:
                individual: 表达式树个体
                constants: 新的常数值数组
            
            返回:
                更新常数后的表达式树
            """
            optIndividual = individual
            c = 0  # 常数索引计数器
            
            # 遍历表达式树中的所有节点
            for i in range(0, len(optIndividual)):
                # 找到数值常数节点（Terminal且值不是字符串）
                if isinstance(optIndividual[i], gp.Terminal) and not isinstance(optIndividual[i].value, str) and c < len(constants):
                    # 设置新的常数值，并处理数值异常
                    optIndividual[i].value = np.nan_to_num(constants[c])
                    optIndividual[i].name = str(optIndividual[i].value)  # 更新节点名称
                    c += 1  # 移动到下一个常数
            return optIndividual

        def evaluate(constants, individual):
            """
            评估函数：使用给定常数值计算个体的适应度
            
            参数:
                constants: 待评估的常数值数组
                individual: 遗传编程个体
            
            返回:
                适应度值（损失值）
            """
            # 将常数分别设置到f函数和g函数中
            individual_f_ = setConstants(individual[0], constants[:num_constants_f])      # 前num_constants_f个常数给f函数
            individual_g_ = setConstants(individual[1], constants[num_constants_f:])     # 剩余常数给g函数
            
            # 计算适应度
            return eval_func((individual_f_, individual_g_), pset, X0, sparse_A, Y, Mask)

        def evaluateLM(constants):
            """
            评估包装函数：为scipy优化器提供统一接口
            
            参数:
                constants: 待优化的常数值数组
            
            返回:
                目标函数值
            """
            return evaluate(constants, individual)

        # ========== 步骤5：执行多重启动优化 ==========
        # 备选的优化配置（已注释，可根据需要启用）：
        # res = multistart(evaluateLM, constants, nrestart=1, method='BFGS', options={'maxiter': 8})
        # res = multistart(on_core_i, evaluateLM, constants, nrestart=1, method='BFGS', options={'maxiter': 8})
        
        # 当前使用的优化配置：单次重启，BFGS算法，无迭代限制
        res = multistart(on_core_i, evaluateLM, constants, nrestart=1, method='BFGS')
        
        # 其他备选配置：
        # res = multistart(on_core_i, evaluateLM, constants, nrestart=1, method='L-BFGS-B', options={'maxiter': 8})
        # res = multistart(on_core_i, evaluateLM, constants, nrestart=1, method='L-BFGS-B')
        # res = multistart(on_core_i, evaluateLM, constants, nrestart=1, method='BFGS')
        # res = multistart(evaluateLM, constants, nrestart=2, method='BFGS', options={'maxiter': 8})
        # res = multistart(evaluateLM, constants, nrestart=2, method='BFGS')
        
        # print(res)  # 调试输出（已注释）
        # print(time.time()-s_t)  # 计时输出（已注释）
        
        # ========== 步骤6：处理优化结果并更新个体 ==========
        if res is not None:
            # 优化成功：使用优化后的常数值更新个体
            individual_f = setConstants(individual[0], res.x[:num_constants_f])    # 更新f函数的常数
            individual_g = setConstants(individual[1], res.x[num_constants_f:])    # 更新g函数的常数
        else:
            # 优化失败（超时或其他错误）：使用随机常数值
            individual_f = setConstants(individual[0], np.array([sampling_const() for _ in range(num_constants_f)]))
            individual_g = setConstants(individual[1], np.array([sampling_const() for _ in range(num_constants_g)]))
        
        # print('opt constant\'s cost = %.2fs' % (time.time() - s_t), res.fun)  # 性能统计（已注释）

        return (individual_f, individual_g)  # 返回优化后的个体
    else:
        # 如果个体中没有常数，直接返回原个体
        return individual


# ============== 遗传算法核心组件 ==============

def tournamentSelection(P, num=12, prob=0.9):
    """
    锦标赛选择算法：遗传算法中的个体选择策略,选择优秀个体进行繁殖
    
    工作原理：
    1. 从种群中随机选择num个 个体组成锦标赛
    2. 将参赛个体按适应度排序（越小越好）
    3. 以概率prob选择最优个体，否则继续淘汰最优个体
    4. 重复直到只剩一个个体
    
    参数:
        P: 种群列表，每个个体格式为[方程对, 适应度值]
        num: 锦标赛参赛个体数量（默认12个）
        prob: 选择最优个体的概率（默认0.9）
    
    返回:
        被选中的个体
    """
    # 从种群P中随机抽取num个个体参加锦标赛
    Q = random.sample(P, num)
    
    # 锦标赛循环：逐步淘汰个体直到剩下一个
    while len(Q) > 1:
        # 按适应度排序（升序，越小的适应度越好）
        sort_index = np.argsort([ii[1] for ii in Q])
        E = Q[sort_index[0]]  # 获取当前最优个体
        
        # 以概率prob选择最优个体，否则淘汰最优个体继续比赛
        if random.uniform(0, 1) < prob:
            break  # 选择当前最优个体
        Q.remove(E)  # 淘汰最优个体，增加选择多样性
    
    return E  # 返回被选中的个体


def eval_pop(pop):
    """
    顺序评估种群中所有个体的适应度
    
    参数:
        pop: 待评估的种群列表
    
    返回:
        pop_eval: 评估后的种群，每个个体包含[方程对, 适应度值]
    """
    pop_eval = []
    
    # 遍历种群中的每个个体，显示进度条
    for idx in tqdm(range(len(pop))):
        # 提取个体的f函数和g函数
        ind_i_1 = pop[idx][0][0]  # f函数（节点内部动力学）
        ind_i_2 = pop[idx][0][1]  # g函数（节点间相互作用）
        
        # 计算个体适应度
        ind_i_fitness = eval_func((ind_i_1, ind_i_2), pset, X0, sparse_A, Y, Mask)
        
        # 存储评估结果：[方程对, 适应度值]
        pop_eval.append([(ind_i_1, ind_i_2), ind_i_fitness])
    
    return pop_eval


def multi_processing_eval_pop(pop, num_running_each_core=10):
    """
    多进程并行评估种群适应度：大幅提升评估效率
    
    工作原理：
    1. 将种群分成多个批次
    2. 每个批次分配给不同的CPU核心并行处理
    3. 等待所有进程完成后合并结果
    
    参数:
        pop: 待评估的种群
        num_running_each_core: 每个核心处理的个体数量
    
    返回:
        new_pop: 评估完成的种群
    """
    store_all = []   # 存储异步任务对象
    store_all2 = []  # 存储任务结果
    
    # ========== 创建多进程池并分配任务 ==========
    with multiprocessing.Pool() as p:
        # 计算需要的批次数量
        num_batches = math.ceil(len(pop) / num_running_each_core)
        
        # 为每个批次创建异步评估任务
        for i in range(num_batches):
            # 计算当前批次的个体范围
            start_idx = i * num_running_each_core
            end_idx = start_idx + num_running_each_core
            current_batch = pop[start_idx:end_idx]
            
            # 提交异步评估任务
            pop_eval = p.apply_async(eval_pop, args=(current_batch,))
            store_all.append(pop_eval)
        
        # 关闭进程池，等待所有任务完成
        p.close()  # 不再接受新任务
        p.join()   # 等待所有进程完成
        
        # ========== 收集所有任务结果 ==========
        # for i in tqdm(store_all):  # 可选：显示进度条
        for i in store_all:
            store_all2.append(i.get())  # 获取异步任务结果

    # ========== 合并所有批次的评估结果 ==========
    new_pop = []
    for store_i in store_all2:
        new_pop += store_i  # 将各批次结果合并到统一列表中
    
    # print(len(new_pop))  # 调试输出：打印最终种群大小

    return new_pop


# ============== 遗传算法进化和选择策略 ==============

def evolve(pop, num_evo):
    """
    遗传算法进化函数：通过选择、变异产生新一代个体
    
    工作流程：
    1. 从当前种群中选择优秀个体作为父代
    2. 对父代进行变异操作产生新个体
    3. 新个体初始适应度设为极大值（待评估）
    
    参数:
        pop: 当前种群
        num_evo: 要产生的新个体数量
    
    返回:
        pop_new: 新产生的个体列表
    """
    pop_new = []
    
    # 循环产生指定数量的新个体
    for _ in range(num_evo):
        # 步骤1：使用锦标赛选择算法选择一个优秀个体作为父代
        ind_one = tournamentSelection(pop, num=12)
        
        # 步骤2：对选中的个体进行变异操作
        # gp.Mutations_f_g 对f函数和g函数同时进行变异
        ind_one_new = gp.Mutations_f_g(
            ind_one[0],           # 原个体的方程对(f, g)  
            pset,                 # 原语集合
            sampling_const,       # 常数采样函数
            converter,            # 表达式转换器
            min_=min_,           # 最小树深度
            max_=max_            # 最大树深度
        )
        
        # ind_one_old, ind_one_fit_old = pop.pop(0)  # 旧版本代码（已注释）
        
        # 步骤3：将新个体加入种群，初始适应度设为极大值（表示未评估）
        pop_new.append([ind_one_new, 1e30])
    
    return pop_new


def choose(pop, num_choose):
    """
    复杂度感知选择：结合适应度和复杂度进行个体选择
    
    选择策略：
    - 优先选择适应度好的个体
    - 对复杂度过于集中的个体进行惩罚，鼓励多样性
    - 稀有复杂度的个体获得选择优势
    
    参数:
        pop: 种群列表
        num_choose: 要选择的个体数量
    
    返回:
        pop_choose: 选中的个体列表
    """
    # 统计种群中各种复杂度的分布情况
    Stat_Complex = stat_complex(pop)
    
    fitness_list = []
    for p_i in pop:
        # 计算个体的复杂度（f函数和g函数节点数之和）
        complex_of_eq = len(p_i[0][0]) + len(p_i[0][1])
        
        # 计算调整后的适应度：原适应度 × 复杂度惩罚因子
        # 复杂度越稀有（在种群中占比越小），惩罚越小
        complexity_penalty = np.exp(float(Stat_Complex[complex_of_eq] / np.sum(list(Stat_Complex.values()))))
        adjusted_fitness = p_i[1] * complexity_penalty
        
        fitness_list.append(adjusted_fitness)
    
    # 按调整后的适应度排序，选择前num_choose个个体
    pop_choose = np.array(pop)[np.argsort(fitness_list)[:num_choose]].tolist()
    
    # 打印选中个体的复杂度分布（用于监控多样性）
    print([len(p_i[0][0])+len(p_i[0][1]) for p_i in pop_choose])
    
    return pop_choose


def choose_best(pop, num_choose):
    """
    简单最优选择：纯粹基于适应度选择最佳个体
    
    选择策略：直接按适应度排序，选择最优的个体
    
    参数:
        pop: 种群列表
        num_choose: 要选择的个体数量（-1表示选择所有个体）
    
    返回:
        pop_choose: 按适应度排序的个体列表
    """
    if num_choose == -1:
        # 选择所有个体，按适应度排序
        pop_choose = np.array(pop)[np.argsort([p_i[1] for p_i in pop])].tolist()
    else:
        # 选择前num_choose个最优个体
        pop_choose = np.array(pop)[np.argsort([p_i[1] for p_i in pop])[:num_choose]].tolist()
    
    # print([len(p_i[0][0])+len(p_i[0][1]) for p_i in pop_choose])  # 调试输出（已注释）
    return pop_choose


def choose_diversity(pop, num_choose):
    """
    多样性优先选择：在每个复杂度级别中选择最优个体
    
    选择策略：
    1. 将种群按复杂度分组
    2. 每个复杂度组选择一个最优个体
    3. 确保选中的个体具有不同的复杂度，最大化多样性
    
    参数:
        pop: 种群列表
        num_choose: 要选择的个体数量（None表示每个复杂度选一个）
    
    返回:
        pop_choose: 多样性选择的个体列表
    """
    # 使用get_return_pop函数进行多样性选择
    # filter=False表示不过滤高误差个体，保留各复杂度的代表
    pop_choose = get_return_pop(pop, filter=False)
    
    if num_choose is None:
        return pop_choose  # 返回所有复杂度级别的代表个体
    
    return pop_choose[:num_choose]  # 返回前num_choose个个体


# ============== 多进程常数优化系统 ==============

def opt_constant_one_core(pop, core_i, opt_const_dim=-1):
    """
    单核心常数优化函数：在指定CPU核心上优化一批个体的常数参数
    
    功能：对分配给当前核心的个体进行常数优化，包括：
    1. 提取个体中的常数参数
    2. 使用数值优化算法找到最优常数值
    3. 重新计算优化后个体的适应度
    
    参数:
        pop: 分配给当前核心的个体列表
        core_i: 当前CPU核心编号（用于调试和进度跟踪）
        opt_const_dim: 要优化的常数维度限制（-1表示优化所有常数）
    
    返回:
        pop_opt: 优化后的个体列表，格式为[[个体, 适应度], ...]
    """
    pop_opt = []
    
    # 遍历分配给当前核心的所有个体，显示进度条
    for i in tqdm(range(len(pop))):
        # print(' on core [%s] : %s/%s ... ' % (core_i, i + 1, len(pop)))  # 调试信息（已注释）
        
        # 步骤1：深拷贝个体，避免修改原始数据
        ind_i = copy.deepcopy(pop[i][0])
        
        # 步骤2：优化个体中的常数参数
        # optimizeConstants函数会：
        # - 提取f函数和g函数中的所有常数
        # - 使用scipy优化器找到最优常数值
        # - 将优化后的常数设置回表达式树中
        ind_i = optimizeConstants(ind_i, X0, sparse_A, Y, Mask, core_i, opt_const_dim=opt_const_dim)
        
        # 步骤3：重新计算优化后个体的适应度
        ind_i_fitness = eval_func(ind_i, pset, X0, sparse_A, Y, Mask)
        
        # 步骤4：保存优化结果
        pop_opt.append([ind_i, ind_i_fitness])
    
    return pop_opt


def multi_processing_opt_constant(pop, num_running_each_core=10, opt_const_dim=-1):
    """
    多进程并行常数优化主函数：将种群分批并行优化常数参数
    
    工作流程：
    1. 将种群按指定大小分成多个批次
    2. 每个批次分配给不同的CPU核心并行处理
    3. 等待所有核心完成优化任务
    4. 合并所有批次的优化结果
    
    性能优势：
    - 充分利用多核CPU资源
    - 大幅减少常数优化的总时间
    - 适合处理大规模种群
    
    参数:
        pop: 待优化的种群列表
        num_running_each_core: 每个核心处理的个体数量（默认10个）
        opt_const_dim: 要优化的常数维度限制（-1表示优化所有常数）
    
    返回:
        new_pop: 完成常数优化的新种群
    """
    store_all = []   # 存储异步任务对象
    store_all2 = []  # 存储任务执行结果
    
    # ========== 创建多进程池并分配优化任务 ==========
    with multiprocessing.Pool() as p:
        # 计算需要的批次数量（每个批次包含num_running_each_core个个体）
        num_batches = int(len(pop) / num_running_each_core)
        
        # 为每个批次创建异步优化任务
        for i in range(num_batches):
            # 计算当前批次的个体范围
            start_idx = i * num_running_each_core
            end_idx = start_idx + num_running_each_core
            current_batch = pop[start_idx:end_idx]
            
            # 提交异步常数优化任务
            # 每个任务在独立的CPU核心上执行opt_constant_one_core函数
            pop_opt = p.apply_async(opt_constant_one_core, args=(
                current_batch,     # 当前批次的个体
                i + 1,            # 核心编号（从1开始）
                opt_const_dim     # 常数维度限制
            ))
            store_all.append(pop_opt)
        
        # 关闭进程池，等待所有任务完成
        p.close()  # 不再接受新任务
        p.join()   # 等待所有进程完成
        
        # ========== 收集所有任务的执行结果 ==========
        # for i in tqdm(store_all):  # 可选：显示进度条
        for i in store_all:
            store_all2.append(i.get())  # 获取异步任务的结果

    # ========== 合并所有批次的优化结果 ==========
    new_pop = []
    for store_i in store_all2:
        new_pop += store_i  # 将各批次的优化结果合并到统一列表中
    
    # 输出优化后的种群大小（用于验证数据完整性）
    print(len(new_pop))

    return new_pop


def get_return_pop(pop, filter=True):
    """
    多样性优先种群筛选函数：从种群中选择具有不同复杂度的代表性个体
    
    核心思想：
    - 将种群按复杂度（表达式节点数）分组
    - 每个复杂度组选择适应度最好的个体
    - 确保返回的个体具有最大的结构多样性
    - 可选择性过滤掉误差过大的个体
    
    参数:
        pop: 输入种群列表，格式为[[方程对, 适应度], ...]
        filter: 是否过滤高误差个体（True=过滤，False=保留所有）
    
    返回:
        best_ind_list_filter: 筛选后的多样性个体列表
    """
    
    # ========== 步骤1：计算每个个体的复杂度 ==========
    complex_of_eq_list = []  # 存储所有个体的复杂度
    
    for idx in range(len(pop)):
        # 计算个体复杂度：f函数节点数 + g函数节点数
        complex_of_eq = len(pop[idx][0][0]) + len(pop[idx][0][1])
        
        # 将复杂度添加到个体信息中：[方程对, 适应度, 复杂度]
        pop[idx].append(complex_of_eq)
        complex_of_eq_list.append(complex_of_eq)
    
    # ========== 步骤2：按复杂度分组选择最优个体 ==========
    # 获取所有不同的复杂度值，并排序
    complex_of_eq_list_sort = sorted(list(set(complex_of_eq_list)))
    
    best_ind_list = []  # 存储每个复杂度组的最优个体
    
    # 遍历每个复杂度级别
    for complex_i in complex_of_eq_list_sort:
        best_ind_complex_i = None  # 当前复杂度组的最优个体
        
        # 在当前复杂度组中寻找适应度最好的个体
        for p_i in pop:
            if p_i[-1] == complex_i:  # 如果个体属于当前复杂度组
                # 如果是第一个个体，或者适应度更好（值更小）
                if best_ind_complex_i is None or p_i[1] < best_ind_complex_i[1]:
                    best_ind_complex_i = p_i
        
        # 将当前复杂度组的最优个体添加到结果列表
        best_ind_list.append(best_ind_complex_i)

    # ========== 步骤3：可选的误差过滤 ==========
    if filter:
        # 过滤模式：只保留误差在容忍范围内的个体
        best_ind_list_filter = []
        for ind in best_ind_list:
            if ind[1] <= tol_err:  # 适应度小于等于误差容忍阈值
                best_ind_list_filter.append(ind)
    else:
        # 非过滤模式：保留所有复杂度组的代表个体
        best_ind_list_filter = best_ind_list

    return best_ind_list_filter


# ============== 主函数：完整的符号回归筛选流程 ==============
def main(case_name):
    """
    符号回归主函数：从预训练知识库中筛选最佳网络动力学方程
    
    完整流程：
    1. 加载预训练的方程字符串库
    2. 转换为遗传编程树结构
    3. 可选的常数优化
    4. 评估所有候选方程的适应度
    5. 筛选多样性最佳方程
    6. 输出结果到CSV文件
    
    参数:
        case_name: 案例名称（如'heat_dim=0', 'mutu_dim=1'等）
    """
    
    # ========== 步骤1：初始化和设置 ==========
    seed = 666  # 固定随机种子，确保结果可重现
    random.seed(seed)
    np.random.seed(seed)

    s_time = time.time()  # 记录开始时间
    
    POP_saved_to_draw = []  # 预留：用于保存绘图数据（当前未使用）
    
    # ========== 步骤2：从文件加载预训练方程库 ==========
    print('\r\n*init pop from file ...')
    
    POP = []  # 初始化种群列表
    
    # 读取预训练的方程字符串CSV文件
    # 文件格式：每行包含f_eq_str（f函数字符串）和g_eq_str（g函数字符串）
    init_pop_str = pd.read_csv(
        'search_2nd_phase/eq_str_%s%s.csv' % (case_name, add_str), 
        header=None, 
        names=['f_eq_str', 'g_eq_str']
    )

    # 遍历所有预训练方程，转换为遗传编程树
    for ii in tqdm(range(len(init_pop_str))):
        # 提取f函数和g函数的字符串表达式
        f_eq_str = init_pop_str.iloc[ii]['f_eq_str']  # f函数字符串
        g_eq_str = init_pop_str.iloc[ii]['g_eq_str']  # g函数字符串
        
        # 将字符串表达式转换为遗传编程树
        b1 = gp.PrimitiveTree.from_string_sympy(f_eq_str, pset_f)  # f函数树
        b2 = gp.PrimitiveTree.from_string_sympy(g_eq_str, pset_g)  # g函数树
        
        # 添加到种群中，初始适应度设为极大值（表示未评估）
        POP.append([(b1, b2), 1e30])
    
    
    # ========== 步骤3：可选的线性缩放扩展（已注释的备选方案） ==========
    # 以下代码可以对每个方程进行线性变换，增加种群多样性
    # # linear scale
    # print('\r\n*linear scale ...')
    # for ii in tqdm(range(len(init_pop_str))):
    #     f_eq_str = init_pop_str.iloc[ii]['f_eq_str']
    #     g_eq_str = init_pop_str.iloc[ii]['g_eq_str']
    #     linear_a, linear_b = random.uniform(min_const, max_const), random.uniform(min_const, max_const)
    #     f_eq_str_scaled = "%.4f + %.4f * (%s)"%(linear_a, linear_b, f_eq_str)  # f_scaled = a + b*f
    #     b1 = gp.PrimitiveTree.from_string_sympy(f_eq_str_scaled, pset_f)
    #     linear_a, linear_b = random.uniform(min_const, max_const), random.uniform(min_const, max_const)
    #     g_eq_str_scaled = "%.4f + %.4f * (%s)"%(linear_a, linear_b, g_eq_str)  # g_scaled = a + b*g
    #     b2 = gp.PrimitiveTree.from_string_sympy(g_eq_str_scaled, pset_g)
    #     POP.append([(b1, b2), 1e30])
    #
    # print('\r\n* opt linear scale ...')

    # ========== 步骤4：可选的常数优化 ==========
    # 如果启用常数优化标志（opt_flag=1），则对所有个体进行常数参数优化
    if opt_flag == 1:
        print('\r\n* optimizing constants ...')
        # s_opt_time = time.time()  # 计时开始（已注释）
        
        # 使用多进程并行优化常数
        # list(reversed(POP)): 反转种群顺序（可能为了负载均衡）
        # num_running_each_core: 每个CPU核心处理的个体数量
        # opt_const_dim=-1: 优化所有常数维度
        POP = multi_processing_opt_constant(
            list(reversed(POP)),
            num_running_each_core=max(1, len(POP) // multiprocessing.cpu_count()), 
            opt_const_dim=-1
        )
        # print("-opt cost = %.2fs" % (time.time() - s_opt_time))  # 输出优化耗时（已注释）

    # ========== 步骤5：评估所有个体的适应度 ==========
    # 备选评估方案（已注释）：
    # print('\r\n* eval linear scale ...')
    # POP = multi_processing_eval_pop(POP, num_running_each_core=max(1, len(POP) // multiprocessing.cpu_count()))
    # POP = eval_pop(POP)  # 单进程评估（较慢）
    
    print('\r\n* eval init pop ...')
    # 使用多进程并行评估所有个体的适应度
    POP = multi_processing_eval_pop(
        POP, 
        num_running_each_core=max(1, len(POP) // multiprocessing.cpu_count())
    )
    # ========== 步骤6：分析和输出初步结果 ==========
    # 备选的详细分析输出（已注释）：
    # best_ind = POP[np.argsort([s_i[1] for s_i in POP])[0]]
    # ii = -1
    # print('**[ %s ] mean :' % (ii + 1), np.mean([s_i[1] for s_i in POP]), 'best :', best_ind[1], ' | ',
    #       str(sp.simplify(
    #           gp.TreeToDeadStr(best_ind[0][0], converter))), ' | ', str(sp.simplify(
    #         gp.TreeToDeadStr(best_ind[0][1], converter))),
    #       "time cost = %.2fs" % (time.time() - s_time))
    
    init_pop_size = len(POP)  # 记录初始种群大小
    
    # 找到当前最佳个体（适应度最小的个体）
    best_ind = POP[np.argsort([s_i[1] for s_i in POP])[0]]
    ii = -1  # 用于标识当前是初始评估阶段
    
    # 输出评估结果统计信息
    print('**[ %s ] mean :' % (ii + 1),                    # 阶段标识
          np.mean([s_i[1] for s_i in POP]),                # 种群平均适应度
          'best :', best_ind[1], ' | ',                    # 最佳适应度
          str(sp.simplify(gp.TreeToDeadStr(best_ind[0][0], converter))), ' | ',  # 最佳f函数
          str(sp.simplify(gp.TreeToDeadStr(best_ind[0][1], converter))),         # 最佳g函数
          "time cost = %.2fs" % (time.time() - s_time))    # 总耗时

    # ========== 步骤7：筛选多样性最佳方程 ==========
    # 使用多样性选择策略，每个复杂度级别选择一个最优代表
    best_list = get_return_pop(POP, filter=False)
    
    # 将筛选出的代表个体按适应度排序（从好到差）
    best_list = np.array(best_list)[np.argsort([s_i[1] for s_i in best_list])]
    
    # ========== 步骤8：构建输出数据并保存到CSV ==========
    # 准备输出数据的字典结构
    rows = {
        'case_name': [],  # 案例名称
        'f_eq': [],       # f函数的数学表达式（字符串形式）
        'g_eq': [],       # g函数的数学表达式（字符串形式）
        'fitness': [],    # 适应度值
        'complex': []     # 复杂度值
    }
    
    # 取前10个最佳的多样性代表个体
    for best_i in best_list[:10]:
        rows['case_name'].append(case_name)
        
        # 将遗传编程树转换为简化的数学表达式字符串
        rows['f_eq'].append(str(sp.simplify(gp.TreeToDeadStr(best_i[0][0], converter))))
        rows['g_eq'].append(str(sp.simplify(gp.TreeToDeadStr(best_i[0][1], converter))))
        
        rows['fitness'].append(str(best_i[1]))    # 适应度
        rows['complex'].append(str(best_i[-1]))   # 复杂度
    
    # 保存结果到CSV文件
    output_filename = 'pretrain_knowledge_eq_%s_dim%s_optflag%s_addstr%s.csv' % (
        args.case_name, args.dim, args.opt_flag, args.add_str
    )
    pd.DataFrame(rows).to_csv(output_filename)
    print(f'Results saved to: {output_filename}')
    

# ============== 程序入口 ==============
if __name__ == "__main__":
    """
    程序主入口：批量处理多个维度的案例
    
    执行流程：
    - 遍历case_name_list中的所有案例（不同维度）
    - 每个案例独立执行完整的符号回归筛选流程
    - 为每个案例生成独立的结果CSV文件
    
    案例列表格式：['heat_dim=0', 'heat_dim=1', ...] 或类似
    """
    # 批量处理所有指定的案例维度
    for case_name in case_name_list:
        print(f'\n{"="*50}')
        print(f'Processing case: {case_name}')
        print(f'{"="*50}')
        
        # 执行主函数：完整的符号回归筛选流程
        main(case_name)
        
        print(f'Completed case: {case_name}')
    
    print(f'\n{"="*50}')
    print('All cases completed successfully!')
    print(f'{"="*50}')
