#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
符号回归工具模块
主要功能：从数据中自动发现数学表达式，支持多种符号回归算法
支持的算法：gplearn(遗传编程), PySR(Julia后端), transformer(神经网络)
"""

import time

# 导入符号回归相关库
import pysr  # PySR符号回归库
import torch  # PyTorch深度学习框架

# 初始化Julia环境(PySR需要)
pysr.julia_helpers.init_julia()
from pysr import PySRRegressor  # PySR回归器
from gplearn.genetic import SymbolicRegressor  # 基于遗传编程的符号回归
import sympy as sp  # 符号数学库

# 导入基于Transformer的符号回归模型
from transformer_sr.NSR_fast import FastNSR

# 设置快速模式标志
fast_flag = True

# 初始化神经符号回归模型
nsr = FastNSR(fast_flag)

def perform_symbolic_regression(X_train, Y_train, SRtype):
    """
    执行符号回归分析
    
    参数:
        X_train: 训练数据的输入特征 (numpy array或tensor)
        Y_train: 训练数据的目标值 (numpy array或tensor)
        SRtype: 符号回归算法类型，可选 'gplearn', 'pysr', 'transformer'
    
    返回:
        res_str_reduced: 发现的数学表达式(简化后)
        predictions: 模型在训练数据上的预测值
    """
    
    if SRtype == 'gplearn':
        # 使用gplearn(基于遗传编程的符号回归)
        
        # 定义允许使用的数学函数集合
        function_set = ['add',     # 加法
                        'sub',     # 减法
                        'mul',     # 乘法
                        'div',     # 除法
                        'sqrt',    # 平方根
                        'log',     # 对数
                        # 'abs',   # 绝对值(注释掉)
                        'neg',     # 负数
                        'inv',     # 倒数
                        ]
        
        # 函数转换器：将gplearn的函数名转换为sympy表达式
        converter = {
            'sub': lambda x, y: x - y,      # 减法
            'div': lambda x, y: x / y,      # 除法
            'mul': lambda x, y: x * y,      # 乘法
            'add': lambda x, y: x + y,      # 加法
            'neg': lambda x: -x,            # 负数
            'pow': lambda x, y: x ** y,     # 幂运算
            'cos': lambda x: sp.cos(x),     # 余弦函数
            'inv': lambda x: x ** (-1),     # 倒数
            'sqrt': lambda x: sp.sqrt(x),   # 平方根
        }
        # 创建遗传编程符号回归器
        est_gp = SymbolicRegressor(
            population_size=5000,           # 种群大小：每代包含5000个个体
            generations=50,                 # 进化代数：最多进化50代
            function_set=function_set,      # 使用的函数集合
            stopping_criteria=0.0001,      # 停止条件：误差小于0.0001时停止
            p_crossover=0.7,               # 交叉概率：70%
            p_subtree_mutation=0.1,        # 子树变异概率：10%
            p_hoist_mutation=0.05,         # 提升变异概率：5%
            p_point_mutation=0.1,          # 点变异概率：10%
            max_samples=0.9,               # 最大样本使用比例：90%
            verbose=1,                     # 详细输出模式
            parsimony_coefficient=0.01,    # 简约系数：控制模型复杂度
            random_state=0                 # 随机种子：保证结果可重现
        )
        
        # 训练模型
        est_gp.fit(X_train, Y_train)
        
        # 获取最佳表达式并简化
        res_str_reduced = sp.simplify(sp.sympify(str(est_gp._program), locals=converter))
    elif SRtype == 'pysr':
        # 使用PySR(基于Julia的高性能符号回归)
        
        SR_model = PySRRegressor(
            binary_operators=["+", "-", "*", "/"],      # 二元运算符：加减乘除
            # binary_operators=["+", "-", "*", "/", "^"],  # 可选：添加幂运算
            # unary_operators=["sin", "cos", "exp", "log", "sqrt", "abs"],  # 可选：更多一元运算符
            unary_operators=["exp", "sin", "cos"],      # 一元运算符：指数、正弦、余弦
            
            # 嵌套约束：防止函数过度嵌套（如exp(exp(x))等）
            nested_constraints={
                "exp": {"exp": 0, "sin": 0, "cos": 0},  # exp函数内不能再嵌套这些函数
                "sin": {"exp": 0, "sin": 0, "cos": 0},  # sin函数内不能再嵌套这些函数
                "cos": {"exp": 0, "sin": 0, "cos": 0}   # cos函数内不能再嵌套这些函数
            },
            # unary_operators=[ "exp"],                   # 可选：只使用exp函数
            # nested_constraints={"exp": {"exp": 0},},    # 可选：exp函数不能嵌套
            
            niterations=10,          # 迭代次数：10次
            populations=50,          # 种群数量：50个种群
            population_size=100,     # 每个种群大小：100个个体
            
            # 早停条件：当损失小于1e-4且复杂度小于20时停止
            early_stop_condition=(
                "stop_if(loss, complexity) = loss < 1e-4 && complexity < 20"
                # 如果找到了好的且简单的方程就提前停止
            ),
            
            timeout_in_seconds=60 * 60 * 0.5,  # 超时设置：30分钟后停止
            # ^ 或者可以设置24小时后停止
            
            maxsize=20,              # 最大表达式大小：20
            # ^ 允许更高的复杂度
            
            maxdepth=10,             # 最大嵌套深度：10
            # ^ 但是避免过度深度嵌套
            
            loss="L1DistLoss()",     # 损失函数：L1距离损失
            denoise=True,            # 启用去噪功能
            temp_equation_file=True, # 使用临时方程文件
        )
        
        # 训练模型
        SR_model.fit(X_train, Y_train)

        # 获取前10个最佳表达式（按复杂度排序）
        res_str_reduced = [SR_model.sympy(i) for i in range(len(SR_model.equations_['equation']))][
                          :10]  # 截取前10个，最大复杂度为20

        # print(SR_model)  # 可选：打印模型信息
    elif SRtype == 'transformer':
        # 使用基于Transformer的神经符号回归
        
        SR_model = FastNSR(True)  # 创建快速神经符号回归模型

        # 为了获得最佳结果，确保数据在合适的支持范围内
        max_supp = torch.max(X_train)  # 获取训练数据的最大值
        min_supp = torch.min(X_train)  # 获取训练数据的最小值

        num_vars = X_train.size(-1)  # 获取变量维度数
        # 数据标准化处理
        X = X_train * (max_supp - min_supp) + min_supp

        # 运行神经符号回归模型
        prediction = nsr.run(X, torch.transpose(Y_train, 0, 1), dim=0)

        # 获取前10个最佳表达式（按复杂度排序）
        res_str_reduced = [SR_model.sympy(i) for i in range(len(SR_model.equations_['equation']))][
                          :10]  # 截取前10个，最大复杂度为20

        # print(SR_model)  # 可选：打印模型信息
        
    else:
        # 未知的符号回归类型
        print("Unknown SRtype [%s] in performing symbolic regression" % SRtype)
        exit(1)

    # 返回发现的数学表达式和模型预测值
    return res_str_reduced, SR_model.predict(X_train)
