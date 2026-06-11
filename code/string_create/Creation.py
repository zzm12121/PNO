# ==============================================================================
# 动力学方程字符串生成器
# Dynamics Equation String Generator
# 
# 功能：自动生成各种动力学方程的字符串表达式，用于创建大规模训练数据集
# 
# 在整个系统中的位置：
#   Creation.py → 生成方程字符串 → DynamicsEquation_new.py → 解析并计算 → NetworkSystem_new.py → 仿真数据
# 
# 核心特性：
# 1. 基于符号数学的方程骨架生成
# 2. 随机参数采样生成方程变体
# 3. 支持固定方程模板和随机生成两种模式
# 4. 自动过滤无效和重复的方程
# 5. 批量生成并导出为数据集
# 
# 生成的方程类型：
# - 自身动力学方程：f(x_1) 如 "0.1*x_1 - 0.2*x_1**2"
# - 相互作用方程：g(x_1, x_2) 如 "0.5*(x_2 - x_1)"
# ==============================================================================

import pandas as pd
import numpy as np
import sympy as sp                          # 符号数学库
from string_create.lib import class_utils   # 方程类工具
from string_create.lib import load_utils    # 加载配置工具
from string_create.lib import generate_utils # 生成工具
from string_create.lib import data_utils    # 数据处理工具


class Creation:
    """
    动力学方程字符串生成器主类
    
    这个类负责：
    1. 创建方程生成环境
    2. 生成方程骨架（结构模板）
    3. 为骨架填入随机参数
    4. 生成大量方程变体并保存为数据集
    """
    
    def __init__(self, cfg):
        """
        初始化方程生成器
        
        参数:
            cfg: 配置对象，包含生成参数
                - type: 方程类型（如 "HeatDiffusion", "GeneRegulatory" 等）
                - num: 要生成的方程骨架数量
                - fixed: 是否使用固定的方程模板（True）或随机生成（False）
                - fixed_expr: 固定方程的表达式（当 fixed=True 时使用）
                - skeleton_detail: 骨架生成的详细配置
                - const_detail: 常数采样的详细配置
        """
        # 基本配置参数
        self.type_eq = cfg.type                     # 方程类型
        self.num_eq = cfg.num                       # 要生成的方程数量
        self.fixed = cfg.fixed                      # 是否使用固定模板
        self.fixed_expr = cfg.fixed_expr            # 固定方程表达式
        self.config_eq = cfg.skeleton_detail        # 骨架生成配置
        self.config_eq_const = cfg.const_detail     # 常数采样配置

        # 创建方程生成环境（包含变量、操作符、约束等）
        self.env = load_utils.create_env(self.fixed, self.config_eq)
        
        # 生成方程骨架列表（结构模板，不含具体参数值）
        self.skeleton_list = self.skeleton_create(self.fixed, self.fixed_expr, self.env, self.num_eq)
        
        # 创建常数采样器
        self.const = load_utils.const(self.fixed, self.config_eq_const)
        
        # 生成完整的方程数据集（为每个骨架生成多个参数变体）
        self.equation_dataset = self.equation_create(self.fixed, self.const, self.skeleton_list, cfg)

        # 保存数据集到 CSV 文件
        self.file_path = 'data/%s_dataset_%s.csv' % (self.type_eq, cfg.num * cfg.const_detail.num_of_sample)
        self.equation_dataset.to_csv(self.file_path, index=False)
        print(f"方程数据集已保存到: {self.file_path}")
        print(f"总共生成了 {len(self.equation_dataset)} 个方程")

    def skeleton_create(self, fixed_flag, fixed_expr, env, num_eq):
        """
        生成方程骨架（结构模板）
        
        骨架是方程的基本结构，包含变量和占位符常数，但不包含具体的数值。
        例如："{c1}*x_1 + {c2}*x_1**2" 是一个骨架，{c1}, {c2} 是常数占位符
        
        参数:
            fixed_flag: 是否使用固定方程（True）或随机生成（False）
            fixed_expr: 固定方程的 sympy 表达式
            env: 方程生成环境，包含变量、操作符等
            num_eq: 要生成的骨架数量
            
        返回:
            skeletons: 方程骨架对象列表，每个骨架包含表达式模板和常数字典
        """
        count = 0
        skeletons = []
        
        if fixed_flag:
            # 模式1：使用固定的方程模板
            print(f"使用固定方程模板生成 {num_eq} 个骨架...")
            
            while count < num_eq:
                # 简化固定表达式
                fixed_eq = sp.simplify(fixed_expr)
                
                # 将 sympy 表达式转换为前缀表示
                prefix = env.sympy_to_prefix(fixed_eq)
                
                # 将前缀表示转换回中缀表示（字符串形式）
                infix, _ = env._prefix_to_infix(prefix, coefficients=env.coefficients, variables=env.variables)
                
                print(f"环境中的常数: {env.const}")
                
                # 创建常数占位符字典
                consts_elemns = {y: y for x in env.const for y in x}
                
                # 创建方程骨架对象
                skeletons.append(class_utils.GenerateEquation(expr=infix, coeff_dict=consts_elemns))
                count = count + 1

        else:
            # 模式2：随机生成方程骨架
            print(f"随机生成 {num_eq} 个方程骨架...")
            
            while count < num_eq:
                try:
                    # 随机生成前缀表达式
                    prefix, variables = env.generate_equation(np.random)
                    
                    # 添加常数标识符
                    prefix = env.add_identifier_constants(prefix)
                    
                    # 提取常数信息
                    consts = env.return_constants(prefix)
                    
                    # 转换为中缀表达式
                    infix, _ = env._prefix_to_infix(prefix, coefficients=env.coefficients, variables=env.variables)
                    
                    # 创建常数占位符字典
                    consts_elemns = {y: y for x in consts.values() for y in x}

                    # 创建方程骨架对象
                    skeletons.append(class_utils.GenerateEquation(expr=infix, coeff_dict=consts_elemns))
                    count = count + 1
                    
                    if count % 10 == 0:
                        print(f"已生成 {count}/{num_eq} 个骨架")
                        
                # 异常处理：各种可能的生成错误
                except TimeoutError:
                    continue  # 超时，重试
                except generate_utils.NotCorrectIndependentVariables:
                    continue  # 变量不正确，重试
                except generate_utils.UnknownSymPyOperator:
                    continue  # 未知操作符，重试
                except generate_utils.ValueErrorExpression:
                    continue  # 表达式值错误，重试
                except generate_utils.ImAccomulationBounds:
                    continue  # 累积边界问题，重试
                except RecursionError:
                    continue  # 递归错误，重试
                except KeyError:
                    continue  # 键值错误，重试
                except TypeError:
                    continue  # 类型错误，重试
                except Exception as E:
                    continue  # 其他异常，重试
                    
        print(f"成功生成 {len(skeletons)} 个方程骨架")
        return skeletons

    def equation_create(self, fixed_flag, const, skeleton_list, cfg):
        """
        为每个方程骨架生成具体的参数值，产生最终的方程字符串
        
        这个方法是数据集生成的核心，它：
        1. 为每个骨架采样随机的常数值
        2. 将常数值填入骨架，生成完整的方程字符串
        3. 过滤掉无效的方程（包含 zoo, I, 缺少变量等）
        4. 避免重复的方程
        5. 根据方程类型应用不同的验证规则
        
        参数:
            fixed_flag: 是否使用固定方程
            const: 常数采样配置对象
            skeleton_list: 方程骨架列表
            cfg: 配置对象
            
        返回:
            pd.DataFrame: 包含所有有效方程的数据集，列名为 'equation'
        """
        print(f"开始为 {len(skeleton_list)} 个骨架生成方程，每个骨架生成 {const.num_sample} 个变体...")
        
        # 根据方程类型判断处理方式
        # 如果类型以 "F" 结尾，通常表示一维（单变量）方程，如 HeatDiffusionF
        if cfg.type[-1] == "F":
            # 处理一维方程（通常只包含 x_1）
            print("处理一维方程类型...")
            
            idx = 0
            rows = {'equation': []}  # 存储生成的方程字符串
            
            while idx < len(skeleton_list):
                print(f"正在处理第 {idx+1}/{len(skeleton_list)} 个骨架...")
                
                sample_time = 0      # 成功采样次数
                repeat_time = 0      # 连续失败次数
                
                # 为当前骨架生成指定数量的方程变体
                while sample_time < const.num_sample:
                    skeleton = skeleton_list[idx]
                    
                    # 采样随机常数值
                    w_const, wout_consts = data_utils.sample_symbolic_constants(fixed_flag, skeleton, const)
                    dict_const = w_const
                    
                    # 将常数值填入骨架，生成完整方程
                    eq_string = skeleton.expr.format(**dict_const)
                    
                    # 用 sympy 简化表达式
                    eq_string = str(sp.simplify(eq_string))
                    
                    # 验证方程的有效性
                    # zoo: 表示无穷大，I: 表示虚数单位
                    is_invalid = (
                        (eq_string.find("zoo") != -1) or      # 包含无穷大
                        (eq_string.find("I") != -1) or        # 包含虚数
                        (eq_string in rows["equation"])       # 已存在的重复方程
                    )
                    
                    if is_invalid:
                        repeat_time = repeat_time + 1
                        # 如果连续失败次数达到上限，跳过这次采样
                        if repeat_time == const.repeat:
                            sample_time = sample_time + 1
                            repeat_time = 0
                            print(f"  跳过第 {sample_time} 个样本（连续失败 {const.repeat} 次）")
                    else:
                        # 成功生成有效方程
                        sample_time = sample_time + 1
                        repeat_time = 0
                        rows["equation"].append(str(eq_string))
                        if sample_time % 50 == 0:
                            print(f"  已生成 {sample_time}/{const.num_sample} 个有效方程")
                
                idx = idx + 1
                
        else:
            # 处理多维方程（通常包含 x_1, x_2, ...）
            print("处理多维方程类型...")
            
            idx = 0
            rows = {'equation': []}  # 存储生成的方程字符串
            
            while idx < len(skeleton_list):
                print(f"正在处理第 {idx+1}/{len(skeleton_list)} 个骨架...")
                
                sample_time = 0      # 成功采样次数
                repeat_time = 0      # 连续失败次数
                
                # 为当前骨架生成指定数量的方程变体
                while sample_time < const.num_sample:
                    skeleton = skeleton_list[idx]
                    
                    # 采样随机常数值
                    w_const, wout_consts = data_utils.sample_symbolic_constants(fixed_flag, skeleton, const)
                    dict_const = w_const
                    
                    # 将常数值填入骨架，生成完整方程
                    eq_string = skeleton.expr.format(**dict_const)
                    
                    # 用 sympy 简化表达式
                    eq_string = str(sp.simplify(eq_string))
                    
                    # 验证方程的有效性（多维方程需要包含 x_2）
                    is_invalid = (
                        (eq_string.find("x_2") == -1) or      # 缺少 x_2 变量
                        (eq_string.find("zoo") != -1) or      # 包含无穷大
                        (eq_string.find("I") != -1) or        # 包含虚数
                        (eq_string in rows["equation"])       # 已存在的重复方程
                    )
                    
                    if is_invalid:
                        repeat_time = repeat_time + 1
                        # 如果连续失败次数达到上限，跳过这次采样
                        if repeat_time == const.repeat:
                            sample_time = sample_time + 1
                            repeat_time = 0
                            print(f"  跳过第 {sample_time} 个样本（连续失败 {const.repeat} 次）")
                    else:
                        # 成功生成有效方程
                        sample_time = sample_time + 1
                        repeat_time = 0
                        rows["equation"].append(str(eq_string))
                        print(f"  成功生成方程: {eq_string}")
                        if sample_time % 50 == 0:
                            print(f"  已生成 {sample_time}/{const.num_sample} 个有效方程")
                
                idx = idx + 1
        
        print(f"\n方程生成完成！总共生成了 {len(rows['equation'])} 个有效方程")
        return pd.DataFrame(rows)
