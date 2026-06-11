"""
神经网络模型集合
主要包含：时间嵌入模型、注意力机制模块、集合学习网络
用途：处理时间序列数据和无序集合数据，适用于网络动力学建模
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============== 时间嵌入基础函数 ==============
def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    """
    Time2Vec核心函数：将时间信息转换为向量表示
    
    参数:
        tau: 时间输入
        f: 激活函数（sin或cos）
        out_features: 输出特征维度
        w, b: 周期性部分的权重和偏置
        w0, b0: 线性部分的权重和偏置
        arg: 可选的额外参数
    
    返回:
        时间向量表示，结合周期性和线性成分
    """
    # 周期性部分：f(w*tau + b)，捕获时间的周期性模式
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    
    # 线性部分：w0*tau + b0，捕获时间的趋势性变化
    v2 = torch.matmul(tau, w0) + b0
    
    # 拼接周期性和线性部分
    return torch.cat([v1, v2], 1)


# ============== 时间嵌入激活函数 ==============
class SineActivation(nn.Module):
    """
    基于正弦函数的时间激活模块
    用途：捕获时间数据中的周期性模式，适合建模具有周期性的动力学过程
    """
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        
        # 线性部分参数
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))    # 线性权重
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))    # 线性偏置
        
        # 周期性部分参数（正弦）
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))  # 周期权重
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))  # 周期偏置
        
        self.f = torch.sin  # 使用正弦函数作为周期性激活

    def forward(self, tau):
        """前向传播：将时间转换为正弦周期性向量表示"""
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    """
    基于余弦函数的时间激活模块
    用途：与正弦函数互补，提供不同相位的周期性特征
    """
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        
        # 线性部分参数
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))    # 线性权重
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))    # 线性偏置
        
        # 周期性部分参数（余弦）
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))  # 周期权重
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))  # 周期偏置
        
        self.f = torch.cos  # 使用余弦函数作为周期性激活

    def forward(self, tau):
        """前向传播：将时间转换为余弦周期性向量表示"""
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    """
    Time2Vec时间向量化模块
    
    核心思想：将标量时间转换为向量表示，结合线性和周期性成分
    - 线性成分：捕获时间的单调趋势
    - 周期性成分：捕获时间的周期性模式（如昼夜循环、季节变化等）
    
    应用：时间序列预测、动力学系统建模
    """
    def __init__(self, activation, hiddem_dim):
        """
        初始化Time2Vec模块
        
        参数:
            activation: 激活函数类型 ("sin" 或 "cos")
            hiddem_dim: 隐藏层维度，决定时间向量的长度
        """
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, hiddem_dim)     # 使用正弦激活
        elif activation == "cos":
            self.l1 = CosineActivation(1, hiddem_dim)   # 使用余弦激活

    def forward(self, x):
        """
        前向传播：时间标量 → 时间向量
        
        输入: x - 时间标量 [batch_size, 1]
        输出: 时间向量表示 [batch_size, hiddem_dim]
        """
        x = self.l1(x)
        return x


# ============== 注意力机制模块 ==============
class MAB(nn.Module):
    """
    多头注意力块 (Multi-head Attention Block)
    
    核心功能：实现多头注意力机制，允许模型同时关注不同位置的信息
    应用：处理序列数据和集合数据，捕获元素间的复杂关系
    """
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        """
        初始化多头注意力块
        
        参数:
            dim_Q: Query（查询）的输入维度
            dim_K: Key（键）的输入维度  
            dim_V: Value（值）的输出维度
            num_heads: 注意力头的数量
            ln: 是否使用层归一化
        """
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        
        # 线性变换层：将输入映射为Q、K、V
        self.fc_q = nn.Linear(dim_Q, dim_V)  # Query变换
        self.fc_k = nn.Linear(dim_K, dim_V)  # Key变换
        self.fc_v = nn.Linear(dim_K, dim_V)  # Value变换
        
        # 可选的层归一化
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
            
        # 输出投影层
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        """
        前向传播：计算多头注意力
        
        参数:
            Q: 查询矩阵 [batch_size, seq_len_q, dim_Q]
            K: 键矩阵 [batch_size, seq_len_k, dim_K]
        
        返回:
            O: 注意力输出 [batch_size, seq_len_q, dim_V]
        """
        # 线性变换得到Q、K、V
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        # 多头注意力：将特征维度分割为多个头
        dim_split = self.dim_V // self.num_heads
        # print('dim_split=',dim_split, 'K.size=',K.size())  # 调试信息
        
        # 重新组织张量以支持多头计算
        Q_ = torch.cat(Q.split(dim_split, 2), 0)  # [batch*heads, seq_len_q, dim_split]
        K_ = torch.cat(K.split(dim_split, 2), 0)  # [batch*heads, seq_len_k, dim_split]
        V_ = torch.cat(V.split(dim_split, 2), 0)  # [batch*heads, seq_len_k, dim_split]

        # 计算注意力权重：A = softmax(QK^T / sqrt(d))
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        
        # 加权求和：O = Q + A*V（包含残差连接）
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        
        # 第一次层归一化（可选）
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        
        # 前馈网络 + 残差连接
        O = O + F.relu(self.fc_o(O))
        
        # 第二次层归一化（可选）
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        
        return O


class SAB(nn.Module):
    """
    自注意力块 (Self-Attention Block)
    
    核心功能：对输入序列进行自注意力计算，Q、K、V都来自同一输入
    特点：捕获序列内部元素之间的关系和依赖
    """
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        """
        初始化自注意力块
        
        参数:
            dim_in: 输入特征维度
            dim_out: 输出特征维度
            num_heads: 注意力头数量
            ln: 是否使用层归一化
        """
        super(SAB, self).__init__()
        # 使用MAB实现自注意力（Q=K=X）
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        """
        前向传播：自注意力计算
        
        参数:
            X: 输入序列 [batch_size, seq_len, dim_in]
        
        返回:
            自注意力输出 [batch_size, seq_len, dim_out]
        """
        return self.mab(X, X)  # Q=X, K=X


class ISAB(nn.Module):
    """
    诱导集注意力块 (Induced Set Attention Block)
    
    核心思想：使用可学习的诱导点来减少计算复杂度
    - 传统自注意力：O(n²) 复杂度
    - 诱导注意力：O(nm) 复杂度，其中m << n
    
    应用：处理大规模集合数据，保持性能的同时降低计算成本
    """
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        """
        初始化诱导集注意力块
        
        参数:
            dim_in: 输入特征维度
            dim_out: 输出特征维度
            num_heads: 注意力头数量
            num_inds: 诱导点数量（通常远小于输入序列长度）
            ln: 是否使用层归一化
        """
        super(ISAB, self).__init__()
        
        # 可学习的诱导点矩阵
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)  # Xavier初始化
        
        # 两个多头注意力块
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)  # 诱导点 → 输入
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)  # 输入 → 诱导点

    def forward(self, X):
        """
        前向传播：两阶段诱导注意力
        
        参数:
            X: 输入集合 [batch_size, seq_len, dim_in]
        
        返回:
            诱导注意力输出 [batch_size, seq_len, dim_out]
        """
        # 阶段1：诱导点关注输入，得到诱导表示
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        
        # 阶段2：输入关注诱导表示，得到最终输出
        return self.mab1(X, H)


class PMA(nn.Module):
    """
    池化多头注意力 (Pooling by Multi-head Attention)
    
    核心功能：将变长集合压缩为固定长度的表示
    应用：集合分类、集合回归等任务的最后一层
    """
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        """
        初始化池化多头注意力
        
        参数:
            dim: 特征维度
            num_heads: 注意力头数量
            num_seeds: 种子向量数量（决定输出序列长度）
            ln: 是否使用层归一化
        """
        super(PMA, self).__init__()
        
        # 可学习的种子向量（查询向量）
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)  # Xavier初始化
        
        # 多头注意力模块
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        """
        前向传播：注意力池化
        
        参数:
            X: 输入集合 [batch_size, seq_len, dim]
        
        返回:
            池化结果 [batch_size, num_seeds, dim]
        """
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


# ============== 集合学习网络 ==============
class DeepSet(nn.Module):
    """
    深度集合网络 (DeepSet)
    
    核心思想：处理无序集合数据，满足置换不变性
    架构：编码器-聚合-解码器
    - 编码器：对每个元素独立编码
    - 聚合：使用置换不变的聚合函数（如mean）
    - 解码器：基于聚合结果生成输出
    
    应用：点云分类、分子性质预测、网络节点集合建模
    """
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
        """
        初始化深度集合网络
        
        参数:
            dim_input: 输入元素的特征维度
            num_outputs: 输出序列长度
            dim_output: 输出特征维度
            dim_hidden: 隐藏层维度
        """
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        
        # 编码器：对集合中每个元素进行独立编码
        self.enc = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden)
        )
        
        # 解码器：基于聚合特征生成最终输出
        self.dec = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_outputs * dim_output)
        )

    def forward(self, X):
        """
        前向传播：编码-聚合-解码
        
        参数:
            X: 输入集合 [batch_size, set_size, dim_input]
        
        返回:
            输出 [batch_size, num_outputs, dim_output]
        """
        # 编码：对每个元素独立编码
        X = self.enc(X)  # [batch_size, set_size, dim_hidden]
        
        # 聚合：使用均值池化（置换不变）
        X = X.mean(-2)  # [batch_size, dim_hidden]
        
        # 解码：生成最终输出
        X = self.dec(X)  # [batch_size, num_outputs * dim_output]
        
        # 重塑为所需输出形状
        X = X.reshape(-1, self.num_outputs, self.dim_output)
        return X


class SetTransformer(nn.Module):
    """
    集合Transformer (Set Transformer)
    
    核心特点：结合Transformer注意力机制和集合处理能力
    - 使用ISAB降低计算复杂度
    - 保持置换不变性和置换等变性
    - 强大的表示学习能力
    
    优势：
    - 比DeepSet表达能力更强
    - 比标准Transformer计算更高效
    - 适合处理大规模集合数据
    
    应用：复杂集合建模、图神经网络、网络动力学预测
    """
    def __init__(self, dim_input, num_outputs, dim_output,
                 num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        """
        初始化集合Transformer
        
        参数:
            dim_input: 输入特征维度
            num_outputs: 输出序列长度
            dim_output: 输出特征维度
            num_inds: 诱导点数量（控制计算复杂度）
            dim_hidden: 隐藏层维度
            num_heads: 注意力头数量
            ln: 是否使用层归一化
        """
        super(SetTransformer, self).__init__()
        
        # 编码器：堆叠的诱导集注意力块
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        )

        # 解码器：池化注意力 + 自注意力 + 线性输出
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),  # 池化为固定长度
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),   # 自注意力精化
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),   # 自注意力精化
            nn.Linear(dim_hidden, dim_output)                # 线性输出层
        )

    def forward(self, X):
        """
        前向传播：编码-解码
        
        参数:
            X: 输入集合 [batch_size, set_size, dim_input]
        
        返回:
            输出 [batch_size, num_outputs, dim_output]
        """
        return self.dec(self.enc(X))
