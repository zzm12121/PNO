import os
import time
from copy import copy
import matplotlib.pyplot as plt

import hydra
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

import torch
import torch.nn as nn
from torch_scatter import scatter_sum

import adamod

from data_create.NetworkSystemInstances_new import GeneralDynamics
from data_create.lib.Topo import Topo
from data_create.lib.InitCondition import InitCondition
from string_create.Creation import Creation
from nn_models import *

from generate_data import GenerateData

###
if torch.cuda.is_available():
    device = torch.device("cuda:2")
else:
    device = torch.device("cpu")
# device = torch.device("cpu")
print("using device : ", device)


# ---------------------------------------------------------------------------------------
#
#     IEEE 754 float encoding
#
#     16 bit: num_e_bits=5, num_m_bits=10 | 32 bit: num_e_bits=8, num_m_bits=23
#
# ---------------------------------------------------------------------------------------
def float2bit(f, num_e_bits=8, num_m_bits=23, bias=127., dtype=torch.float32):
    ## f: [b,n,d]
    ## return [b,n,d,32]
    ## SIGN BIT
    s = (torch.sign(f + 0.001) * -1 + 1) * 0.5  # Swap plus and minus => 0 is plus and 1 is minus
    s = s.unsqueeze(-1)
    f1 = torch.abs(f)
    ## EXPONENT BIT
    e_scientific = torch.floor(torch.log2(f1))
    e_scientific[e_scientific == float("-inf")] = -(2 ** (num_e_bits - 1) - 1)
    e_scientific[torch.isnan(e_scientific) == True] = -(2 ** (num_e_bits - 1) - 1)

    e_decimal = e_scientific + (2 ** (num_e_bits - 1) - 1)
    e = integer2bit(e_decimal, num_bits=num_e_bits)
    ## MANTISSA
    f2 = f1 / 2 ** (e_scientific)
    m2 = remainder2bit(f2 % 1, num_bits=bias)
    fin_m = m2[:, :, :, :num_m_bits]  # [:,:,:,8:num_m_bits+8]
    return torch.cat([s, e, fin_m], dim=-1).type(dtype)


def remainder2bit(remainder, num_bits=127):
    dtype = remainder.type()
    exponent_bits = torch.arange(num_bits).type(dtype)
    exponent_bits = exponent_bits.repeat(remainder.shape + (1,))
    out = (remainder.unsqueeze(-1) * 2 ** exponent_bits) % 1
    return torch.floor(2 * out)


def integer2bit(integer, num_bits=8):
    dtype = integer.type()
    exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
    exponent_bits = exponent_bits.repeat(integer.shape + (1,))
    out = integer.unsqueeze(-1) / 2 ** exponent_bits
    return (out - (out % 1)) % 2


def move_list_to_device(a_list, device):
    return [item.to(device) for item in a_list]


class Encoder(nn.Module):
    def __init__(
            self,
            precision,
            hidden_dim,
            max_dim=5,
    ):
        super().__init__()

        self.name = 'Encoder'
        self.hidden_dim = hidden_dim
        self.precision = precision

        # encoders
        # self.encode_t = nn_models.Time2Vec('sin', hidden_dim)
        self.encode_t_1 = nn.Sequential(
            nn.Linear(precision, hidden_dim),
        )
        self.encode_t_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.encode_x_i_1_enc_multidim = nn.ModuleList([])
        for i in range(max_dim):
            self.encode_x_i_1_enc_multidim.append(
                nn.Sequential(
                    nn.Linear(int((i + 1) * precision), hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
        self.encode_x_i_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.encode_x_j_set_phi_1_enc_multidim = nn.ModuleList([])
        for i in range(max_dim):
            self.encode_x_j_set_phi_1_enc_multidim.append(
                nn.Sequential(
                    nn.Linear(int((i + 1) * precision), hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
        self.encode_x_j_set_phi_2 = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.encode_x_j_set_rho_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.encode_x_j_set_rho_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.encode_point_embedding_1 = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
        )
        self.encode_point_embedding_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.encode_set_transformer = SetTransformer(hidden_dim, 10, hidden_dim,
                                                               num_inds=(hidden_dim) // 8, dim_hidden=hidden_dim,
                                                               num_heads=8, ln=True)

        self.encode_z_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.encode_z_logsigma = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x_context, state_dim):
        # t_context: [b, #points, 1]
        # i_context: [b, #points, 1]
        # x_i_context: [b, #points, 1]
        # x_j_set_context: [b, m, state_dim], where m = 1 + #neighbors of node i
        # points_info_context: [m, ], where m = 1 + #neighbors of node i
        t_context, i_context, x_i_context, x_j_set_context, points_info_context = x_context

        # ----------- encoding process ---------------
        # [b, #points, d]
        t_context_encoded = self.encode_t_1(t_context)
        t_context_encoded = self.encode_t_2(t_context_encoded) + t_context_encoded
        # [b, #points, d]
        x_i_context_encoded = self.encode_x_i_1_enc_multidim[state_dim - 1](x_i_context)
        x_i_context_encoded = self.encode_x_i_2(x_i_context_encoded) + x_i_context_encoded
        # [b, m, d]
        # print(x_j_set_context.size())
        x_j_set_context_encoded = self.encode_x_j_set_phi_1_enc_multidim[state_dim - 1](x_j_set_context)

        x_j_set_context_encoded_ = torch.cat(
            [x_i_context_encoded[:, points_info_context.long(), :], x_j_set_context_encoded], dim=-1)

        x_j_set_context_encoded = self.encode_x_j_set_phi_2(x_j_set_context_encoded_) + x_j_set_context_encoded
        # [b, #points, d]
        x_j_set_context_encoded = scatter_sum(x_j_set_context_encoded, points_info_context.long(), dim=1,
                                              dim_size=x_i_context_encoded.size(1))
        # [b, #points, d]
        x_j_set_context_encoded = self.encode_x_j_set_rho_1(x_j_set_context_encoded) + x_j_set_context_encoded
        x_j_set_context_encoded = self.encode_x_j_set_rho_2(x_j_set_context_encoded) + x_j_set_context_encoded

        print(t_context_encoded.size(), x_i_context_encoded.size(), x_j_set_context_encoded.size())
        print(torch.max(points_info_context) + 1)
        # [b, #points, d + d+ d]
        point_embeddings_encoded = torch.cat([t_context_encoded, x_i_context_encoded, x_j_set_context_encoded], dim=-1)
        # [b, #points, d]
        point_embeddings_encoded = self.encode_point_embedding_1(point_embeddings_encoded)
        point_embeddings_encoded = self.encode_point_embedding_2(point_embeddings_encoded) + point_embeddings_encoded

        # [b, 1, d]
        z_encoded = self.encode_set_transformer(point_embeddings_encoded)
        # [b, 1, d]
        z_encoded = torch.mean(z_encoded, dim=1, keepdim=True)

        print('isnan',
              torch.isnan(torch.cat([t_context_encoded, x_i_context_encoded, x_j_set_context_encoded], dim=-1)).sum(),
              torch.isnan(point_embeddings_encoded).sum(), torch.isnan(z_encoded).sum())

        # [b, 1, d]
        z_mean = self.encode_z_mean(z_encoded)
        z_logsigma = self.encode_z_logsigma(z_encoded)
        z_logsigma = 0.1 + 0.9 * torch.sigmoid(z_logsigma)

        # [b, 1, d]
        return torch.distributions.Normal(z_mean, z_logsigma)


class CNP(nn.Module):
    def __init__(
            self,
            precision,
            hidden_dim,
            encoder=None,
            type='self',
            max_dim=5,
    ):
        super().__init__()

        self.name = 'CNP'
        self.hidden_dim = hidden_dim
        self.precision = precision
        
        self.type = type

        # encoders
        if encoder is None:
            self.encoder = Encoder(precision, hidden_dim, max_dim)
        else:
            self.encoder = encoder

        # decoders
        if type == 'self':
            self.decode_encode_x_multidim = nn.ModuleList([])
            for i in range(max_dim):
                self.decode_encode_x_multidim.append(
                    nn.Sequential(
                        nn.Linear(int(i + 1), hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.LeakyReLU(inplace=True),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                )
                
            #self.decode_encode_x_multidim_w = nn.ModuleList([])
            #for i in range(max_dim):
            #    self.decode_encode_x_multidim_w.append(
            #        nn.Sequential(
            #            nn.Linear(hidden_dim, hidden_dim),
            #            nn.LeakyReLU(inplace=True),
            #            nn.Linear(hidden_dim, (i + 1)*hidden_dim),
            #        )
            #)
            
        else:
            self.decode_encode_x_multidim = nn.ModuleList([])
            for i in range(max_dim):
                self.decode_encode_x_multidim.append(
                    nn.Sequential(
                        nn.Linear(int((i + 1) * 2), hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.LeakyReLU(inplace=True),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                )
            
            #self.decode_encode_x_multidim_w = nn.ModuleList([])
            #for i in range(max_dim):
            #    self.decode_encode_x_multidim_w.append(
            #        nn.Sequential(
            #            nn.Linear(hidden_dim, hidden_dim),
            #            nn.LeakyReLU(inplace=True),
            #            nn.Linear(hidden_dim, int((i + 1) * 2)*hidden_dim),
            #        )
            #)

        # self.decode_f = nn.Sequential(
        #     nn.Linear(hidden_dim + hidden_dim, hidden_dim),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(hidden_dim, 1),
        # )

        if self.type == 'self':
            input_dim = hidden_dim
        else:
            input_dim = hidden_dim + hidden_dim
            
        self.decode_f_non_lin_layer_1_w = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim * hidden_dim // 2)),
            nn.LeakyReLU(inplace=True),
            nn.Linear(int(hidden_dim * hidden_dim // 2), hidden_dim * hidden_dim),
        )
        
        self.decode_f_non_lin_layer_1_w_emb = nn.Sequential(
            nn.Linear(hidden_dim * hidden_dim, hidden_dim, bias=False),
        )
        
        self.decode_f_non_lin_layer_1_b = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.decode_f_non_lin_layer_1_b_emb = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        
        self.decode_f_non_lin_layer_1_act = nn.LeakyReLU(inplace=True)

        self.decode_f_non_lin_layer_2_w = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim),
        )
               
        self.decode_f_non_lin_layer_2_w_emb = nn.Sequential(
            nn.Linear(hidden_dim * hidden_dim, hidden_dim, bias=False),
        )

        self.decode_f_lin_layer_1_w = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim),
        )

        self.decode_f_lin_layer_1_w_emb = nn.Sequential(
            nn.Linear(hidden_dim * hidden_dim, hidden_dim, bias=False),
        )
        
        self.decode_f_lin_layer_1_b = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.decode_f_lin_layer_1_b_emb = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        
        self.decode_f_weights_emb = nn.Sequential(
            nn.Linear(hidden_dim*5, hidden_dim, bias=False),
        )
        
        
        # noise
        self.decode_f_non_lin_layer_1_w_noise = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * hidden_dim // 2)),
            nn.LeakyReLU(inplace=True),
            nn.Linear(int(hidden_dim * hidden_dim // 2), hidden_dim * hidden_dim),
        )
        self.decode_f_non_lin_layer_1_b_noise = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decode_f_non_lin_layer_1_act_noise = nn.LeakyReLU(inplace=True)

        self.decode_f_non_lin_layer_2_w_noise = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim),
        )
        self.decode_f_non_lin_layer_2_b_noise = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        #

        self.decode_decode_x_mean_multidim_w = nn.ModuleList([])
        for i in range(max_dim):
            self.decode_decode_x_mean_multidim_w.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim * (i + 1)),
                )
            )
        self.decode_decode_x_mean_multidim_b = nn.ModuleList([])
        for i in range(max_dim):
            self.decode_decode_x_mean_multidim_b.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(hidden_dim, int(i + 1)),
                )
            )
        
        self.decode_decode_x_noise_multidim_w = nn.ModuleList([])
        for i in range(max_dim):
            self.decode_decode_x_noise_multidim_w.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim * (i + 1)),
                )
            )
        self.decode_decode_x_noise_multidim_b = nn.ModuleList([])
        for i in range(max_dim):
            self.decode_decode_x_noise_multidim_b.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(hidden_dim, int(i + 1)),
                )
            )
            
        ## save the weights
        self.saved_weights = None
        

    def encode(self, x_context, state_dim, num_sampling=1):
        # ----------- encoding process ---------------
        # [b, 1, d]
        z_dist = self.encoder(x_context, state_dim)
        # print(num_sampling)
        z_sampling = z_dist.rsample([num_sampling])  # # [num_sampling, b, 1, d]
        # z_sampling = z_dist.loc.unsqueeze(0).repeat(num_sampling, 1, 1, 1)  # # [num_sampling, b, 1, d]

        return z_sampling, z_dist

    def decode(self, z_sampling, x_target, state_dim, mu_std_target=None, num_sampling=1, weights_emb=None):
        # ----------- decoding process ---------------
        # x_target_in: [b, ?, state_dim]
        # x_target_out: [b, ?, 1]
        x_target_in, x_target_out = x_target
        # [b, ?, d] -> [num_sampling, b, ?, d]
        
        x_target_in = x_target_in.unsqueeze(0).repeat(num_sampling, 1, 1, 1)
        
        _, b, n, _ = x_target_in.size()
        
        #encode_x_w = self.decode_encode_x_multidim_w[state_dim - 1](z_sampling).squeeze(2).view(num_sampling,
        #                                                                                                  b,
        #                                                                                                  -1,
        #                                                                                                  self.hidden_dim)
        #                                                                                                  
        #x_target_in_encoded = torch.matmul(
        #    x_target_in.view(num_sampling * b, n, -1), encode_x_w.view(num_sampling * b, -1, self.hidden_dim)
        #)
        
        x_target_in_encoded = self.decode_encode_x_multidim[state_dim - 1](x_target_in)
            
        x_target_in_encoded = x_target_in_encoded.view(num_sampling, b, -1, self.hidden_dim)

        _, b, n, d = x_target_in_encoded.size()
        
        if weights_emb is None:
            input_z_sampling_weights_emb = z_sampling
        else:
            input_z_sampling_weights_emb = torch.cat([z_sampling, weights_emb], dim=-1)
        print('input_z_sampling_weights_emb.size=',input_z_sampling_weights_emb.size(), self.type, weights_emb is None)

        # [num_sampling, b, 1, d] -> [num_sampling, b, d, d]
        w_1 = self.decode_f_non_lin_layer_1_w(input_z_sampling_weights_emb).squeeze(2).view(num_sampling, -1, self.hidden_dim,
                                                                          self.hidden_dim)
        # [num_sampling, b, 1, d] -> [num_sampling, b, 1, d]
        b_1 = self.decode_f_non_lin_layer_1_b(input_z_sampling_weights_emb).view(num_sampling, -1, 1, self.hidden_dim)

        # [num_sampling, b, 1, d] -> [num_sampling, b, d, 1]
        w_2 = self.decode_f_non_lin_layer_2_w(input_z_sampling_weights_emb).squeeze(2).view(num_sampling, -1, self.hidden_dim,
                                                                          self.hidden_dim)
        # [num_sampling, b, 1, d] -> [num_sampling, b, d, d]
        w_3 = self.decode_f_lin_layer_1_w(input_z_sampling_weights_emb).squeeze(2).view(num_sampling, -1,
                                                                      self.hidden_dim,
                                                                      self.hidden_dim)
        # [num_sampling, b, 1, d] -> [num_sampling, b, d, 1]
        b_4 = self.decode_f_lin_layer_1_b(input_z_sampling_weights_emb).view(num_sampling, -1,
                                                           1,
                                                           self.hidden_dim)
        
        
        
        # num_sampling*b, n, d
        non_lin_1 = self.decode_f_non_lin_layer_1_act(
            torch.matmul(x_target_in_encoded.view(num_sampling * b, n, d), w_1.view(num_sampling * b, d, d)) \
            + b_1.view(num_sampling * b, 1, d))
        non_lin_out = torch.matmul(non_lin_1.view(num_sampling * b, n, d), w_2.view(num_sampling * b, d, -1))

        lin_out = torch.matmul(x_target_in_encoded.view(num_sampling * b, n, d), w_3.view(num_sampling * b, d, -1))

        pre_x_target_out_mean = non_lin_out + lin_out + b_4.view(num_sampling * b, 1, -1)
        
        
        # embedding for generated weights
        # [num_sampling, b, 1, d]
        w_1_emb = self.decode_f_non_lin_layer_1_w_emb(w_1.view(num_sampling, -1, 1, self.hidden_dim*self.hidden_dim))
        b_1_emb = self.decode_f_non_lin_layer_1_b_emb(b_1.view(num_sampling, -1, 1, self.hidden_dim))
        w_2_emb = self.decode_f_non_lin_layer_2_w_emb(w_2.view(num_sampling, -1, 1, self.hidden_dim*self.hidden_dim))
        w_3_emb = self.decode_f_lin_layer_1_w_emb(w_3.view(num_sampling, -1, 1, self.hidden_dim*self.hidden_dim))
        b_4_emb = self.decode_f_lin_layer_1_b_emb(b_4.view(num_sampling, -1, 1, self.hidden_dim))
        weights_emb_ = self.decode_f_weights_emb(torch.cat([w_1_emb, b_1_emb, w_2_emb, w_3_emb, b_4_emb], dim=-1))

        
        # [num_sampling, b, 1, d] -> [num_sampling, b, d, d]
        w_1_noise = self.decode_f_non_lin_layer_1_w_noise(z_sampling).squeeze(2).view(num_sampling, -1, self.hidden_dim,
                                                                                      self.hidden_dim)
        # [num_sampling, b, 1, d] -> [num_sampling, b, 1, d]
        b_1_noise = self.decode_f_non_lin_layer_1_b_noise(z_sampling).view(num_sampling, -1, 1, self.hidden_dim)

        # [num_sampling, b, 1, d] -> [num_sampling, b, d, 1]
        w_2_noise = self.decode_f_non_lin_layer_2_w_noise(z_sampling).squeeze(2).view(num_sampling, -1, self.hidden_dim,
                                                                                      self.hidden_dim)
        b_2_noise = self.decode_f_non_lin_layer_2_b_noise(z_sampling).squeeze(2).view(num_sampling, -1, 1,
                                                                                      self.hidden_dim)
        # num_sampling*b, n, d
        non_lin_1_noise = self.decode_f_non_lin_layer_1_act_noise(
            torch.matmul(x_target_in_encoded.view(num_sampling * b, n, d), w_1_noise.view(num_sampling * b, d, d)) \
            + b_1_noise.view(num_sampling * b, 1, d))
        pre_x_target_out_std = torch.matmul(non_lin_1_noise.view(num_sampling * b, n, d),
                                            w_2_noise.view(num_sampling * b, d, -1)) + b_2_noise.view(num_sampling * b,
                                                                                                      1, -1)
        
        # print('in model',pre_x_target_out_mean.size())
        # print('in model',pre_x_target_out_std.size())

        decode_x_mean_w = self.decode_decode_x_mean_multidim_w[state_dim - 1](z_sampling).squeeze(2).view(num_sampling,
                                                                                                          -1,
                                                                                                          self.hidden_dim,
                                                                                                          state_dim)
        decode_x_mean_b = self.decode_decode_x_mean_multidim_b[state_dim - 1](z_sampling).squeeze(2).view(num_sampling,
                                                                                                          -1,
                                                                                                          1,
                                                                                                          state_dim)
        pre_x_target_out_mean = torch.matmul(
            pre_x_target_out_mean.view(num_sampling * b, n, d), decode_x_mean_w.view(num_sampling * b, d, -1)
        ) + decode_x_mean_b.view(num_sampling * b, 1, -1)
        
        
        decode_x_noise_w = self.decode_decode_x_noise_multidim_w[state_dim - 1](z_sampling).squeeze(2).view(
            num_sampling, -1,
            self.hidden_dim,
            state_dim)
        decode_x_noise_b = self.decode_decode_x_noise_multidim_b[state_dim - 1](z_sampling).squeeze(2).view(
            num_sampling, -1,
            1,
            state_dim)
        pre_x_target_out_std = torch.matmul(
            pre_x_target_out_std.view(num_sampling * b, n, d), decode_x_noise_w.view(num_sampling * b, d, -1)
        ) + decode_x_noise_b.view(num_sampling * b, 1, -1)

        # pre_x_target_out_mean = self.decode_decode_x_mean_multidim[state_dim - 1](pre_x_target_out_mean)
        # pre_x_target_out_std = self.decode_decode_x_noise_multidim[state_dim - 1](pre_x_target_out_std)
        
        pre_x_target_out_mean = pre_x_target_out_mean.view(num_sampling, b, -1, state_dim)
        pre_x_target_out_std = pre_x_target_out_std.view(num_sampling, b, -1, state_dim)
        
        #pre_x_target_out_std = torch.ones_like(pre_x_target_out_mean)

        # Bound the variance
        pre_x_target_out_std = 0.01 + 0.99 * torch.nn.functional.softplus(pre_x_target_out_std)

        # print(torch.isnan(pre_x_target_out).any(), torch.isinf(pre_x_target_out).any())
        
        
        self.saved_weights = [w_1.detach(), b_1.detach(), w_2.detach(), w_3.detach(), b_4.detach(), decode_x_mean_w.detach(), decode_x_mean_b.detach()]
        
        
        if mu_std_target is not None:
            pre_x_target_out_mean = (pre_x_target_out_mean - mu_std_target[0].unsqueeze(0).repeat(num_sampling, 1, 1,
                                                                                                   1)) / mu_std_target[
                                         1].unsqueeze(0).repeat(num_sampling, 1, 1,
                                                                1)  # normalize predictions for training
            pre_x_target_out_std = pre_x_target_out_std / mu_std_target[1].unsqueeze(0).repeat(num_sampling, 1, 1,
                                                                                                1)  # normalize predictions for training
            #pre_x_target_out_mean = (pre_x_target_out_mean - mu_std_target[1].unsqueeze(0).repeat(num_sampling, 1, 1,
            #                                                                                      1)) / (mu_std_target[
            #                                                                                                 0].unsqueeze(
            #    0).repeat(num_sampling, 1, 1,
            #              1) - mu_std_target[
            #                                                                                                 1].unsqueeze(
            #    0).repeat(num_sampling, 1, 1,
            #              1))  # normalize predictions for training
            #pre_x_target_out_std = pre_x_target_out_std / (mu_std_target[
            #                                                   0].unsqueeze(0).repeat(num_sampling, 1, 1,
            #                                                                          1) - mu_std_target[
            #                                                   1].unsqueeze(0).repeat(num_sampling, 1, 1,
            #                                                                          1))  # normalize predictions for training
        
        # [num_sampling, b, ?, 1]
        #poster_dist = torch.distributions.Normal(pre_x_target_out_mean, pre_x_target_out_std)
        poster_dist = torch.distributions.Normal(pre_x_target_out_mean, pre_x_target_out_std)

        return poster_dist, weights_emb_

    def forward(self, x_context, x_target, state_dim, mu_std_target=None, num_sampling=1, weights_emb=None):
        # ----------- encoding process ---------------
        z_sampling, z_dist = self.encode(x_context, state_dim, num_sampling=num_sampling)  # # [num_sampling, b, 1, d]

        # ----------- decoding process ---------------
        # x_target_in: [b, ?, state_dim]
        # x_target_out: [b, ?, 1]
        x_target_in, x_target_out = x_target

        poster_dist, weights_emb_ = self.decode(z_sampling, x_target, state_dim, mu_std_target=mu_std_target,
                                  num_sampling=num_sampling, weights_emb=weights_emb)

        loss = {}
        if x_target_out is not None:
            if mu_std_target is not None:
                x_target_out_ = (x_target_out - mu_std_target[0]) / mu_std_target[1]  # normalize x_target_out for training
                #x_target_out_ = (x_target_out - mu_std_target[1]) / (
                #        mu_std_target[0] - mu_std_target[1])  # normalize x_target_out for training
            else:
                x_target_out_ = x_target_out
            # print(mu_std_target[0], mu_std_target[1])
            print(torch.any(torch.isnan(x_target_out_)))
            # [num_sampling, b, ?, 1]
            log_p = poster_dist.log_prob(x_target_out_.unsqueeze(0).repeat(num_sampling, 1, 1, 1))
            # [num_sampling, b, ?]
            log_p = log_p.sum(-1)
            # [num_sampling, b,]
            log_p = log_p.sum(-1)
            # [b,]
            log_p = log_p.mean(0).view(-1)

            z_prior = torch.distributions.Normal(torch.zeros_like(z_dist.loc), torch.ones_like(z_dist.loc))
            # [b, 1]
            loss_kl = torch.distributions.kl_divergence(z_dist, z_prior).sum(-1).view(-1)
            # loss_kl = torch.Tensor([0.])

            loss_total = (-log_p + loss_kl).mean()
            #loss_total = (torch.log10(-log_p + loss_kl)).mean()
            # loss_total = -log_p.mean()

            loss['loss'] = loss_total
            loss['kl'] = loss_kl.mean()
            loss['neg_log_p'] = -log_p.mean()

            # print(torch.isnan(log_p).any(), torch.isinf(log_p).any())
        return {'pre_dist': poster_dist, 'loss': loss, 'weights_emb': weights_emb_}


class CNPFoundationModel(nn.Module):
    def __init__(
            self,
            precision=32,
            hidden_dim=512,
            max_dim=5,
    ):
        super().__init__()

        self.name = 'CNPFoundationModel'
        self.hidden_dim = hidden_dim
        self.precision = precision  # float32
        
        #encoder = Encoder(precision, hidden_dim, max_dim)
        encoder = None

        self.CNP_self = CNP(precision, hidden_dim, encoder=encoder, type='self', max_dim=max_dim)
        self.CNP_interaction = CNP(precision, hidden_dim, encoder=encoder, type='interaction', max_dim=max_dim)

    def forward(self, x_context, x_target_self, x_target_interaction, x_target_total, points_info, state_dim,
                mu_std_target=None):
        #res_self = self.CNP_self(x_context, x_target_self, state_dim, (mu_std_target[0][0], mu_std_target[1][0]))
        #res_interaction = self.CNP_interaction(x_context, x_target_interaction, state_dim,
        #                                       (mu_std_target[0][1], mu_std_target[1][1]))
        if mu_std_target is None:
            res_self = self.CNP_self(x_context, x_target_self, state_dim, None, weights_emb=None)
            res_interaction = self.CNP_interaction(x_context, x_target_interaction, state_dim, None, weights_emb=res_self['weights_emb'])
        else:
            res_self = self.CNP_self(x_context, x_target_self, state_dim, (mu_std_target[0][0], mu_std_target[1][0]), weights_emb=None)
            res_interaction = self.CNP_interaction(x_context, x_target_interaction, state_dim, (mu_std_target[0][1], mu_std_target[1][1]), weights_emb=res_self['weights_emb'])
        
        loss_1 = res_self['loss']['loss']
        loss_2 = res_interaction['loss']['loss']

        # if mu_std_target is None:
        #     x_target_total_ = x_target_total
        # else:  # normalize x_target_out for training
        #     x_target_total_ = (x_target_self[1] - mu_std_target[0]) / mu_std_target[1] \
        #                                + scatter_sum((x_target_interaction[1] - mu_std_target[0]) / mu_std_target[1],
        #                                               points_info.long(),
        #                                               dim=1, dim_size=x_target_self[1].size(1))
        # [num_sampling, b, ?, 1] -> [b, ?, 1]                           
        # loss_3 = torch.mean(torch.abs(res_self['pre_dist'].loc.mean(0) + scatter_sum(res_interaction['pre_dist'].loc.mean(0),
        #                                                        points_info.long(),
        #                                                        dim=1, 
        #                                                        dim_size=x_target_self[1].size(1)) - x_target_total_), dim=1).mean()
        # [num_sampling, b, ?, 1]
        # mu_3 = res_self['pre_dist'].loc + scatter_sum(res_interaction['pre_dist'].loc,
        #                                                         points_info.long(),
        #                                                         dim=2,
        #                                                         dim_size=x_target_self[1].size(1))
        # sum_mu_i_2 = res_self['pre_dist'].loc**2 + scatter_sum(res_interaction['pre_dist'].loc**2,
        #                                                         points_info.long(),
        #                                                         dim=2,
        #                                                         dim_size=x_target_self[1].size(1))
        # var_3 = res_self['pre_dist'].scale**2 + scatter_sum(res_interaction['pre_dist'].scale**2,
        #                                                         points_info.long(),
        #                                                         dim=2,
        #                                                         dim_size=x_target_self[1].size(1)) #+ sum_mu_i_2 - mu_3**2
        # dist_3 = torch.distributions.Normal(mu_3, torch.sqrt(var_3))
        # # [num_sampling, b, ?, 1]
        # loss_3 = dist_3.log_prob(x_target_total_.unsqueeze(0).repeat(mu_3.size(0), 1, 1, 1))
        # # [num_sampling, b, 1]
        # loss_3 = loss_3.sum(2)
        # # [b, 1]
        # loss_3 = -loss_3.mean(0).mean()

        loss = loss_1 + loss_2  # + loss_3
        # loss = loss_2

        loss_details = {
            'loss': loss,
            'loss_1': {'loss_1': res_self['loss']['loss'].item(), 'kl': res_self['loss']['kl'].item(),
                       'neg_log_p': res_self['loss']['neg_log_p'].item()},
            'loss_2': {'loss_2': res_interaction['loss']['loss'].item(), 'kl': res_interaction['loss']['kl'].item(),
                       'neg_log_p': res_interaction['loss']['neg_log_p'].item()},
            # 'loss_3': loss_3.item(),
            'loss_3': 0.,
        }

        return loss_details


class TrainCNPFoundationModel:
    def __init__(
            self,
            cfg,
            model_path=None
    ):
        # build model
        self.max_dim = 5
        # self.max_dim = self.cfg.dimension
        self.model = CNPFoundationModel(precision=32, hidden_dim=128, max_dim=self.max_dim)

        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=device))

        # if torch.cuda.device_count() > 1:  # 检查电脑是否有多块GPU
        #    print(f"Let's use {torch.cuda.device_count()} GPUs!")
        #    self.model = nn.DataParallel(self.model)  # 将模型对象转变为多GPU并行运算的模型
        self.model = self.model.to(device)

        print(f'Total number of parameters: {self.count_parameters()}')

        # build params for training
        self.params = {'lr': 1e-4, #5e-4,
                       'weight_decay': 0.,
                       'decay_lr': 10.,
                       'batch_size': 32,
                       'num_epochs': 500}

        #self.dataname = 'data/dataset_dim=1_7systems.csv'
        self.dataname = 'data/dataset_5000_test_dim=1.csv'
        # datasets
        #dataset_1 = pd.read_csv('data/dataset_dim=1_5000.csv', header=None,
        #                        names=['no', 'f_eq_str', 'g_eq_str', 'state'])
        dataset_1 = pd.read_csv('data/dataset_dim=1_22000.csv', header=0,
                                names=['no', 'f_eq_str_0', 'g_eq_str_0', 'eq_type', 'state'])
        dataset_2 = pd.read_csv('data/dataset_dim=2_6000.csv', header=0,
                                names=['no', 'f_eq_str_0', 'f_eq_str_1', 'g_eq_str_0', 'g_eq_str_1', 'eq_type', 'state'])
        self.dataset = {1: dataset_1, 2: dataset_2}
        self.dataset_size = {1: len(dataset_1), 2: len(dataset_2)}
        #self.dataset_size = {1: len(dataset_1), }

        # self.dataset = self.dataset[:300]
        self.max_state_dim = 2

        self.cfg = cfg

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # return sum(p.numel() for p in self.model.parameters())

    def shuffle_datasets(self, ):
        for key in self.dataset.keys():
            self.dataset[key] = shuffle(self.dataset[key])

    def make_batch(self, data=None, flag_IEEE754=True):
        #  shuffle
        #self.shuffle_datasets()

        #if data is not None:
        #    data = shuffle(data)

        batch_size = self.params['batch_size']
        lineno = [0] * self.max_state_dim
        # while lineno < len(data):
        while sum(lineno) < sum(list(self.dataset_size.values())):

            start_time = time.time()
            # keep same dim in one batch
            # state_dim = np.random.randint(1, self.max_dim + 1)
            # state_dim = np.random.randint(1, self.max_state_dim + 1)
            
            state_dim = np.random.choice(a=[key for key in list(self.dataset_size.keys()) if lineno[key-1] < self.dataset_size[key]], size=1)[0]
            print(' -- state_dim = %s (lineno = %s, dataset_size = %s)'%(state_dim, lineno, self.dataset_size))
            #state_dim = 2
            # selected_dim = np.random.randint(1, state_dim + 1)  # selected dim for target's output

            # keep same topo in one batch
            N_sampled = np.random.randint(int(self.cfg.topo.max_num / 5), self.cfg.topo.max_num)
            topo_type_sampled = np.random.choice(a=self.cfg.topo.type_list, size=1)[0]
            topo = Topo(N_sampled, topo_type_sampled)
            N_sampled = topo.N

            # N_sampled = data[lineno]['state_data'].size(1)
            # topo_type_sampled = 'grid'
            # topo = Topo(N_sampled, topo_type_sampled)

            # same sampling x_i and same sampling t
            num_sampled_x_i = np.random.randint(1, int(N_sampled / 10))
            # num_sampled_x_i = 5
            sampled_x_i_idxs_ = np.random.choice(a=list(range(N_sampled)), size=num_sampled_x_i,
                                                 replace=False)  # [num_sampled_x_i, ]

            # tal_time_steps = int((self.cfg.t.end - self.cfg.t.start) / self.cfg.t.inc)
            total_time_steps = int((self.cfg.t.end - self.cfg.t.start) / self.cfg.t.inc / 2)
            num_sampled_t_idxs_for_one_x_i = np.random.randint(int(total_time_steps * 0.4), int(total_time_steps * 1))
            #num_sampled_t_idxs_for_one_x_i = total_time_steps
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

            eq_type_batch = []

            num_in_batch = 0
            while num_in_batch < batch_size:

                if data is None:

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
                        
                        
                    # print(eq_type, f_eq_str, g_eq_str)
                    print(eq_type)
                    
                    # add edge weights
                    topo.sparse_adj = torch.cat([topo.sparse_adj[0].view(1, -1), topo.sparse_adj[1].view(1, -1), torch.ones_like(topo.sparse_adj[1]).view(1, -1)], dim=0)


                    # generate data
                    if eq_type == 'Heat' or eq_type == 'Gene' or eq_type == 'Mutualistic' or eq_type == 'Mutualistic2':
                        init_cond = InitCondition(N_sampled, state_dim, lbs=torch.ones(state_dim) * (0.),
                                                  ubs=torch.ones(state_dim) * 25., num_sampling=1,
                                                  constraint=None)
                        t_start = self.cfg.t.start
                        t_inc = self.cfg.t.inc
                        t_end = self.cfg.t.end
                    elif eq_type == 'Neural' or eq_type == 'Excitation':
                        init_cond = InitCondition(N_sampled, state_dim, lbs=torch.ones(state_dim) * (-3.),
                                                  ubs=torch.ones(state_dim) * 3., num_sampling=1,
                                                  constraint=None)
                        t_start = self.cfg.t.start
                        t_inc = self.cfg.t.inc
                        t_end = self.cfg.t.end
                    elif eq_type == 'Kuramoto':
                        init_cond = InitCondition(N_sampled, state_dim, lbs=torch.ones(state_dim) * (-5.),
                                                  ubs=torch.ones(state_dim) * 5., num_sampling=1,
                                                  constraint=None)
                        t_start = self.cfg.t.start
                        t_inc = self.cfg.t.inc
                        t_end = self.cfg.t.end
                    elif eq_type == 'Ecosystems' or eq_type == 'Plant' or eq_type == 'Lotka_Volterra':
                        init_cond = InitCondition(N_sampled, state_dim, lbs=torch.ones(state_dim) * (0.),
                                                  ubs=torch.ones(state_dim) * 10., num_sampling=1,
                                                  constraint=None)
                        t_start = self.cfg.t.start
                        t_inc = self.cfg.t.inc
                        t_end = self.cfg.t.end
                    elif eq_type == 'Epidemic' or eq_type == 'Population':
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
                    elif eq_type == 'FitzHughNagumo':
                        init_cond = InitCondition(N_sampled, state_dim, lbs=torch.ones(state_dim) * (-1.),
                                                  ubs=torch.ones(state_dim) * 1., num_sampling=1,
                                                  constraint=None)
                        t_start = 0.
                        t_inc = 0.5
                        t_end = 50.
                        
                        # add edge weights: 1/degree_i
                        edge_weights = torch.ones_like(topo.sparse_adj[1])
                        edge_weights = scatter_sum(edge_weights.view(-1), topo.sparse_adj[1].long(), dim=0, dim_size=N_sampled)
                        edge_weights = 1.0 / edge_weights.float()
                        edge_weights = edge_weights[topo.sparse_adj[1].long()]
                        topo.sparse_adj = torch.cat([topo.sparse_adj[0].view(1, -1), topo.sparse_adj[1].view(1, -1), edge_weights.view(1, -1)], dim=0)
                        
                    else:
                        print("unknown eq_type [%s]" % eq_type)
                        exit(1)
                    # print(init_cond.sampled_init_condition.sum(-1))
                    
                    s_ns = GeneralDynamics(N_sampled, state_dim, topo_type_sampled, f_eq_str, g_eq_str, topo, init_cond)
                    s_data = s_ns.simulating_data(t_start,
                                                  t_inc,
                                                  t_end,
                                                  self.cfg.resample_init_condition,
                                                  norm_state=True)
                    # print(eq_type, f_eq_str, g_eq_str)
                    if s_data is None:
                        print(f_eq_str, g_eq_str)
                        print('**Simulation failed!!!**')
                        continue

                    num_in_batch += 1
                    eq_type_batch.append(eq_type)

                    # print('**Simulation succeed!!!**[ %s (dataset lineno %s)]**' % (num_in_batch, lineno))
                else:
                    if lineno >= len(data):
                        break
                    s_data = data[lineno]
                    num_in_batch += 1
                    lineno += 1

                # s_ns.display(s_data)

                # exit(1)

                # s_data : {'t': torch.from_numpy(t_range.reshape(-1, 1)),  # [len(t_range),]
                #         'state_data': torch.from_numpy(New_X.reshape(len(t_range), self.N, self.dim)),
                #         # [len(t_range), N, dim]
                #         'total_diff': total_diff_signal,
                #         'self_diff': (self_diff_in, self_diff_out),
                #         'interact_diff': (interact_diff_in, interact_diff_out), # [len(t_range), #edges, dim + dim]
                #         }
                t_context_one = s_data['t'][sampled_t_idxs].view(-1, 1)  # [#points, 1]
                i_context_one = sampled_x_i_idxs.view(-1, 1)  # [#points, 1]
                x_i_context_one = s_data['state_data'][sampled_t_idxs, sampled_x_i_idxs, :].view(-1, state_dim)  # [#points, 1]
                x_j_set_context_one = []
                points_info_context_one = []
                # row, col = topo.sparse_adj
                row, col, weights = s_data['sparse_adj']
                row = row.long()
                col = col.long()
                # add self loop
                row = torch.cat([row, torch.arange(N_sampled)], dim=0)
                col = torch.cat([col, torch.arange(N_sampled)], dim=0)

                for iiii in range(len(sampled_x_i_idxs)):
                    sampled_x_i_idx = sampled_x_i_idxs[iiii]
                    neibors = s_data['state_data'][sampled_t_idxs[iiii], row, :][col == sampled_x_i_idx].view(
                        -1, state_dim)
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
                x_target_self_in_batch.append(
                    s_data['self_diff'][0].view(-1, state_dim).unsqueeze(
                        0))
                x_target_self_out_batch.append(
                    s_data['self_diff'][1][:, :].view(-1, state_dim).unsqueeze(0))
                x_target_interaction_in_batch.append(
                    s_data['interact_diff'][0].view(-1, state_dim + state_dim).unsqueeze(
                        0))
                x_target_interaction_out_batch.append(
                    s_data['interact_diff'][1][:, :].view(-1, state_dim).unsqueeze(
                        0))
                x_target_total_batch.append(
                    s_data['total_diff'][:, :].view(-1, state_dim).unsqueeze(
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

            # print('target in batch', x_target_self_out_batch.size(), x_target_interaction_out_batch.size(),
            #      x_target_total_batch.size())
            print('target in batch', x_target_self_in_batch.mean(), x_target_self_in_batch.std(),
                  x_target_interaction_in_batch.mean(), x_target_interaction_in_batch.std())

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
                          'mu_x_target_self': torch.mean(
                              x_target_self_out_batch.float().view(x_target_self_out_batch.size(0), -1, state_dim),
                              dim=1).view(-1, 1, state_dim),
                          'std_x_target_self': torch.std(
                              x_target_self_out_batch.float().view(x_target_self_out_batch.size(0), -1, state_dim),
                              dim=1).view(-1, 1, state_dim) + 1.,  # 1e-5
                          'mu_x_target_interaction': torch.mean(
                              x_target_interaction_out_batch.float().view(x_target_interaction_out_batch.size(0), -1, state_dim),
                              dim=1).view(-1, 1, state_dim),
                          'std_x_target_interaction': torch.std(
                              x_target_interaction_out_batch.float().view(x_target_interaction_out_batch.size(0), -1, state_dim),
                              dim=1).view(-1, 1, state_dim) + 1.,  # 1e-5
                          'eq_type_batch': eq_type_batch,
                          }

            print("make a batch cost = %.2f" % (time.time() - start_time))
            yield batch_data

    # train
    def train(self, data=None):
        start_time = time.time()

        # optim = torch.optim.AdaMod(
        #    self.model.parameters(),
        #    self.params['lr'],
        #    weight_decay=self.params['weight_decay'])

        optim = adamod.AdaMod(self.model.parameters(), self.params['lr'], betas=(0.9, 0.98), beta3=0.999,
                              weight_decay=self.params['weight_decay'])
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optim,
        #                                                   gamma=float((1. / self.params['decay_lr']) ** (
        #                                                           1. / self.params['num_epochs'])))

        for epoch in range(self.params['num_epochs']):

            # lr = scheduler.get_last_lr()[0]
            lr = self.params['lr']

            print("epoch = " + str(epoch + 1), 'lr = ', str(lr), end='\r\n')

            self.model.train()

            # gen_batch = self.make_batch(data, flag_IEEE754=False)
            gen_batch = self.make_batch(data, flag_IEEE754=True)
            step = 0
            loss_total = 0.
            for batch_data in gen_batch:
                step += 1

                start_forward_time = time.time()

                optim.zero_grad()

                x_context = move_list_to_device(batch_data['x_context'], device)
                x_target_self = move_list_to_device(batch_data['x_target_self'], device)
                x_target_interaction = move_list_to_device(batch_data['x_target_interaction'], device)
                x_target_total = batch_data['x_target_total'].to(device)
                points_info = batch_data['points_info'].to(device)
                state_dim = batch_data['state_dim']
                mu_x_target_self = batch_data['mu_x_target_self'].to(device)
                std_x_target_self = batch_data['std_x_target_self'].to(device)
                mu_x_target_interaction = batch_data['mu_x_target_interaction'].to(device)
                std_x_target_interaction = batch_data['std_x_target_interaction'].to(device)
                #mu_x_target = (
                #    batch_data['max_x_target_self'].to(device), batch_data['max_x_target_interaction'].to(device))
                #std_x_target = (
                #    batch_data['min_x_target_self'].to(device), batch_data['min_x_target_interaction'].to(device))

                loss_details = self.model(x_context, x_target_self, x_target_interaction, x_target_total, points_info,
                                          state_dim, ((mu_x_target_self, mu_x_target_interaction), (std_x_target_self, std_x_target_interaction)))
                #loss_details = self.model(x_context, x_target_self, x_target_interaction, x_target_total, points_info,
                #                          state_dim, None)
                # loss_details = self.model(x_context, x_target_self, x_target_interaction, x_target_total, points_info,
                #                          state_dim, (mu_x_target, std_x_target))

                print("forward cost = %.2f" % (time.time() - start_forward_time))

                start_backward_time = time.time()
                loss_details['loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                optim.step()

                loss_total += loss_details['loss'].item() * batch_data['num_in_batch']

                print("backward cost = %.2f" % (time.time() - start_backward_time))

                print('epoch %s, step %s (has %s), loss = %s (loss1 = %s, loss2 = %s, loss3 = %s), total loss = %s, '
                      'cost_time = %.2f' \
                      % (epoch + 1, step, batch_data['num_in_batch'],
                         loss_details['loss'].item(),
                         loss_details['loss_1'],
                         loss_details['loss_2'],
                         loss_details['loss_3'],
                         loss_total / (step * batch_data['num_in_batch']),
                         time.time() - start_time), end='\r\n')
            # scheduler.step()
            #if torch.cuda.is_available():
            #    torch.cuda.empty_cache()  # 释放显存

            if epoch % 1 == 0:
                torch.save(self.model.state_dict(),
                           "saved_model_train_on_exist_oneNODE_new_hypernet_NP_multidim_7sys+2sys.pkl")
                self.model.eval()
                res_self = self.model.CNP_self(x_context, x_target_self, state_dim, None, weights_emb=None)
                res_interaction = self.model.CNP_interaction(x_context, x_target_interaction, state_dim, None, weights_emb=res_self['weights_emb'])
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                
                #x_target_out_ = (x_target_self[1] - mu_x_target_self) / std_x_target_self  # normalize x_target_out for training
                x_target_out_ = x_target_self[1]
                
                # for ii in range(x_target_self[1].size(0)):
                #    axs[0].scatter(x_target_self[1][ii].cpu(),res_self['pre_dist'].loc.detach().cpu().mean(0)[ii], alpha=0.5)
                #    axs[1].scatter(x_target_interaction[1][ii].cpu(),res_interaction['pre_dist'].loc.detach().cpu().mean(0)[ii], alpha=0.5)
                axs[0].plot([torch.min(x_target_out_.cpu().view(-1)), torch.max(x_target_out_.cpu().view(-1))],
                            [torch.min(x_target_out_.cpu().view(-1)), torch.max(x_target_out_.cpu().view(-1))],
                            'k:')
                axs[0].scatter(x_target_out_.cpu().view(-1),
                               res_self['pre_dist'].loc.detach().cpu().mean(0).view(-1), alpha=0.5)
                               
                #x_target_out_ = (x_target_interaction[1] - mu_x_target_interaction) / std_x_target_interaction  # normalize x_target_out for training     
                x_target_out_ = x_target_interaction[1]         
                axs[1].plot([torch.min(x_target_out_.cpu().view(-1)),
                             torch.max(x_target_out_.cpu().view(-1))],
                            [torch.min(x_target_out_.cpu().view(-1)),
                             torch.max(x_target_out_.cpu().view(-1))], 'k:')
                axs[1].scatter(x_target_out_.cpu().view(-1),
                               res_interaction['pre_dist'].loc.detach().cpu().mean(0).view(-1), alpha=0.5)
                plt.savefig('eval_oneNODE_epoch_%s_multidim.png' % (epoch + 1))
                plt.close()

    # train
    def get_embeddings(self, add_str='', data=None):
        start_time = time.time()

        if True:

            self.model.eval()

            gen_batch = self.make_batch(data, flag_IEEE754=True)
            step = 0
            loss_total = 0.
            for batch_data in gen_batch:
                step += 1

                x_context = move_list_to_device(batch_data['x_context'], device)
                x_target_self = move_list_to_device(batch_data['x_target_self'], device)
                x_target_interaction = move_list_to_device(batch_data['x_target_interaction'], device)
                x_target_total = batch_data['x_target_total'].to(device)
                points_info = batch_data['points_info'].to(device)
                state_dim = batch_data['state_dim']

                eq_type_batch = batch_data['eq_type_batch']

                # [b, 1, d]
                z_1_dist = self.model.CNP_self.encoder(x_context, state_dim)
                z_2_dist = self.model.CNP_interaction.encoder(x_context, state_dim)

                # [b, d+d]
                Z1 = z_1_dist.loc.detach().cpu().sum(1).numpy()
                Z2 = z_2_dist.loc.detach().cpu().sum(1).numpy()

                labels = eq_type_batch  # [b, ]

                np.savetxt('display_embeddings/_Z1_embeddings_%s_step%s.txt'%(add_str, step), Z1)
                np.savetxt('display_embeddings/_Z2_embeddings_%s_step%s.txt' % (add_str, step), Z2)
                # np.savetxt('display_embeddings/_labels_%s_step%s.txt' % (add_str, step), labels)

                with open('display_embeddings/_labels_%s_step%s.txt' % (add_str, step),'w') as f:
                    for label_idx in range(len(labels)):
                        if label_idx == 0:
                            f.write(labels[label_idx])
                        else:
                            f.write('\n'+labels[label_idx])
                print('step %s (has %s), %s'%(step, batch_data['num_in_batch'], time.time()-start_time))


# Simulation_config,MutualisticInteraction_config, GeneRegulatory_config
@hydra.main(config_name="Simulation_config", version_base='1.2', config_path='configs')
def main(cfg):
    # generate_data = GenerateData(cfg, eq_filename='data/dataset_5000.csv')
    # generate_data.load_data_from_file('data/dataset_5000.csv_norm_stateTrue.pkl')

    #trainmodel = TrainCNPFoundationModel(cfg, model_path="saved_model_train_on_exist_oneNODE_new_hypernet_NP_multidim_7sys+2sys.pkl")
    trainmodel = TrainCNPFoundationModel(cfg, model_path="saved_model_train_on_exist_oneNODE_new_hypernet_NP_multidim_11sys+3sys.pkl")
    # trainmodel = TrainCNPFoundationModel(cfg)
    # trainmodel.train(generate_data.generated_data)
    #trainmodel.train()
    add_str = 'trainset'
    #add_str = 'testset'
    trainmodel.get_embeddings(add_str=add_str)


if __name__ == "__main__":
    main()
