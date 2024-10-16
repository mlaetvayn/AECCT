import numpy as np
import torch
from torch.nn import LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import logging
import einops
from dataset import sign_to_bin
from quantize import AAPLinearTraining, AAPLinearInference, aap_quantization
from configuration import Config, Code


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        if N > 1:
            self.norm2 = LayerNorm(layer.size)

    def forward(self, x, mask):
        for idx, layer in enumerate(self.layers, start=1):
            x = layer(x, mask)
            if idx == len(self.layers)//2 and len(self.layers) > 1:
                x = self.norm2(x)
        return self.norm(x)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout, config: Config):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        x = x + self.dropout(sublayer(x))
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout, config: Config):
        super(EncoderLayer, self).__init__()
        self.self_attn: MultiHeadedAttention = self_attn
        self.feed_forward: PositionwiseFeedForward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, config), 2)
        self.size = size
        self.config = config

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, lambda x: self.feed_forward(x))


def lpe_attn(x, heads: int, d_head: int):
    x = x.view(x.size(0), x.size(0), heads, d_head)
    scores = torch.matmul(x, x.transpose(-2, -1)) \
    / math.sqrt(d_head)
    p_attn = F.softmax(scores, dim=-1)
    x = torch.matmul(p_attn, x)
    x = x.view(x.size(0), x.size(1), heads * d_head)
    lpe = torch.mean(x, dim=1)
    return lpe


class MultiHeadedAttention(nn.Module):
    def __init__(
            self,
            args: Config,
            dropout=0.1,
    ):
        super(MultiHeadedAttention, self).__init__()
        assert args.d_model % args.h == 0
        self.d_k = args.d_model // args.h
        self.h = args.h
        if args.use_aap_linear_training:
            self.linears = clones(AAPLinearTraining(args.d_model, args.d_model, config=args), 4)
        elif args.use_aap_linear_inference:
            self.linears = clones(AAPLinearInference(args.d_model, args.d_model, config=args), 4)
        else:
            self.linears = clones(nn.Linear(args.d_model, args.d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.num_variable_nodes = args.code.n
        self.num_heads_one_ring = args.num_heads_for_one_ring

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = self.hpsa(query, key, value, mask=mask)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, q, k, v, mask):
        scores = torch.matmul(q, k.transpose(-2, -1)) \
            / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.bool(), -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, v), p_attn

    def hpsa(self, q, k, v, mask):
        variable_to_variable_mask = mask[:, :, :self.num_variable_nodes, :self.num_variable_nodes]
        check_to_check_mask = mask[:, :, self.num_variable_nodes:, self.num_variable_nodes:]
        variable_to_check_mask = mask[:, :, :self.num_variable_nodes, self.num_variable_nodes:]
            

        def split_variable_and_check_nodes(nodes: torch.Tensor) -> torch.Tensor:
            num_nodes = nodes.size(2)
            num_check_nodes = num_nodes - self.num_variable_nodes
            return torch.split(nodes, [self.num_variable_nodes, num_check_nodes], dim=2)

        num_heads_second_ring = self.h - self.num_heads_one_ring

        query_first_ring_heads, query_second_ring_heads = torch.split(q, [self.num_heads_one_ring, num_heads_second_ring], dim=1)
        key_first_ring_heads, key_second_ring_heads = torch.split(k, [self.num_heads_one_ring, num_heads_second_ring], dim=1)
        value_first_ring_heads, value_second_ring_heads = torch.split(v, [self.num_heads_one_ring, num_heads_second_ring], dim=1)

        # first ring
        q_variable, q_check = split_variable_and_check_nodes(query_first_ring_heads)
        k_variable, k_check = split_variable_and_check_nodes(key_first_ring_heads)
        v_variable, v_check = split_variable_and_check_nodes(value_first_ring_heads)
        
        first_ring_valiable_nodes, v_to_c_attn = self.attention(q_variable, k_check, v_check, variable_to_check_mask)
        first_ring_check_nodes, c_to_v_attn = self.attention(q_check, k_variable, v_variable, variable_to_check_mask.transpose(-2, -1))

        # second ring
        q_variable, q_check = split_variable_and_check_nodes(query_second_ring_heads)
        k_variable, k_check = split_variable_and_check_nodes(key_second_ring_heads)
        v_variable, v_check = split_variable_and_check_nodes(value_second_ring_heads)

        second_ring_variable_nodes, v_to_v_attn = self.attention(q_variable, k_variable, v_variable, variable_to_variable_mask)
        second_ring_check_nodes, c_to_c_attn = self.attention(q_check, k_check, v_check, check_to_check_mask)

        # merge
        first_ring_heads = torch.cat([first_ring_valiable_nodes, first_ring_check_nodes], dim=2)
        second_ring_heads = torch.cat([second_ring_variable_nodes, second_ring_check_nodes], dim=2)
        context = torch.cat([first_ring_heads, second_ring_heads], dim=1)

        # create attention map
        first_ring_heads_v = torch.cat([torch.zeros_like(v_to_v_attn), v_to_c_attn], dim=-1)
        first_ring_heads_c = torch.cat([c_to_v_attn, torch.zeros_like(c_to_c_attn)], dim=-1)
        first_ring_heads_attn = torch.cat([first_ring_heads_v, first_ring_heads_c], dim=-2)

        second_ring_heads_v = torch.cat([v_to_v_attn, torch.zeros_like(v_to_c_attn)], dim=-1)
        second_ring_heads_c = torch.cat([torch.zeros_like(c_to_v_attn), c_to_c_attn], dim=-1)
        second_ring_heads_attn = torch.cat([second_ring_heads_v, second_ring_heads_c], dim=-2)

        attn = torch.concat([first_ring_heads_attn, second_ring_heads_attn], dim=1)

        return context, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(
            self,
            args: Config,
            dropout=0,
        ):
        super(PositionwiseFeedForward, self).__init__()
        if args.use_aap_linear_training:
            self.w_1 = AAPLinearTraining(args.d_model, args.d_model*4, config=args)
            self.w_2 = AAPLinearTraining(args.d_model*4, args.d_model, config=args)
        elif args.use_aap_linear_inference:
            self.w_1 = AAPLinearInference(args.d_model, args.d_model*4, config=args)
            self.w_2 = AAPLinearInference(args.d_model*4, args.d_model, config=args)
        else:
            self.w_1 = nn.Linear(args.d_model, args.d_model*4)
            self.w_2 = nn.Linear(args.d_model*4, args.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.w_2(self.dropout(F.relu(self.w_1(x))))
        return out


class ECC_Transformer(nn.Module):
    def __init__(
        self,
        args: Config,
        dropout=0,
    ):
        super(ECC_Transformer, self).__init__()
        code = args.code
        c = copy.deepcopy
        attn = MultiHeadedAttention(args)
        ff = PositionwiseFeedForward(args)

        emb_dim = args.d_model - args.lpe_dim

        self.src_embed = torch.nn.Parameter(torch.empty(
            (code.n + code.pc_matrix.size(0), emb_dim)))
        self.decoder = Encoder(EncoderLayer(
            args.d_model, c(attn), c(ff), dropout, args), args.N_dec)
        self.oned_final_embed = torch.nn.Sequential(
            *[nn.Linear(args.d_model, 1)])
        self.out_fc = nn.Linear(code.n + code.pc_matrix.size(0), code.n)

        self.get_mask(code)
        self.create_laplacian_pe(self.get_adj_matrix(args.code).float().squeeze(0).squeeze(0))
        self.lpe_proj = nn.Linear(self.lpe.size(-1), args.lpe_dim)
        self.attn_lpe = lambda x: lpe_attn(x, args.lpe_num_heads, args.lpe_dim // args.lpe_num_heads)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, magnitude, syndrome):
        emb = torch.cat([magnitude, syndrome], -1).unsqueeze(-1)
        emb = self.src_embed.unsqueeze(0) * emb
        lpe = self.lpe_proj(self.lpe)
        lpe = self.attn_lpe(lpe).unsqueeze(0)
        bached_lpe = einops.repeat(lpe, "a b c -> (repeat a) b c", repeat=emb.size(0))
        emb = torch.cat([emb, bached_lpe], dim=-1)
        emb = self.decoder(emb, self.src_mask)
        return self.out_fc(self.oned_final_embed(emb).squeeze(-1))

    def loss(self, z_pred, z2, y):
        loss = F.binary_cross_entropy_with_logits(
            z_pred, sign_to_bin(torch.sign(z2)))
        x_pred = sign_to_bin(torch.sign(-z_pred * torch.sign(y)))
        return loss, x_pred

    def get_mask(self, code: Code):
        def build_mask(code):
            mask_size = code.n + code.pc_matrix.size(0)
            mask = torch.eye(mask_size, mask_size)
            for ii in range(code.pc_matrix.size(0)):
                idx = torch.where(code.pc_matrix[ii] > 0)[0]
                for jj in idx:
                    for kk in idx:
                        if jj != kk:
                            mask[jj, kk] += 1
                            mask[kk, jj] += 1
                            mask[code.n + ii, jj] += 1
                            mask[jj, code.n + ii] += 1
            src_mask = ~ (mask > 0).unsqueeze(0).unsqueeze(0)
            return src_mask
        src_mask = build_mask(code)
        self.register_buffer('src_mask', src_mask)

    def get_adj_matrix(self, code):
        mask_size = code.n + code.pc_matrix.size(0)
        mask = torch.eye(mask_size, mask_size)
        for ii in range(code.pc_matrix.size(0)):
            idx = torch.where(code.pc_matrix[ii] > 0)[0]
            for jj in idx:
                mask[code.n + ii, jj] = 1
                mask[jj, code.n + ii] = 1
        src_mask = mask.unsqueeze(0).unsqueeze(0)
        return src_mask

    def create_laplacian_pe(self, adjacency_matrix):
        degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))
        laplacian_matrix = degree_matrix - adjacency_matrix
        eigenvalues, eigenvectors = torch.linalg.eig(laplacian_matrix)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        lpe_eigenvectors = eigenvectors.unsqueeze(1)
        lpe_eigenvalues = eigenvalues.unsqueeze(0).unsqueeze(0)
        lpe_eigenvalues = einops.repeat(lpe_eigenvalues, "a b c -> (repeat a) b c", repeat=lpe_eigenvectors.size(0))
        lpe = torch.cat([lpe_eigenvalues, lpe_eigenvectors], dim=1)
        self.register_buffer('lpe', lpe.transpose(-2, -1).float())


def freeze_weights(inference_model: ECC_Transformer, config: Config):
    with torch.no_grad():
        block: EncoderLayer
        for block in inference_model.decoder.layers:
            proj: AAPLinearInference
            for proj in block.self_attn.linears:
                q_w, s_w = aap_quantization(
                    proj.weight,
                    dequantize=False,
                    initial_percentile=config.initial_percentile,
                    delta=proj.delta,
                )
                proj.weight.copy_(q_w)
                proj.s_w = torch.nn.Parameter(data=s_w, requires_grad=False)
            
            q_w, s_w = aap_quantization(
                block.feed_forward.w_1.weight,
                dequantize=False,
                initial_percentile=config.initial_percentile,
                delta=block.feed_forward.w_1.delta,
            )
            block.feed_forward.w_1.weight.copy_(q_w)
            block.feed_forward.w_1.s_w = torch.nn.Parameter(data=s_w, requires_grad=False)

            q_w, s_w = aap_quantization(
                block.feed_forward.w_2.weight,
                dequantize=False,
                initial_percentile=config.initial_percentile,
                delta=block.feed_forward.w_2.delta,
            )
            block.feed_forward.w_2.weight.copy_(q_w)
            block.feed_forward.w_2.s_w = torch.nn.Parameter(data=s_w, requires_grad=False)
