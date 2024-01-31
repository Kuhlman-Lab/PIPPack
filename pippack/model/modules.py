import logging
import math
from typing import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import pippack.data.residue_constants as rc
from pippack.data.features import get_bb_frames, torsion_angles_to_frames, frames_and_literature_positions_to_atom14_pos
from pippack.model.loss import rotamer_recovery_from_coords, nll_chi_loss, offset_mse, supervised_chi_loss, BlackHole, sc_rmsd, local_interresidue_sc_clash_loss, unclosed_proline_loss

# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    # Features [...,N,C] at Neighbor indices [...,N,K] => [...,N,K,C]
    is_batched = neighbor_idx.dim() == 3
    n_feat_dims = nodes.dim() - (1 + is_batched)

    # Flatten and expand indices per batch [...,N,K] => [...,NK] => [...,NK,C]
    neighbors_flat = neighbor_idx.view((*neighbor_idx.shape[:-2], -1))
    for _ in range(n_feat_dims):
        neighbors_flat = neighbors_flat.unsqueeze(-1)
    neighbors_flat = neighbors_flat.expand(*([-1] * (1 + is_batched)), *nodes.shape[-n_feat_dims:])
    
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, -n_feat_dims - 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape) + list(nodes.shape[-n_feat_dims:]))
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


def get_act_fxn(act: str):
    if act == 'relu':
        return F.relu
    elif act == 'gelu':
        return F.gelu
    elif act == 'elu':
        return F.elu
    elif act == 'selu':
        return F.selu
    elif act == 'celu':
        return F.celu
    elif act == 'leaky_relu':
        return F.leaky_relu
    elif act == 'prelu':
        return F.prelu
    elif act == 'silu':
        return F.silu
    elif act == 'sigmoid':
        return nn.Sigmoid()


class MLP(nn.Module):
    def __init__(self, num_in, num_inter, num_out, num_layers, act='relu', bias=True):
        super().__init__()
        
        # Linear layers for MLP
        self.W_in = nn.Linear(num_in, num_inter, bias=bias)
        self.W_inter = nn.ModuleList([nn.Linear(num_inter, num_inter, bias=bias) for _ in range(num_layers - 2)])
        self.W_out = nn.Linear(num_inter, num_out, bias=bias)
        
        # Activation function
        self.act = get_act_fxn(act)
        
    def forward(self, X):
        
        # Embed inputs with input layer
        X = self.act(self.W_in(X))
        
        # Pass through intermediate layers
        for layer in self.W_inter:
            X = self.act(layer(X))
            
        # Get output from output layer
        X = self.W_out(X)
        
        return X


class MPNNLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30, edge_update=False, act='relu', extra_params=0):
        super(MPNNLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.edge_update = edge_update
        
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(2)])
        self.W_v = MLP(num_hidden + num_in, num_hidden + extra_params, num_hidden, num_layers=3, act=act)
        self.dense = MLP(num_hidden, num_hidden * 4, num_hidden, num_layers=2, act=act)
        
        self.act = get_act_fxn(act)
        
        if edge_update:
            self.W_e = MLP(num_hidden + num_in, num_hidden + extra_params, num_hidden, num_layers=3, act=act)
            self.dropout2 = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm(num_hidden)
            
    def forward(self, h_V, h_E, E_idx=None, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """
        
        if torch.is_tensor(E_idx):
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            # Concatenate h_V_i to h_E_ij
            h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
            h_EV = torch.cat([h_V_expand, h_EV], -1)
        else:
            # Concatenate h_V_i to h_E_ij
            h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
            h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W_v(h_EV)
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm[0](h_V + self.dropout[0](dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout[1](dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
            
        if self.edge_update:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
            h_EV = torch.cat([h_V_expand, h_EV], -1)
            h_message = self.W_e(h_EV)
            h_E = self.norm2(h_E + self.dropout2(h_message))
            
            return h_V, h_E
        else:
            return h_V


class InvariantPointMessagePassing(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, n_points=8, dropout=0.1, act='relu', edge_update=False, position_scale=10.0):
        super().__init__()
        
        self.edge_update = edge_update
        self.n_points = n_points
        self.position_scale = position_scale
        self.points_fn_node = nn.Linear(node_dim, n_points * 3)
        if edge_update:
            self.points_fn_edge = nn.Linear(node_dim, n_points * 3)

        # Input to message is: 2*node_dim + edge_dim + 3*3*n_points
        self.node_message_fn = MLP(2 * node_dim + edge_dim + 9 * n_points, hidden_dim, hidden_dim, 3, act=act)
        if edge_update:
            self.edge_message_fn = MLP(2 * node_dim + edge_dim + 9 * n_points, hidden_dim, hidden_dim, 3, act=act)

        # Dropout and layer norms
        n_layers = 2
        if edge_update:
            n_layers = 4
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

        # Feedforward layers
        self.node_dense = MLP(hidden_dim, hidden_dim * 4, hidden_dim, num_layers=2, act=act)
        if edge_update:
            self.edge_dense = MLP(hidden_dim, hidden_dim * 4, hidden_dim, num_layers=2, act=act)
    
    def _get_message_input(self, h_V, h_E, E_idx, X, edge=False):
        # Get backbone global frames from N, CA, and C
        bb_to_global = get_bb_frames(X[..., 0, :], X[..., 1, :], X[..., 2, :])
        bb_to_global = bb_to_global.scale_translation(1 / self.position_scale)
        
        # Generate points in local frame of each node
        if not edge:
            p_local = self.points_fn_node(h_V).reshape((*h_V.shape[:-1], self.n_points, 3)) # [B, L, N, 3]
        else:
            p_local = self.points_fn_edge(h_V).reshape((*h_V.shape[:-1], self.n_points, 3)) # [B, L, N, 3]
        
        # Project points into global frame
        p_global = bb_to_global[..., None].apply(p_local) # [B, L, N, 3]
        p_global_expand = p_global.unsqueeze(-3).expand(*E_idx.shape, *p_global.shape[-2:]) # [B, L, K, N, 3]

        # Get neighbor points in global frame for each node
        neighbor_idx = E_idx.view((*E_idx.shape[:-2], -1)) # [B, LK]
        neighbor_p_global = torch.gather(p_global, -3, neighbor_idx[..., None, None].expand(*neighbor_idx.shape, self.n_points, 3))
        neighbor_p_global = neighbor_p_global.view(*E_idx.shape, self.n_points, 3) # [B, L, K, N, 3]

        # Form message components:
        # 1. Node i's local points
        p_local_expand = p_local.unsqueeze(-3).expand(*E_idx.shape, *p_local.shape[-2:]) # [B, L, K, N, 3]
        
        # 2. Distance between node i's local points and its CA
        p_local_norm = torch.sqrt(torch.sum(p_local_expand ** 2, dim=-1) + 1e-8) # [B, L, K, N]

        # 3. Node j's points in i's local frame
        neighbor_p_local = bb_to_global[..., None, None].invert_apply(neighbor_p_global) # [B, L, K, N, 3]

        # 4. Distance between node j's points in i's local frame and i's CA
        neighbor_p_local_norm = torch.sqrt(torch.sum(neighbor_p_local ** 2, dim=-1) + 1e-8) # [B, L, K, N]

        # 5. Distance between node i's global points and node j's global points
        neighbor_p_global_norm = torch.sqrt(
            torch.sum(
                (p_global_expand - neighbor_p_global) ** 2, 
                dim=-1) + 1e-8) # [B, L, K, N]
        
        # Node message
        node_expand = h_V.unsqueeze(-2).expand(*E_idx.shape, h_V.shape[-1])
        neighbor_edge = cat_neighbors_nodes(h_V, h_E, E_idx)
        message_in = torch.cat(
            (node_expand, 
             neighbor_edge, 
             p_local_expand.view((*E_idx.shape, -1)), 
             p_local_norm, 
             neighbor_p_local.view((*E_idx.shape, -1)), 
             neighbor_p_local_norm,
             neighbor_p_global_norm), dim=-1)
        
        return message_in
    
    def forward(self, h_V, h_E, E_idx, X, mask_V=None, mask_attend=None):
        # Get message fn input
        message_in = self._get_message_input(h_V, h_E, E_idx, X)

        # Update nodes
        node_m = self.node_message_fn(message_in)
        if mask_attend is not None:
            node_m = node_m * mask_attend[..., None]
        node_m = torch.mean(node_m, dim=-2)
        h_V = self.norm[0](h_V + self.dropout[0](node_m))
        node_m = self.node_dense(h_V)
        h_V = self.norm[1](h_V + self.dropout[1](node_m))
        if mask_V is not None:
            h_V = h_V * mask_V[..., None]

        if self.edge_update:
            # Get message fn input
            message_in = self._get_message_input(h_V, h_E, E_idx, X, edge=True)
                    
            # Update edges
            edge_m = self.edge_message_fn(message_in)
            if mask_attend is not None:
                edge_m = edge_m * mask_attend[..., None]
            h_E = self.norm[2](h_E + self.dropout[2](edge_m))
            edge_m = self.edge_dense(h_E)
            h_E = self.norm[3](h_E + self.dropout[3](edge_m))
            if mask_attend is not None:
                h_E = h_E * mask_attend[..., None]

        return h_V, h_E


class IPMP_IPA(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=16, n_heads=1, n_query_points=4, n_value_points=8, edge_update=False, position_scale=10.0, dropout=0.1, act='relu'):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_query_points = n_query_points
        self.n_value_points = n_value_points
        self.position_scale = position_scale
        self.edge_update = edge_update
        
        # Linear layers for queries, keys, and values
        self.linear_q = nn.Linear(node_dim, hidden_dim * n_heads)
        self.linear_kv = nn.Linear(node_dim, 2 * hidden_dim * n_heads)
        
        self.linear_q_points = nn.Linear(node_dim, n_heads * n_query_points * 3)
        self.linear_kv_points = nn.Linear(node_dim, n_heads * (n_query_points + n_value_points) * 3)
        
        self.linear_b = nn.Linear(edge_dim, n_heads)
        
        self.head_weights = nn.Parameter(torch.zeros((n_heads)))
        with torch.no_grad():
            self.head_weights.fill_(0.541324854612918)
        
        out_dim = n_heads * (edge_dim + hidden_dim + n_value_points * 4)
        self.linear_out = nn.Linear(out_dim, node_dim)
        
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])
        
        self.node_dense = MLP(node_dim, node_dim * 4, node_dim, num_layers=2, act=act)
        
        if edge_update:
            # Linear layers for queries, keys, and values
            self.linear_q_e = nn.Linear(node_dim, hidden_dim * n_heads)
            self.linear_kv_e = nn.Linear(node_dim, 2 * hidden_dim * n_heads)
            
            self.linear_q_points_e = nn.Linear(node_dim, n_heads * n_query_points * 3)
            self.linear_kv_points_e = nn.Linear(node_dim, n_heads * (n_query_points + n_value_points) * 3)
            
            self.linear_b_e = nn.Linear(edge_dim, n_heads)
            
            self.head_weights_e = nn.Parameter(torch.zeros((n_heads)))
            with torch.no_grad():
                self.head_weights_e.fill_(0.541324854612918)
            
            out_dim = n_heads * (edge_dim + hidden_dim + n_value_points * 4)
            self.linear_out_e = nn.Linear(out_dim, edge_dim)
            
            self.dropout_e = nn.ModuleList([nn.Dropout(dropout) for _ in range(2)])
            self.norm_e = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])
            
            self.edge_dense = MLP(edge_dim, edge_dim * 4, edge_dim, num_layers=2, act=act)
        
    def _get_node_update(self, h_V, h_E, E_idx, X, mask_attend=None):
        # Get backbone global frames from N, CA, and C
        scaled_X = X / self.position_scale
        bb_to_global = get_bb_frames(scaled_X[..., 0, :], scaled_X[..., 1, :], scaled_X[..., 2, :])
        
        # Generate queries, keys, and values from nodes
        q = self.linear_q(h_V) # [*, N_res, H * C]
        q = q.view(q.shape[:-1] + (self.n_heads, -1)) # [*, N_res, H, C]
        
        kv = self.linear_kv(h_V) # [*, N_res, 2 * H * C]
        kv = kv.view(kv.shape[:-1] + (self.n_heads, -1)) # [*, N_res, H, 2 * C]
        k, v = torch.split(kv, self.hidden_dim, dim=-1) # 2 [*, N_res, H, C]
        
        # Generate query, key, and value points from nodes
        q_pts = self.linear_q_points(h_V) # [*, N_res, H * P_q * 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1) # 3 [*, N_res, H * P_q]
        q_pts = torch.stack(q_pts, dim=-1) # [*, N_res, H * P_q, 3]
        q_pts = bb_to_global[..., None].apply(q_pts) # [*, N_res, H * P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.n_heads, self.n_query_points, 3)
        ) # [*, N_res, H, P_q, 3]
        
        kv_pts = self.linear_kv_points(h_V) # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1) # 3 [*, N_res, H * (P_q + P_v)]
        kv_pts = torch.stack(kv_pts, dim=-1) # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = bb_to_global[..., None].apply(kv_pts) # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.n_heads, -1, 3))
        k_pts, v_pts = torch.split(
            kv_pts, [self.n_query_points, self.n_value_points], dim=-2
        )# [*, N_res, H, P_q, 3], [*, N_res, H, P_v, 3]
        
        # Compute attention bias
        b = self.linear_b(h_E) # [*, N_res, K, H]
        
        # Compute attention weight
        a = torch.einsum("...ihc,...ijhc->...ijh", q, gather_nodes(k, E_idx))
        a *= math.sqrt(1.0 / (3 * self.hidden_dim))
        a += math.sqrt(1.0 / 3) * b # [*, N_res, K, H]
        
        pt_att = q_pts.unsqueeze(-4) - gather_nodes(k_pts, E_idx) # [*, N_res, K, H, P_q, 3]
        pt_att = torch.sum(pt_att ** 2, dim=-1) # [*, N_res, K, H, P_q]
        
        head_weights = F.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        ) # [*, 1, 1, H, 1]
        pt_att = math.sqrt(1.0 / (3 * (self.n_query_points * 9.0 / 2))) * head_weights * pt_att # [*, N_res, K, H, P_q]
        pt_att = torch.sum(pt_att, dim=-1) * -0.5 # [*, N_res, K, H]
        
        if mask_attend is not None:
            att_mask = 1e5 * (mask_attend - 1)
        else:
            att_mask = torch.zeros_like(E_idx)
        
        a = a + pt_att + att_mask[..., None] # [*, N_res, K, H]
        a = F.softmax(a, dim=-2) # [*, N_res, K, H]
        
        # Compute update
        # [*, N_res, H, C_hidden]
        o = torch.einsum('...ijh,...ijhc->...ihc', a, gather_nodes(v, E_idx))
        o = o.view(*o.shape[:-2], -1)

        o_pt = torch.einsum("...ijh,...ijhpx->...ihpx", a, gather_nodes(v_pts, E_idx))
        o_pt = bb_to_global[..., None, None].invert_apply(o_pt) # [*, N_res, H, P_v, 3]
        o_pt_norm = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + 1e-8).view(*o_pt.shape[:-3], -1)
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)
        
        o_pair = torch.einsum("...ijh,...ijc->...ihc", a, h_E) # [*, N_res, H, C_z]
        o_pair = o_pair.view(*o_pair.shape[:-2], -1)
        
        # Compute node update
        s = self.linear_out(
            torch.cat(
                (o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1
            )
        )
        
        return s
        
    def _get_edge_update(self, h_V, h_E, E_idx, X, mask_attend=None):
        # Get backbone global frames from N, CA, and C
        scaled_X = X / self.position_scale
        bb_to_global = get_bb_frames(scaled_X[..., 0, :], scaled_X[..., 1, :], scaled_X[..., 2, :])
        
        # Generate queries, keys, and values from nodes
        q = self.linear_q_e(h_V) # [*, N_res, H * C]
        q = q.view(q.shape[:-1] + (self.n_heads, -1)) # [*, N_res, H, C]
        
        kv = self.linear_kv_e(h_V) # [*, N_res, 2 * H * C]
        kv = kv.view(kv.shape[:-1] + (self.n_heads, -1)) # [*, N_res, H, 2 * C]
        k, v = torch.split(kv, self.hidden_dim, dim=-1) # 2 [*, N_res, H, C]
        
        # Generate query, key, and value points from nodes
        q_pts = self.linear_q_points_e(h_V) # [*, N_res, H * P_q * 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1) # 3 [*, N_res, H * P_q]
        q_pts = torch.stack(q_pts, dim=-1) # [*, N_res, H * P_q, 3]
        q_pts = bb_to_global[..., None].apply(q_pts) # [*, N_res, H * P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.n_heads, self.n_query_points, 3)
        ) # [*, N_res, H, P_q, 3]
        
        kv_pts = self.linear_kv_points_e(h_V) # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1) # 3 [*, N_res, H * (P_q + P_v)]
        kv_pts = torch.stack(kv_pts, dim=-1) # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = bb_to_global[..., None].apply(kv_pts) # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.n_heads, -1, 3))
        k_pts, v_pts = torch.split(
            kv_pts, [self.n_query_points, self.n_value_points], dim=-2
        )# [*, N_res, H, P_q, 3], [*, N_res, H, P_v, 3]
        
        # Compute attention bias
        b = self.linear_b_e(h_E) # [*, N_res, K, H]
        
        # Compute attention weight
        a = torch.einsum("...ihc,...ijhc->...ijh", q, gather_nodes(k, E_idx))
        a *= math.sqrt(1.0 / (3 * self.hidden_dim))
        a += math.sqrt(1.0 / 3) * b # [*, N_res, K, H]
        
        pt_att = q_pts.unsqueeze(-4) - gather_nodes(k_pts, E_idx) # [*, N_res, K, H, P_q, 3]
        pt_att = torch.sum(pt_att ** 2, dim=-1) # [*, N_res, K, H, P_q]
        
        head_weights = F.softplus(self.head_weights_e).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        ) # [*, 1, 1, H, 1]
        pt_att = math.sqrt(1.0 / (3 * (self.n_query_points * 9.0 / 2))) * head_weights * pt_att # [*, N_res, K, H, P_q]
        pt_att = torch.sum(pt_att, dim=-1) * -0.5 # [*, N_res, K, H]
        
        if mask_attend is not None:
            att_mask = 1e5 * (mask_attend - 1)
        else:
            att_mask = torch.zeros_like(E_idx)
        
        a = a + pt_att + att_mask[..., None] # [*, N_res, K, H]
        a = F.softmax(a, dim=-2) # [*, N_res, K, H]
        
        # Compute update
        # [*, N_res, K, H, C_hidden]
        o = torch.einsum('...ijh,...ijhc->...ijhc', a, gather_nodes(v, E_idx))
        o = o.view(*o.shape[:-2], -1)

        o_pt = torch.einsum("...ijh,...ijhpx->...ijhpx", a, gather_nodes(v_pts, E_idx))
        o_pt = bb_to_global[..., None, None, None].invert_apply(o_pt) # [*, N_res, K, H, P_v, 3]
        o_pt_norm = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + 1e-8).view(*o_pt.shape[:-3], -1)
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)
        
        o_pair = torch.einsum("...ijh,...ijc->...ijhc", a, h_E) # [*, N_res, K, H, C_z]
        o_pair = o_pair.view(*o_pair.shape[:-2], -1)
        
        # Compute edge update
        s = self.linear_out_e(
            torch.cat(
                (o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1
            )
        )
        
        return s
        
    def forward(self, h_V, h_E, E_idx, X, mask_V=None, mask_attend=None):
        s = self._get_node_update(h_V, h_E, E_idx, X, mask_attend)
        h_V = self.norm[0](h_V + self.dropout[0](s))
        node_m = self.node_dense(h_V)
        h_V = self.norm[1](h_V + self.dropout[1](node_m))
        
        if mask_V is not None:
            h_V = h_V * mask_V[..., None]
            
        if self.edge_update:
            s = self._get_edge_update(h_V, h_E, E_idx, X, mask_attend)
            if mask_attend is not None:
                s = s * mask_attend[..., None]
            h_E = self.norm_e[0](h_E + self.dropout_e[0](s))
            edge_m = self.edge_dense(h_E)
            h_E = self.norm_e[1](h_E + self.dropout_e[1](edge_m))
            if mask_attend is not None:
                h_E = h_E * mask_attend[..., None]
        
        return h_V, h_E


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, period_range=[2,1000], max_relative_feature=32, af2_relpos=False):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.period_range = period_range
        self.max_relative_feature = max_relative_feature 
        self.af2_relpos = af2_relpos
        
    def _transformer_encoding(self, E_idx):
        # i-j
        N_nodes = E_idx.size(1)
        ii = torch.arange(N_nodes, dtype=torch.float32, device=E_idx.device).view((1, -1, 1))
        d = (E_idx.float() - ii).unsqueeze(-1)
        
        # Original Transformer frequencies
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32, device=E_idx.device)
            * -(np.log(10000.0) / self.num_embeddings)
        )
        
        # Grid-aligned
        # frequency = 2. * np.pi * torch.exp(
        #     -torch.linspace(
        #         np.log(self.period_range[0]), 
        #         np.log(self.period_range[1]),
        #         self.num_embeddings / 2
        #     )
        # )
        angles = d * frequency.view((1,1,1,-1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        
        return E

    def _af2_encoding(self, E_idx, residue_index=None):
        # i-j
        if residue_index is not None:
            offset = residue_index[..., None] - residue_index[..., None, :]
            offset = torch.gather(offset, -1, E_idx)
        else:
            N_nodes = E_idx.size(1)
            ii = torch.arange(N_nodes, dtype=torch.float32, device=E_idx.device).view((1, -1, 1))
            offset = (E_idx.float() - ii)
        
        relpos = torch.clip(offset.long() + self.max_relative_feature, 0, 2 * self.max_relative_feature)
        relpos = F.one_hot(relpos, 2 * self.max_relative_feature + 1)
        
        return relpos

    def forward(self, E_idx, residue_index=None):

        if self.af2_relpos:
            E = self._af2_encoding(E_idx, residue_index)
        else:
            E = self._transformer_encoding(E_idx)

        return E


class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, augment_eps=0., dropout=0.1, af2_relpos=True):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        
        if af2_relpos:
            num_positional_embeddings = 65

        # Feature dimensions
        node_in = 21 + 3 * 2
        edge_in = num_positional_embeddings + (14 ** 2) * num_rbf

        # Positional encoding
        self.embeddings = PositionalEncodings(num_positional_embeddings, af2_relpos=af2_relpos)
        self.dropout = nn.Dropout(dropout)
        
        # Normalization and embedding
        self.node_embedding = nn.Linear(node_in,  node_features, bias=True)
        self.norm_nodes = nn.LayerNorm(node_features)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1E-6):
        """ Pairwise euclidean distances """
        # Convolutional network on NCHW
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

        # Identify k nearest neighbors (including self)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + 2 * (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, min(self.top_k, X.shape[-2]), dim=-1, largest=False)
        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)

        return D_neighbors, E_idx, mask_neighbors

    def _rbf(self, D):
        # Distance radial basis function
        D_min, D_max, D_count = 0., 20., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)

        # for i in range(D_count):
        #     fig = plt.figure(figsize=(4,4))
        #     ax = fig.add_subplot(111)
        #     rbf_i = RBF.data.numpy()[0,i,:,:]
        #     # rbf_i = D.data.numpy()[0,0,:,:]
        #     plt.imshow(rbf_i, aspect='equal')
        #     plt.axis('off')
        #     plt.tight_layout()
        #     plt.savefig('rbf{}.pdf'.format(i))
        #     print(np.min(rbf_i), np.max(rbf_i), np.mean(rbf_i))
        # exit(0)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def _impute_CB(self, N, CA, C):
        b = CA - N
        c = C - CA
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + CA
        return Cb

    def _atomic_distances(self, X, E_idx):
        
        RBF_all = []
        for i in range(X.shape[-2]):
            for j in range(X.shape[-2]):
                RBF_all.append(self._get_rbf(X[..., i, :], X[..., j, :], E_idx))

        RBF_all = torch.cat(tuple(RBF_all), dim=-1)
        
        return RBF_all

    def forward(self, X, S, BB_D, mask, residue_index=None):
        """ Featurize coordinates as an attributed graph """

        # Data augmentation
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        # Build k-Nearest Neighbors graph
        X_ca = X[:,:,1,:]
        _, E_idx, _ = self._dist(X_ca, mask)

        # Pairwise embeddings
        E_positional = self.embeddings(E_idx, residue_index)
        
        # Pairwise bb atomic distances
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]
        Cb = self._impute_CB(N, Ca, C)
        sc_atoms = X[..., 5:, :]
        X2 = torch.stack((N, Ca, C, O, Cb), dim=-2)
        X2 = torch.cat((X2, sc_atoms), dim=-2) 
        RBF_all = self._atomic_distances(X2, E_idx)
        
        E = torch.cat((E_positional, RBF_all), -1)
            
        Vs = []
        # One-hot encoded sequence
        Vs.append(F.one_hot(S, num_classes=21).float())
        
        # Sin/cos encoded backbone dihedrals
        Vs.append(BB_D.view(*BB_D.shape[:-2], -1))
        
        # Embed nodes
        V = torch.cat(Vs, dim=-1)   
        V = self.node_embedding(V)
        V = self.norm_nodes(V)
        
        # Embed edges
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        return V, E, E_idx, X


def get_atom14_coords(X, S, BB_D, SC_D):

    # Convert angles to sin/cos
    BB_D_sincos = torch.stack((torch.sin(BB_D), torch.cos(BB_D)), dim=-1)
    SC_D_sincos = torch.stack((torch.sin(SC_D), torch.cos(SC_D)), dim=-1)

    # Get backbone global frames from N, CA, and C
    bb_to_global = get_bb_frames(X[..., 0, :], X[..., 1, :], X[..., 2, :])

    # Concatenate all angles
    angle_agglo = torch.cat([BB_D_sincos, SC_D_sincos], dim=-2) # [B, L, 7, 2]

    # Get norm of angles
    norm_denom = torch.sqrt(torch.clamp(torch.sum(angle_agglo ** 2, dim=-1, keepdim=True), min=1e-12))

    # Normalize
    normalized_angles = angle_agglo / norm_denom

    # Make default frames
    default_frames = torch.tensor(rc.restype_rigid_group_default_frame, dtype=torch.float32,
                                  device=X.device, requires_grad=False)

    # Make group ids
    group_idx = torch.tensor(rc.restype_atom14_to_rigid_group, device=X.device,
                             requires_grad=False)

    # Make atom mask
    atom_mask = torch.tensor(rc.restype_atom14_mask, dtype=torch.float32,
                             device=X.device, requires_grad=False)

    # Make literature positions
    lit_positions = torch.tensor(rc.restype_atom14_rigid_group_positions, dtype=torch.float32,
                                 device=X.device, requires_grad=False)

    # Make all global frames
    all_frames_to_global = torsion_angles_to_frames(bb_to_global, normalized_angles, S, default_frames)

    # Predict coordinates
    pred_xyz = frames_and_literature_positions_to_atom14_pos(all_frames_to_global, S, default_frames, group_idx, 
                                                             atom_mask, lit_positions)
    
    # Replace backbone atoms with input coordinates
    pred_xyz[..., :4, :] = X[..., :4, :]

    return pred_xyz


class PIPPack(nn.Module):
    def __init__(
        self, 
        node_features: int = 128,
        edge_features: int = 128,
        hidden_dim: int = 128,
        num_mpnn_layers: int = 3,
        k_neighbors: int = 30,
        augment_eps: float = 0.,
        use_ipmp: bool = False,
        use_ipmp_ipa: bool = False,
        n_points: Optional[int] = None,
        dropout: float = 0.1,
        act: str = "relu",
        predict_bin_chi: bool = True,
        n_chi_bins: int = 72,
        predict_offset: bool = True,
        position_scale: float = 1.0,
        recycle_strategy: str = "mode",
        recycle_SC_D_sc: bool = False,
        recycle_SC_D_probs: bool = False,
        recycle_X: bool = True,
        loss: Optional[Dict[str, Union[float, bool]]] = {
            "chi_nll_loss_weight": 1.0,
            "chi_mse_loss_weight": 1.0,
            "offset_mse_loss_weight": 1.0
        },
    ) -> None:
        """ Graph labeling network """
        super().__init__()
        
        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.recycle_strategy = recycle_strategy
        self.recycle_SC_D_sc = recycle_SC_D_sc
        self.recycle_SC_D_probs = recycle_SC_D_probs
        self.recycle_X = recycle_X
        self.loss = loss
        self.log = logging.getLogger("PIPPack")

        # Featurization layers
        self.features = ProteinFeatures(
            node_features, edge_features, top_k=k_neighbors,
            augment_eps=augment_eps, dropout=dropout
        )

        # Embedding layers
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        
        # Sequence embedding layer
        self.W_seq = nn.Embedding(21, hidden_dim)
        
        # Recycling embedding layers
        if recycle_SC_D_sc:
            self.W_recycle_SC_D_sc = nn.Linear(8, hidden_dim)
        if recycle_SC_D_probs:
            self.W_recycle_SC_D_probs = nn.Linear(4 * (n_chi_bins + 1), hidden_dim)
            
        # Recycling embedding layers
        if recycle_SC_D_sc:
            self.W_recycle_SC_D_sc = nn.Linear(8, hidden_dim)
        if recycle_SC_D_probs:
            self.W_recycle_SC_D_probs = nn.Linear(4 * (n_chi_bins + 1), hidden_dim)


        # MPNN layers
        self.use_ipmp = use_ipmp
        self.use_ipmp_ipa = use_ipmp_ipa
        if use_ipmp:
            self.mpnn_layers = nn.ModuleList([
                InvariantPointMessagePassing(hidden_dim, hidden_dim, hidden_dim, n_points, dropout, act=act, edge_update=True, position_scale=position_scale)
                for _ in range(num_mpnn_layers)
            ])
        elif use_ipmp_ipa:
            self.mpnn_layers = nn.ModuleList([
                IPMP_IPA(hidden_dim, hidden_dim, hidden_dim, edge_update=True, dropout=dropout, act=act)
                for _ in range(num_mpnn_layers)
            ])            
        else:
            self.mpnn_layers = nn.ModuleList([
                MPNNLayer(hidden_dim, hidden_dim * 2, dropout=dropout, edge_update=True, act=act, scale=k_neighbors)
                for _ in range(num_mpnn_layers)
            ])

        # Output layers 
        self.predict_bin_chi = predict_bin_chi
        self.n_chi_bins = n_chi_bins
        out_dim = 8 if not predict_bin_chi else (n_chi_bins + 1) * 4
        self.W_out_chi = MLP(hidden_dim * 2, hidden_dim, out_dim, 3, act=act)

        # Offset prediction
        self.predict_offset = predict_offset
        if predict_offset:
            self.offset_layer = nn.Linear(node_features, 4)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _chi_prediction_from_probs(self, chi_probs, chi_bin_offset=None, strategy="mode"):        
        # One-hot encode predicted chi bin
        if strategy == "mode":
            chi_bin = torch.argmax(chi_probs, dim=-1)
        elif strategy == "sample":
            chi_bin = torch.multinomial(chi_probs.view(-1, chi_probs.shape[-1]), num_samples=1).squeeze(-1).view(*chi_probs.shape[:-1])
        chi_bin_one_hot = F.one_hot(chi_bin, num_classes=self.n_chi_bins + 1)

        # Determine actual chi value from bin
        chi_bin_rad = torch.cat((torch.arange(-torch.pi, torch.pi, 2 * torch.pi / self.n_chi_bins, device=chi_bin.device), torch.tensor([0]).to(device=chi_bin.device)))
        pred_chi_bin = torch.sum(chi_bin_rad.view(*([1] * len(chi_bin.shape)), -1) * chi_bin_one_hot, dim=-1)
        
        # Add bin offset
        if self.predict_offset and chi_bin_offset is not None:
            bin_sample_update = chi_bin_offset
        else:
            bin_sample_update = (2 * torch.pi / self.n_chi_bins) * torch.rand(chi_bin.shape, device=chi_bin.device)
        sampled_chi = pred_chi_bin + bin_sample_update
        
        return sampled_chi

    @property
    def metric_names(self) -> Sequence[str]:
        metrics = [
            "rotamer recovery",
            "rmsd",
        ]

        if self.predict_bin_chi:
            metrics.append(
                "chi nll loss"
            )
            if self.predict_offset:
                metrics.append(
                    "offset mse loss"
                )
        else:
            metrics.append(
                "chi mse loss"
            )

        return metrics

    @property
    def monitor_metric(self) -> str:
        if self.predict_bin_chi:
            return "val chi nll loss mean"
        else:
            return "val chi mse loss mean"

    def compute_loss(self, output, batch, use_sc_bf_mask=False, _return_breakdown=False, _logger=BlackHole(), _log_prefix="train"):
        # Update side chain chi mask based on b-factors, if necessary
        if use_sc_bf_mask:
            SC_D_mask = batch.SC_D_mask * batch.SC_D_BF_mask
        else:
            SC_D_mask = batch.SC_D_mask
        
        loss_fns = {
            "rotamer_recovery": lambda: rotamer_recovery_from_coords(
                batch.S, batch.SC_D, output['final_X'], 
                batch.residue_mask, SC_D_mask,
                _metric=_logger.get_metric(_log_prefix + " rotamer recovery")),
            "rmsd_loss": lambda: sc_rmsd(
                output['final_X'], batch.X, batch.S, 
                batch.X_mask, batch.residue_mask,
                _metric=_logger.get_metric(_log_prefix + " rmsd"))
        }
        
        if self.predict_bin_chi:
            loss_fns.update({
                "chi_nll_loss": lambda: nll_chi_loss(
                    output["chi_log_probs"], batch.SC_D_bin,
                    batch.S, SC_D_mask,
                    _metric=_logger.get_metric(_log_prefix + " chi nll loss"))})
            if self.predict_offset:
                loss_fns.update({
                    "offset_mse_loss": lambda: offset_mse(
                        output["chi_bin_offset"], batch.SC_D_bin_offset,
                        SC_D_mask, self.n_chi_bins, False,
                        _metric=_logger.get_metric(_log_prefix + " offset mse loss"))})
        else:
            loss_fns.update({
                "chi_mse_loss": lambda: supervised_chi_loss(
                    output["norm_chi"], output["unnorm_chi"],
                    batch.S, batch.residue_index,
                    SC_D_mask, batch.SC_D_sincos,
                    1.0, 0.1,
                    _metric=_logger.get_metric(_log_prefix + " chi mse loss"))})
            
        total_loss = 0.
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.loss.get(loss_name + "_weight", 0.0)
            loss = loss_fn()
            if (torch.isnan(loss) or torch.isinf(loss)):
                self.log.warning(f"{loss_name} loss is NaN. Skipping...")
                loss = loss.new_tensor(0., requires_grad=True)
            total_loss = total_loss + weight * loss
            losses[loss_name] = loss.detach().cpu().clone()
        
        if not _return_breakdown:
            return total_loss
        
        return total_loss, losses

    def forward(self, batch, n_recycle=0):
        
        # Add empty previous prediction
        prevs = {
            "pred_X": torch.zeros_like(batch.X),
            "pred_SC_D": torch.zeros_like(batch.SC_D),
            "pred_SC_D_probs": torch.zeros((*batch.S.shape, 4, self.n_chi_bins + 1), device=batch.S.device),
        }
        
        with torch.no_grad():
            # Loop over all recycle iterations
            for _ in range(n_recycle):
                
                outputs = self.single_forward(batch, prevs)
                
                # Create coordinates for prediction
                if self.predict_bin_chi:
                    chi_pred = self._chi_prediction_from_probs(outputs['chi_probs'], outputs.get('chi_bin_offset', None), strategy=self.recycle_strategy)
                else:
                    chi_pred = outputs['norm_chi'] 
                    chi_pred = torch.atan2(chi_pred[..., 0], chi_pred[..., 1])
                aatype_chi_mask = torch.tensor(rc.chi_mask_atom14, dtype=torch.float32, device=chi_pred.device)[batch.S]  
                chi_pred = aatype_chi_mask * chi_pred
                atom14_xyz = get_atom14_coords(batch.X, batch.S, batch.BB_D, chi_pred)
                
                # Update previous predictions
                if self.recycle_X:
                    prevs["pred_X"] = atom14_xyz
                prevs["pred_SC_D"] = chi_pred
                prevs["pred_SC_D_probs"] = outputs.get("chi_probs", None)
                
        # Final prediction
        outputs = self.single_forward(batch, prevs)
        
        # Create coordinates for prediction
        if self.predict_bin_chi:
            chi_pred = self._chi_prediction_from_probs(outputs['chi_probs'], outputs.get('chi_bin_offset', None))
        else:
            chi_pred = outputs['norm_chi']    
            chi_pred = torch.atan2(chi_pred[..., 0], chi_pred[..., 1])
        aatype_chi_mask = torch.tensor(rc.chi_mask_atom14, dtype=torch.float32, device=chi_pred.device)[batch.S]  
        chi_pred = aatype_chi_mask * chi_pred
        atom14_xyz = get_atom14_coords(batch.X, batch.S, batch.BB_D, chi_pred) 
                    
        # Add final predictions to outputs
        outputs['final_SC_D'] = chi_pred
        outputs['final_X'] = atom14_xyz
            
        return outputs
    
    def single_forward(self, batch, prevs):
        """ Graph-conditioned sequence model """
        # Unpack batch
        X = torch.cat((batch.X[..., :4, :], prevs['pred_X'][..., 4:, :]), dim=-2)
        S = batch.S
        BB_D = batch.BB_D_sincos
        mask = batch.residue_mask
        residue_index = batch.residue_index

        # Embed initial features
        V, E, E_idx, X = self.features(X, S, BB_D, mask, residue_index)
        h_V = self.W_v(V)
        h_E = self.W_e(E)
        
        # Update with recycled predictions
        if self.recycle_SC_D_sc:
            pred_SC_D_sc = torch.stack((torch.sin(prevs['pred_SC_D']), torch.cos(prevs['pred_SC_D'])), dim=-1)
            h_V = h_V + self.W_recycle_SC_D_sc(pred_SC_D_sc.view(*pred_SC_D_sc.shape[:-2], -1))
        if self.recycle_SC_D_probs:
            h_V = h_V + self.W_recycle_SC_D_probs(prevs['pred_SC_D_probs'].view(*prevs['pred_SC_D_probs'].shape[:-2], -1))

        # Update with recycled predictions
        if self.recycle_SC_D_sc:
            pred_SC_D_sc = torch.stack((torch.sin(prevs['pred_SC_D']), torch.cos(prevs['pred_SC_D'])), dim=-1)
            h_V = h_V + self.W_recycle_SC_D_sc(pred_SC_D_sc.view(*pred_SC_D_sc.shape[:-2], -1))
        if self.recycle_SC_D_probs:
            h_V = h_V + self.W_recycle_SC_D_probs(prevs['pred_SC_D_probs'].view(*prevs['pred_SC_D_probs'].shape[:-2], -1))

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.mpnn_layers:
            if torch.is_grad_enabled():
                if self.use_ipmp or self.use_ipmp_ipa:
                    h_V, h_E = checkpoint(layer, h_V, h_E, E_idx, X, mask, mask_attend)
                else:
                    h_V, h_E = checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)
            else:
                if self.use_ipmp or self.use_ipmp_ipa:
                    h_V, h_E = layer(h_V, h_E, E_idx, X, mask, mask_attend)
                else:
                    h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        outputs = {}
        # One-hot encoded sequence for node features
        h_S = self.W_seq(S)
        h_VS = torch.cat((h_V, h_S), -1)
        if not self.predict_bin_chi:
            unnorm_chi = self.W_out_chi(h_VS)
            unnorm_chi = unnorm_chi.view(X.shape[0], X.shape[1], 4, 2)
            
            # Normalize chi outputs
            norm_denom = torch.sqrt(torch.clamp(torch.sum(unnorm_chi ** 2, dim=-1, keepdim=True), min=1e-12))
            norm_chi = unnorm_chi / norm_denom
            outputs['unnorm_chi'] = unnorm_chi
            outputs['norm_chi'] = norm_chi
        else:
            CH_logits = self.W_out_chi(h_VS).view(h_V.shape[0], h_V.shape[1], 4, -1)
            chi_log_probs = F.log_softmax(CH_logits, dim=-1)
            chi_probs = F.softmax(CH_logits, dim=-1)
            outputs['chi_log_probs'] = chi_log_probs
            outputs['chi_probs'] = chi_probs
            outputs['chi_logits'] = CH_logits
                
        if self.predict_offset:
            offset = (2 * torch.pi / self.n_chi_bins) * torch.sigmoid(self.offset_layer(h_V))
            outputs['chi_bin_offset'] = offset
            
        return outputs

    def sample(self, batch, temperature=1.0, n_recycle=0):
        
        # Add empty previous prediction
        prevs = {
            "pred_X": torch.zeros_like(batch.X),
            "pred_SC_D": torch.zeros_like(batch.SC_D),
            "pred_SC_D_probs": torch.zeros((*batch.S.shape, 4, self.n_chi_bins + 1), device=batch.S.device),
        }
        
        with torch.no_grad():
            # Loop over all recycle iterations
            for _ in range(n_recycle):
                
                sample_out = self.single_sample(batch, prevs, temperature)
                
                # Create coordinates for prediction
                if self.predict_bin_chi:
                    chi_pred = self._chi_prediction_from_probs(sample_out['chi_probs'], sample_out['chi_bin_offset'], strategy=self.recycle_strategy)
                else:
                    chi_pred = sample_out['norm_chi']
                    chi_pred = torch.atan2(chi_pred[..., 0], chi_pred[..., 1])
                aatype_chi_mask = torch.tensor(rc.chi_mask_atom14, dtype=torch.float32, device=chi_pred.device)[batch.S]  
                chi_pred = aatype_chi_mask * chi_pred
                atom14_xyz = get_atom14_coords(batch.X, batch.S, batch.BB_D, chi_pred)     
                           
                # Update previous predictions
                if self.recycle_X:
                    prevs["pred_X"] = atom14_xyz
                prevs["pred_SC_D"] = chi_pred
                prevs["pred_SC_D_probs"] = sample_out.get("chi_probs", None)
                
            # Final prediction
            sample_out = self.single_sample(batch, prevs, temperature)
            
            # Create coordinates for prediction
            if self.predict_bin_chi:
                chi_pred = self._chi_prediction_from_probs(sample_out['chi_probs'], sample_out['chi_bin_offset'])
            else:
                chi_pred = sample_out['norm_chi']
                chi_pred = torch.atan2(chi_pred[..., 0], chi_pred[..., 1])
            aatype_chi_mask = torch.tensor(rc.chi_mask_atom14, dtype=torch.float32, device=chi_pred.device)[batch.S]  
            chi_pred = aatype_chi_mask * chi_pred
            atom14_xyz = get_atom14_coords(batch.X, batch.S, batch.BB_D, chi_pred)   
                        
            # Add final predictions to outputs
            sample_out['final_SC_D'] = chi_pred
            sample_out['final_X'] = atom14_xyz
            
        return sample_out

    def single_sample(self, batch, prevs, temperature=1.0):
        """ Autoregressive decoding of a model """
        # Unpack batch
        X = torch.cat((batch.X[..., :4, :], prevs['pred_X'][..., 4:, :]), dim=-2)
        S = batch.S
        BB_D = batch.BB_D_sincos
        mask = batch.residue_mask
        residue_index = batch.residue_index
        
        # Prepare node and edge embeddings
        V, E, E_idx, X = self.features(X, S, BB_D, mask, residue_index)
        h_V = self.W_v(V)
        h_E = self.W_e(E)
        
        # Update with recycled predictions
        if self.recycle_SC_D_sc:
            pred_SC_D_sc = torch.stack((torch.sin(prevs['pred_SC_D']), torch.cos(prevs['pred_SC_D'])), dim=-1)
            h_V = h_V + self.W_recycle_SC_D_sc(pred_SC_D_sc.view(*pred_SC_D_sc.shape[:-2], -1))
        if self.recycle_SC_D_probs:
            h_V = h_V + self.W_recycle_SC_D_probs(prevs['pred_SC_D_probs'].view(*prevs['pred_SC_D_probs'].shape[:-2], -1))

        # Update with recycled predictions
        if self.recycle_SC_D_sc:
            pred_SC_D_sc = torch.stack((torch.sin(prevs['pred_SC_D']), torch.cos(prevs['pred_SC_D'])), dim=-1)
            h_V = h_V + self.W_recycle_SC_D_sc(pred_SC_D_sc.view(*pred_SC_D_sc.shape[:-2], -1))
        if self.recycle_SC_D_probs:
            h_V = h_V + self.W_recycle_SC_D_probs(prevs['pred_SC_D_probs'].view(*prevs['pred_SC_D_probs'].shape[:-2], -1))

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.mpnn_layers:
            if self.use_ipmp or self.use_ipmp_ipa:
                h_V, h_E = layer(h_V, h_E, E_idx, X, mask, mask_attend)
            else:
                h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        h_S = self.W_seq(S)
        h_VS = torch.cat((h_V, h_S), dim=-1)
            
        # Chi prediction
        if not self.predict_bin_chi:
            chi_mask = torch.tensor(rc.chi_angles_mask + [[0.0, 0.0, 0.0, 0.0]], device=X.device)[S].unsqueeze(-1)
            unnorm_chi = self.W_out_chi(h_VS)
            unnorm_chi = unnorm_chi.view(X.shape[0], X.shape[1], 4, 2)
            
            # Normalize chi outputs
            norm_denom = torch.sqrt(torch.clamp(torch.sum(unnorm_chi ** 2, dim=-1, keepdim=True), min=1e-12))
            norm_chi = unnorm_chi / norm_denom
            
            # Mask the chi outputs
            unnorm_chi = chi_mask * unnorm_chi
            norm_chi = chi_mask * norm_chi
        else:
            chi_mask = torch.tensor(rc.chi_angles_mask + [[0.0, 0.0, 0.0, 0.0]], device=X.device)[S].unsqueeze(-1)
            h_VS = torch.cat([h_V, h_S], dim=-1)
            if temperature > 0.0:
                CH_logits = self.W_out_chi(h_VS).view(h_V.shape[0], h_V.shape[1], 4, -1) / temperature
                chi_probs = F.softmax(CH_logits, dim=-1)
                CH = torch.multinomial(chi_probs.view(-1, CH_logits.shape[-1]), 1).view(CH_logits.shape[0], CH_logits.shape[1], CH_logits.shape[2], -1).squeeze(-1)
            else:
                CH_logits = self.W_out_chi(h_VS).view(h_V.shape[0], h_V.shape[1], 4, -1)
                chi_probs = F.softmax(CH_logits, dim=-1)
                CH = torch.argmax(chi_probs, dim=-1)

        if self.predict_offset:
            offset = (2 * torch.pi / self.n_chi_bins) * torch.sigmoid(self.offset_layer(h_V))

        output = {
            'norm_chi': norm_chi if not self.predict_bin_chi else None,
            'unnorm_chi': unnorm_chi if not self.predict_bin_chi else None,
            'chi_bin': CH if self.predict_bin_chi else None,
            'chi_probs': chi_probs if self.predict_bin_chi else None,
            'chi_bin_offset': offset if self.predict_offset else None,
            'chi_logits': CH_logits if self.predict_bin_chi else None
        }

        return output


class PIPPackFineTune(PIPPack):
    def __init__(self, gumbel_temp=1.0, **kwargs):
        self.gumbel_temp = gumbel_temp
        super().__init__(**kwargs)
        
    @property
    def metric_names(self) -> Sequence[str]:
        metrics = [
            'rotamer recovery',
            'rmsd',
            'clash loss',
            'proline loss'
        ]
        
        if self.predict_bin_chi:
            metrics.append(
                'chi nll loss'
            )
            if self.predict_offset:
                metrics.append(
                    'offset mse loss'
                )
        else:
            metrics.append(
                'chi mse loss'
            )
            
        return metrics
    
    def _gumbel_sample_from_logits(self, chi_logits, chi_bin_offset=None):
        # Sample from Gumbel-Softmax distribution
        gumbel_chi_bin = F.gumbel_softmax(chi_logits, self.gumbel_temp, hard=True)
        
        # Determine actual chi value from bin
        chi_bin_rad = torch.cat((torch.arange(-torch.pi, torch.pi, 2 * torch.pi / self.n_chi_bins, device=chi_logits.device), torch.tensor([0]).to(device=chi_logits.device)))
        pred_chi_bin = torch.sum(chi_bin_rad.view(*([1] * (len(chi_logits.shape) - 1)), -1) * gumbel_chi_bin, dim=-1)

        # Add bin offset
        if self.predict_offset and chi_bin_offset is not None:
            bin_sample_update = chi_bin_offset
        else:
            bin_sample_update = (2 * torch.pi / self.n_chi_bins) * torch.rand(chi_logits.shape, device=chi_logits.device)
        sampled_chi = pred_chi_bin + bin_sample_update
        
        return sampled_chi
    
    def forward(self, batch, n_recycle=0):
        outputs = super().forward(batch, n_recycle)
        
        # Add a gumbel sample to outputs
        gumbel_sample = self._gumbel_sample_from_logits(outputs['chi_logits'], outputs.get('chi_bin_offset', None))
        aatype_chi_mask = torch.tensor(rc.chi_mask_atom14, dtype=torch.float32, device=gumbel_sample.device)[batch.S]
        chi_pred = aatype_chi_mask * gumbel_sample
        
        atom14_xyz = get_atom14_coords(batch.X, batch.S, batch.BB_D, chi_pred)
        outputs['gumbel_SC_D'] = chi_pred
        outputs['gumbel_X'] = atom14_xyz
        
        return outputs
    
    def compute_loss(self, output, batch, use_sc_bf_mask=False, _return_breakdown=False, _logger=BlackHole(), _log_prefix="train"):
        # Update side chain chi mask based on b-factors, if necessary
        if use_sc_bf_mask:
            SC_D_mask = batch.SC_D_mask * batch.SC_D_BF_mask
        else:
            SC_D_mask = batch.SC_D_mask
        
        loss_fns = {
            "rotamer_recovery": lambda: rotamer_recovery_from_coords(
                batch.S, batch.SC_D, output['final_X'], 
                batch.residue_mask, SC_D_mask,
                _metric=_logger.get_metric(_log_prefix + " rotamer recovery")),
            "rmsd_loss": lambda: sc_rmsd(
                output['final_X'], batch.X, batch.S,
                batch.X_mask, batch.residue_mask,
                _metric=_logger.get_metric(_log_prefix + " rmsd")),
            # Fine-tuning losses on Gumbel sample
            "clash_loss": lambda: local_interresidue_sc_clash_loss(
                batch, output['gumbel_X'], 0.6,
                _metric=_logger.get_metric(_log_prefix + " clash loss"))["mean_loss"],
            "proline_loss": lambda: unclosed_proline_loss(
                batch, output['gumbel_X'],
                _metric=_logger.get_metric(_log_prefix + " proline loss"))
        }
        
        if self.predict_bin_chi:
            loss_fns.update({
                "chi_nll_loss": lambda: nll_chi_loss(
                    output["chi_log_probs"], batch.SC_D_bin,
                    batch.S, SC_D_mask,
                    _metric=_logger.get_metric(_log_prefix + " chi nll loss"))})
            if self.predict_offset:
                loss_fns.update({
                    "offset_mse_loss": lambda: offset_mse(
                        output["chi_bin_offset"], batch.SC_D_bin_offset,
                        SC_D_mask, self.n_chi_bins, False,
                        _metric=_logger.get_metric(_log_prefix + " offset mse loss"))})
        else:
            loss_fns.update({
                "chi_mse_loss": lambda: supervised_chi_loss(
                    output["norm_chi"], output["unnorm_chi"],
                    batch.S, batch.residue_index,
                    SC_D_mask, batch.SC_D_sincos,
                    1.0, 0.1,
                    _metric=_logger.get_metric(_log_prefix + " chi mse loss"))})
            
        total_loss = 0.
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.loss.get(loss_name + "_weight", 0.0)
            loss = loss_fn()
            if (torch.isnan(loss) or torch.isinf(loss)):
                self.log.warning(f"{loss_name} loss is NaN. Skipping...")
                loss = loss.new_tensor(0., requires_grad=True)
            total_loss = total_loss + weight * loss
            losses[loss_name] = loss.detach().cpu().clone()
            
        if not _return_breakdown:
            return total_loss
        
        return total_loss, losses