import torch
import torch.nn as nn
from torch.nn import Linear, ReLU
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GINEConv,global_add_pool,global_mean_pool
from torch_geometric.loader import DataLoader
from torch.optim import Adam, lr_scheduler
from torch_scatter import scatter_add
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from itertools import chain
import numpy as np
import math
import random

from Model import contrastive_learning
from Args import parse_args
from Model import motif_embedding
from motif_extract import mol_motif, motif_graph
import motif_spilit
from Model.atom_motif_attention import AtomMotifAttention
from Model.HMSAF import HMSAF
from Model import utils

import importlib
importlib.reload(utils)
importlib.reload(motif_spilit)
importlib.reload(motif_embedding)
importlib.reload(mol_motif)
importlib.reload(motif_graph)

def global_atom_attr(data):
    motif_batch = data['motif'].batch    # 每个 motif 所属的图索引
    atom_ptr = data['atom'].ptr          # 每个图的 atom 索引起点

    # 获取边的连接信息和属性
    edge_index = data['motif', 'connects', 'motif'].edge_index
    edge_attr = data['motif', 'connects', 'motif'].edge_attr

    # 确定每条边所属的图
    src_motif = edge_index[0]  # 源 motif 索引
    graph_idx = motif_batch[src_motif]  # 每条边所属的图索引

    # 从 edge_attr 中提取本地 atom 索引
    local_atom_idx_src = edge_attr[:, -1]  # 源节点的本地索引
    local_atom_idx_dst = edge_attr[:, -2]  # 目标节点的本地索引

    # 计算全局 atom 索引
    offsets = atom_ptr[graph_idx]  # 每条边对应图的 atom 索引偏移
    global_atom_idx_src = local_atom_idx_src + offsets
    global_atom_idx_dst = local_atom_idx_dst + offsets

    # 构建新的 edge_attr：特征 + 全局 atom 索引 + 边的全局索引
    new_edge_attr = torch.stack([global_atom_idx_src, global_atom_idx_dst],dim=1)
    return new_edge_attr



# Message Passing of motifs
class GINENet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim):
        super(GINENet, self).__init__()
        self.conv1 = GINEConv(nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        ), edge_dim=edge_dim)

        self.conv2 = GINEConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        ), edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        return x


# Heterogeneous GNN for encoding motifs
class MotifGIN(torch.nn.Module):
    def __init__(self, node_dim,edge_dim,type_dim,hidden_dim,Pair_MLP,gnn_type,num_layers=2):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.type_dim = type_dim

        # Initial embedding layers
        self.atom_encoder = nn.Sequential(
                            Linear(node_dim, hidden_dim),
                            ReLU())

        self.edge_encoder = nn.Sequential(
                            Linear(edge_dim, hidden_dim),
                            ReLU())

        self.motif_type_encoder = nn.Embedding(4, type_dim)  # Assuming 4 motif types

        # Internal motif processing using MotifGINLayer :TODO:motif_1
        self.motif_GIN = motif_embedding.GNNModel(hidden_dim, num_layers, Pair_MLP = Pair_MLP,gnn_type = gnn_type)

        self.motif_node_nn = nn.Linear(hidden_dim + type_dim,hidden_dim)

        self.motif_edge_nn = nn.Linear(hidden_dim * 2 + type_dim,hidden_dim)

        self.motif_Message_Passing = GINENet(in_channels=hidden_dim,
                                             hidden_channels=hidden_dim,
                                             out_channels=hidden_dim,
                                             edge_dim=hidden_dim)

    def forward(self, data):
        # # ==================Prepare embeddings# ==================
        # Encode atoms
        x_atom = self.atom_encoder(data['atom'].x.float())

        # Encode motifs:环与官能团类型编码
        motif_type = data['motif'].type
        motif_type_embedding = self.motif_type_encoder(motif_type)

        # Encode edges - atom-atom edges
        edge_index = data[("atom", "motif_internal", "atom")].edge_index
        edge_attr= self.edge_encoder(data[('atom', 'motif_internal', 'atom')].edge_attr.float())

        # ==================encoding node of motifs: type of motif and edge and atom====================
        # input:x, edge_index, edge_attr, motif_atom_edge_index
        motif_atom_edge_index = data["motif", "contains", "atom"].edge_index

        node_alpha,pair_alpha,h_motif_atom, x = self.motif_GIN(x_atom, edge_index, edge_attr, motif_atom_edge_index)
        h_motif_atom = self.motif_node_nn(torch.cat((motif_type_embedding, h_motif_atom), dim=1))

        # ==================encoding edge of motifs: type of motif and edge and atom====================
        motif_edge_attr = torch.cat([data["motif", "connects", "motif"].edge_attr[:,:-2],global_atom_attr(data)],dim=1)    # 用于修改edge_attr中的node（local->全局）
        atom_edge_dim = data["atom", "motif_internal", "atom"].edge_attr.shape[1]


        # 1.type of motif：用于motif-graph边的编码
        src_motif_type = motif_edge_attr[:,0].long()
        dis_motif_type = motif_edge_attr[:,1].long()
        couple_motifs_type = self.motif_type_encoder(src_motif_type) + self.motif_type_encoder(dis_motif_type)

        # 2.edge and atom
        node_indices = motif_edge_attr[:,-2:].long()
        node_embeddings = torch.index_select(x_atom, 0, node_indices[:,0]) + torch.index_select(x_atom, 0, node_indices[:,1])
        edge_embeddings = self.edge_encoder(motif_edge_attr[:,2:2 + atom_edge_dim].float())

        # Combine embeddings：
        h_motif_edge_attr = self.motif_edge_nn(torch.cat([couple_motifs_type, edge_embeddings, node_embeddings], dim=1))

        # ==================Message Passing of motifs:h_motif_atom and h_motif_edge_attr====================
        motif_edge_index = data["motif", "connects", "motif"].edge_index
        h_motif_atom = self.motif_Message_Passing(h_motif_atom,motif_edge_index, h_motif_edge_attr)

        # ============================Pool of motif=============================
        motif_level = global_add_pool(h_motif_atom,batch = data["motif"].batch)

        return node_alpha,pair_alpha,h_motif_atom , x_atom , motif_level


class MotifBasedModel(torch.nn.Module):
    def __init__(self,node_feature_dim,edge_feature_dim,hidden_dim,y_dim,Pair_MLP=True,gnn_type="our"):
        super(MotifBasedModel, self).__init__()

        self.atom_encoder = nn.Sequential(Linear(node_feature_dim, hidden_dim),ReLU())

        # Heterogeneous GNN for joint atom-motif representation learning
        self.motif_gin = MotifGIN(node_feature_dim, edge_feature_dim, 16,hidden_dim, Pair_MLP=Pair_MLP,gnn_type = gnn_type,num_layers=2)

        # atom_motif_attention
        self.atom_motif_attn = AtomMotifAttention(atom_dim=hidden_dim, motif_dim=hidden_dim,num_heads = 4,dropout = 0.2)

        self.DCM_attention = HMSAF(n_head = 4,
                                  input_dim = hidden_dim,
                                  output_dim = hidden_dim,
                                  use_head_interaction = True,
                                  use_gating = True)

        # 添加层归一化以稳定训练
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # 修改输出层为二分类
        self.motif_read_out = nn.Sequential(nn.Linear(hidden_dim, y_dim),
                                            # nn.Sigmoid()
                                            )
        self.atom_read_out = nn.Sequential(nn.Linear(hidden_dim, y_dim),
                                            # nn.Sigmoid()
                                           )

        # 初始alpha和beta权重为0.5，可学习
        # 共享权重的多层感知机
        reduction_ratio = 4
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, (hidden_dim + hidden_dim) // reduction_ratio),
            nn.ReLU(),
            nn.Linear((hidden_dim + hidden_dim) // reduction_ratio, hidden_dim),
            nn.Sigmoid()  # 输出[0,1]的通道权重
        )

        # # 添加额外的融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, data):
        # -------------------------------
        # 1. compute embedding of motif
        # -------------------------------
        node_alpha,pair_alpha,h_motif_atom, x_atom, motif_level = self.motif_gin(data)

        # 保存初始特征用于残差连接
        # atom_initial = x_atom

        # 更新图数据
        data["motif"].x = h_motif_atom
        # TODO 1:修改了这一处：如果使用计算motif_embedding的继续使用，可能会出问题
        # data["atom"].x = x_atom                                      # 改进前：
        data["atom"].x = self.atom_encoder(data["atom"].x.float())     # 改进后：


        # -------------------------------
        # Stage 2: 粗粒度特征 - atom-motif注意力
        # -------------------------------
        hetero_data = self.atom_motif_HeteroData(data)
        coarse_particle_feature, motif_to_atom_attn = self.atom_motif_attn.get_atom_to_atom_attention_efficient(hetero_data)
        coarse_particle_feature = self.layer_norm1(coarse_particle_feature)           # 应用层归一化


        # -------------------------------
        # Stage 3: 细粒度特征 - 节点间动态注意力
        # -------------------------------
        fine_particle_feature,attn_probs = self.DCM_attention(data["atom"].x, data["atom"].batch, motif_to_atom_attn)
        fine_particle_feature = self.layer_norm2(fine_particle_feature)               # 应用层归一化


        # -------------------------------
        # Stage 4: 融合粗细粒度特征:
        # -------------------------------
        # 方法1: 门控机制
        # TODO 2:修改门阀控制
        combined = torch.cat([coarse_particle_feature, fine_particle_feature], dim=-1)
        # 生成通道注意力权重
        channel_weights = self.mlp(combined)
        # 加权融合
        atom = channel_weights * coarse_particle_feature + (1 - channel_weights) * fine_particle_feature
        # 额外的特征融合
        atom = self.fusion(atom)


        # 读出层
        atom_level = global_mean_pool(atom, data["atom"].batch)    # 表示每个分子图的高维特征表示

        y_atom = self.atom_read_out(atom_level)
        y_motif = self.motif_read_out(motif_level)

        return y_atom, y_motif, h_motif_atom,{'gate': channel_weights.mean().item(),"atom_level":atom_level,"node_alpha":node_alpha,"pair_alpha":pair_alpha,"data":data,"attn_probs":attn_probs}

    def atom_motif_HeteroData(self,data):
        hetero_data = HeteroData()

        # Add atom features
        hetero_data['atom'].x = data['atom'].x
        hetero_data['atom'].batch = data['atom'].batch

        # Add motif features
        hetero_data['motif'].x = data['motif'].x
        hetero_data['motif'].batch = data['motif'].batch

        # Add atom-in-motif edges
        hetero_data['atom', 'in', 'motif'].edge_index = data['atom', 'in', 'motif'].edge_index
        return hetero_data

def compute_motif_contrastive_loss(batch, temperature=0.1, eps=1e-8):
    """
    只对 ring5 和 ring6 的 motif 进行结构感知对比学习损失。
    """
    z = batch["motif"].x  # [N, D]
    labels = batch["mol"].y[batch["motif"].batch]  # [N]
    type_list = batch["motif"].type  # [N]

    # 每个 motif 的原子数量
    edge_index = batch["motif", "contains", "atom"].edge_index
    motif_indices = edge_index[0]
    atoms_per_motif = scatter_add(torch.ones_like(motif_indices), motif_indices, dim=0)  # [N]

    # 构造 motif_type
    type_prefix = {0: 'ring', 1: 'non-cycle', 2: 'chain', 3: 'other'}
    motif_type = [f"{type_prefix[int(t)]}{int(n.item())}" for t, n in zip(type_list, atoms_per_motif)]
    type_class = [type_prefix[int(t)] for t in type_list]

    # 只保留 ring5 和 ring6
    allowed_types = {'ring5', 'ring6'}
    valid_mask = [m in allowed_types for m in motif_type]
    valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=z.device)

    # 如果没有足够样本，直接返回 0
    if valid_mask.sum() < 2:
        return torch.tensor(0.0, device=z.device)

    # 过滤有效样本
    z = z[valid_mask]
    labels = labels[valid_mask]
    motif_type = [m for i, m in enumerate(motif_type) if valid_mask[i]]
    type_class = [t for i, t in enumerate(type_class) if valid_mask[i]]

    # 转张量
    labels = labels.view(-1, 1)
    motif_type_tensor = torch.tensor([hash(m) for m in motif_type], device=z.device)
    type_class_tensor = torch.tensor([hash(t) for t in type_class], device=z.device).view(-1, 1)

    # 相似度矩阵
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)

    # 掩码
    N = z.size(0)
    diag_mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
    # 这里使用的是label：
    label_eq = (labels == labels.T)
    label_neq = (labels != labels.T)
    type_eq = (motif_type_tensor.view(-1, 1) == motif_type_tensor.view(1, -1))
    class_eq = (type_class_tensor == type_class_tensor.T)

    pos_mask = label_eq & type_eq & diag_mask
    neg_mask = label_neq & class_eq & diag_mask

    # 对比损失
    pos_sim = sim_matrix.masked_fill(~pos_mask, -1e9)
    neg_sim = sim_matrix.masked_fill(~neg_mask, -1e9)

    # 正样本对
    numerator = torch.exp(pos_sim / temperature).sum(dim=-1)
    # 负样本对
    denominator = numerator + torch.exp(neg_sim / temperature).sum(dim=-1)

    valid = (pos_mask.sum(dim=-1) > 0)
    loss = -torch.log((numerator + eps) / (denominator + eps))

    return loss[valid].mean()


#多标签多分类
def masked_roc_auc(y_true, y_score):
    y_true_np = y_true.cpu().numpy()
    y_score_np = y_score.cpu().numpy()
    aucs = []
    for i in range(y_true_np.shape[1]):
        y_col = y_true_np[:, i]
        p_col = y_score_np[:, i]
        mask = ~np.isnan(y_col)
        if np.sum(mask) >= 2 and len(np.unique(y_col[mask])) > 1:  # 至少有两个类才有意义
            auc = roc_auc_score(y_col[mask], p_col[mask])
            aucs.append(auc)
    return np.mean(aucs) if aucs else float('nan')


# 修改训练函数，添加损失权重动态调整和梯度分析
def train(model, loader, optimizer, criterion, device,argse,check_grad=False, epoch=0):
    model.train()
    total_loss = 0
    atom_loss_total = 0
    motif_loss_total = 0
    gate_values = 0
    alpha_values = 0
    beta_values = 0
    processed_batches = 0

    # 用于计算分类准确率
    total_correct = 0
    total_samples = 0

    # Create progress bar
    pbar = tqdm(loader, desc='Training')

    for batch_idx, batch in enumerate(pbar):
        try:
            optimizer.zero_grad()

            # Move batch to device.
            batch = batch.to(device)

            # Forward pass
            out_atom, out_motif, h_motif_atom,metrics = model(batch)
            y = batch["mol"].y


            # 计算损失
            mask = ~torch.isnan(y)
            loss_atom = criterion(out_atom[mask], y[mask])  # 原子级别的损失
            loss_motif = criterion(out_motif[mask], y[mask])  # 子图级别的损失

            if argse.is_contrastive==True:
                contrastive_loss_ring = contrastive_learning.compute_ring_contrastive_loss(batch,temperature=0.1, eps=1e-8)
                contrastive_loss_noring = contrastive_learning.compute_nonring_contrastive_loss(batch,temperature=0.1,threshold=0.9,eps=1e-8)
                loss = loss_atom + loss_motif + argse.alpha * contrastive_loss_ring +  argse.beta * contrastive_loss_noring
            # 计算对比损失
            else:
                loss = loss_atom + loss_motif

            # 添加L2正则化以防止过拟合
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param, 2)

            loss += 1e-5 * l2_reg  # 轻微的L2正则化

            # 检查损失值是否有效
            if not torch.isfinite(loss):
                print(f"警告: 无效损失值，跳过此批次")
                continue

            loss.backward()

            # 梯度裁剪以防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # 计算分类准确率
            # predictions = torch.sigmoid(out_atom) > 0.5
            # total_correct += (predictions == y).sum().item()
            # total_samples += y.size(0)

            preds = torch.sigmoid(out_atom) >= 0.5
            mask_np = ~torch.isnan(y)
            total_correct += (preds == y)[mask_np].sum().item() / mask.sum().item()
            total_samples += y.size(0)


            # 记录损失
            batch_size = batch.num_graphs
            total_loss += loss.item() * batch_size
            atom_loss_total += loss_atom.item() * batch_size
            motif_loss_total += loss_motif.item() * batch_size

            # 记录门控和权重值:
            gate_values += metrics['gate'] * batch_size
            # alpha_values += metrics['alpha'] * batch_size
            # beta_values += metrics['beta'] * batch_size

            processed_batches += 1

            # 更新进度条信息
            pbar.set_postfix({
                'batch_loss': f'{loss.item():.4f}',
                'atom_loss': f'{loss_atom.item():.4f}',
                'motif_loss': f'{loss_motif.item():.4f}',
                'gate': f'{metrics["gate"]:.4f}',
                'acc': f'{total_correct/total_samples:.4f}'
            })
        except Exception as e:
            print(e)

    if processed_batches > 0:
        avg_samples = total_samples
        return {
            'loss': total_loss / avg_samples,
            'atom_loss': atom_loss_total / avg_samples,
            'motif_loss': motif_loss_total / avg_samples,
            'gate': gate_values / processed_batches,
            'accuracy': total_correct / processed_batches * 100  # 以百分比显示
            # 'alpha': alpha_values / processed_batches,
            # 'beta': beta_values / processed_batches
        }
    else:
        return {'loss': float('inf'), 'atom_loss': float('inf'), 'motif_loss': float('inf'), 'accuracy': 0.0}


# 修改验证函数以匹配新的训练函数格式
@torch.no_grad()
def evaluate(model, loader, criterion, device,end=False):
    model.eval()
    total_loss = 0
    atom_loss_total = 0
    motif_loss_total = 0
    total_samples = 0
    gate_values = 0
    processed_batches = 0
    
    # 用于ROC-AUC计算的真实标签和预测概率
    all_labels = []
    all_probs = []
    roc_auc_list = []
    contrastive_data = []

    # 用于其他指标
    atom_level = torch.empty(0,32).to(device)
    y_atom = torch.empty(0,1).to(device)
    
    # 统计正确预测的样本数
    correct_num = 0
    sample_num = 0
    # Create progress bar for evaluation
    pbar = tqdm(loader, desc='Evaluating')

    for batch_idx, batch in enumerate(pbar):
        try:
            batch = batch.to(device)

            out_atom, out_motif, _, metrics = model(batch)
            y = batch["mol"].y

            mask = ~torch.isnan(y)
            loss_atom = criterion(out_atom[mask], y[mask])  # 原子级别的损失
            loss_motif = criterion(out_motif[mask], y[mask])  # 子图级别的损失
            loss = loss_atom * 0.5 + loss_motif * 0.5

            # 记录总损失
            batch_size = batch.num_graphs
            total_loss += loss.item() * batch_size
            atom_loss_total += loss_atom.item() * batch_size
            motif_loss_total += loss_motif.item() * batch_size

            # 计算预测概率和准确率
            # probs = torch.sigmoid(out_atom)
            # predictions = (probs > 0.5).float()
            # correct += (predictions == y).sum().item()
            #
            # # 收集用于计算ROC-AUC的数据
            # all_labels.append(y.cpu().numpy())
            # all_probs.append(probs.cpu().numpy())
            # 三元组：索引,x,y
            # if end == True:
            #     for smile,X in zip(batch["mol"].smiles,metrics["atom_level"]):
            #         indices = utils.find_substructure_indices(smile,["c1ccncc1","c1ccccc1"])
            #         if len(indices) == 0:
            #             contrastive_data.append((None,None))
            #         else:
            #             contrastive_data.append((indices[0],X))


            preds = torch.sigmoid(out_atom) >= 0.5
            mask_np = ~torch.isnan(y)
            correct_num += (preds == y)[mask_np].sum().item()
            sample_num  += mask_np.sum().item()

            roc_auc = masked_roc_auc(y, out_atom)
            if not np.isnan(roc_auc):
                roc_auc_list.append(roc_auc)

            # 记录门控和权重值
            gate_values += metrics['gate'] * batch_size

            # 记录每一个分子的高维特征表示
            atom_level = torch.cat((atom_level, metrics["atom_level"]), dim = 0)
            y_atom = torch.cat((y_atom, y), dim = 0)

            total_samples += batch_size
            processed_batches += 1

            # Update progress bar
            avg_loss = total_loss / total_samples
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}'
            })

        except Exception as e:
            print(e)

    # 计算整体准确率
    accuracy = correct_num / sample_num * 100  # 以百分比显示
    roc_auc = np.mean(roc_auc_list)


    if processed_batches > 0:
        return total_loss / total_samples, roc_auc, {
            'atom_loss': atom_loss_total / total_samples,
            'motif_loss': motif_loss_total / total_samples,
            'gate': gate_values / processed_batches,
            'accuracy': accuracy,
            "atom_level": atom_level,
            "y_atom": y_atom,
            "contrastive_data":contrastive_data
        }
    else:
        return float('inf'), 0.0, {}


# 修改验证函数以匹配新的训练函数格式
@torch.no_grad()
def evaluate_total(model, loader, device):
    model.eval()

    # 用于其他指标
    h_motif = torch.empty(0, 32).to(device)
    y_motif = torch.empty(0,1).to(device)
    filtered_smiles_motif = []
    attn_probs = []
    # Create progress bar for evaluation
    pbar = tqdm(loader, desc='Evaluating')

    for batch_idx, batch in enumerate(pbar):
        try:
            batch = batch.to(device)

            out_atom, out_motif, _, metrics = model(batch)
            y = batch["mol"].y

            # 不考虑非环：
            ring = metrics["data"]["motif"].type==0
            ring_np = ring.cpu().numpy()

            # 用在对比学习
            h_motif = torch.cat((h_motif, metrics["data"]["motif"].x[ring]), dim = 0)
            y_motif = torch.cat((y_motif, y[metrics["data"]["motif"].batch][ring]), dim = 0)
            smiles_motif = list(chain(*metrics["data"]["motif"].smiles))
            filtered_smiles_motif += [smiles_motif[i] for i in range(len(smiles_motif)) if ring_np[i]]

            # 分解注意力:
            attn_probs.append(metrics["attn_probs"].mean(dim=0))
            # attn_probs = metrics["attn_probs"].mean(dim=0)
            # attn_probs_split = split_attention_by_molecule(attn_probs,batch["atom"].batch)

        except Exception as e:
            print(e)

    return {"h_motif":h_motif,"y_motif":y_motif,"filtered_smiles_motif":filtered_smiles_motif,"attn_probs":attn_probs}


def split_attention_by_molecule(attn_matrix, batch):
    unique_molecules = torch.unique(batch)

    # 存储每个分子的注意力得分矩阵
    molecule_attentions = []

    for mol_idx in unique_molecules:
        # 找到属于当前分子的所有原子索引
        atom_indices = torch.where(batch == mol_idx)[0]

        # 提取分子内的注意力得分矩阵
        mol_attention = attn_matrix[atom_indices][:, atom_indices]

        # 添加到结果列表
        molecule_attentions.append(mol_attention)

    return molecule_attentions

def compute_pos_weight(dataset):
    labels = []
    for data in dataset:
        labels.append(data["mol"].y)

    labels = torch.cat(labels, dim=0)
    mask = ~torch.isnan(labels)
    pos = (labels[mask] == 1).sum().float()
    neg = (labels[mask] == 0).sum().float()
    pos_weight = neg / (pos + 1e-8)
    return pos_weight

def Data_loader(argse):
    # Load the heterogeneous graph dataset
    dataset = motif_spilit.MoleculeMotifDataset(root="./dataset/", name=argse.dataset)
    print(f"Dataset contains {len(dataset)} molecules")
    # Check if the processed dataset has the correct format
    sample_data = dataset[0]
    print("Checking first graph in dataset:")
    print(f"Motif types: {sample_data['motif'].type}")

    n = len(dataset)
    indices = list(range(n))
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    random.shuffle(indices)

    train_index = indices[:train_size]
    val_index = indices[train_size:train_size + val_size]
    test_index = indices[train_size + val_size:]
    total_index = indices[:]

    print(f"\nDataset split:")
    print(f"Train: {len(train_index)} samples")
    print(f"Validation: {len(val_index)} samples")
    print(f"Test: {len(test_index)} samples")

    # Create dataloaders
    def create_dataloader(indices, batch_size=32, shuffle=True):
        return DataLoader(
            dataset=[dataset[i] for i in indices],
            batch_size=batch_size,
            shuffle=shuffle
        )

    pos_weight = compute_pos_weight(dataset).to(argse.device)

    train_loader = create_dataloader(train_index, batch_size=argse.batch_size, shuffle=True)
    val_loader = create_dataloader(val_index, batch_size=argse.batch_size, shuffle=False)
    test_loader = create_dataloader(test_index, batch_size=argse.batch_size, shuffle=False)
    total_loader = create_dataloader(total_index, batch_size=argse.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader,total_loader,sample_data,pos_weight


def main(train_loader, val_loader, test_loader,total_loader,sample_data,pos_weight,argse,gnn_type="our",Pair_MLP=True):
    node_feature_dim = 9
    edge_feature_dim = 3
    hidden_dim = 32
    HIV_XY = list()
    y_dim = sample_data["mol"].y.shape[1]

    model = MotifBasedModel(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=hidden_dim,
        y_dim=y_dim,
        Pair_MLP=Pair_MLP,
        gnn_type=gnn_type
    ).to(argse.device)

    # 调整优化器，使用较小的学习率
    optimizer = Adam(model.parameters(), lr=argse.lr, weight_decay=argse.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=argse.patience, factor=argse.factor, verbose=True
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training loop
    best_val_loss = float('inf')  # 改回用验证损失来保存最佳模型
    best_val_auc = 0.0  # 仍然记录最佳AUC
    print("\nStarting training...")
    print("-" * 50)

    import matplotlib.pyplot as plt

    # 初始化存储训练和验证结果的列表
    train_losses = []
    val_losses = []
    val_aucs = []  # 修改为AUC
    train_accs = []
    val_accs = []
    learning_rates = []
    test_aucs = []
    scatter_result = []
    for epoch in range(argse.epochs):
        print(f"\nEpoch {epoch + 1}/{argse.epochs}")

        # 每5个epoch执行一次详细梯度检查
        check_grad = (epoch % 10 == 0)
        if check_grad:
            print("将对本轮进行详细梯度检查")

        # 传递epoch参数用于动态调整损失权重
        train_metrics = train(model, train_loader, optimizer, criterion, argse.device, argse,check_grad=check_grad, epoch=epoch)
        train_loss = train_metrics['loss']
        train_acc = train_metrics['accuracy']

        # 评估
        val_loss, val_roc_auc, val_metrics = evaluate(model, val_loader, criterion, argse.device)
        _, test_roc_auc, _ = evaluate(model, test_loader, criterion, argse.device)
        val_acc = val_metrics['accuracy']

        # 调整学习率
        scheduler.step(val_loss)

        # 保存最佳模型（基于验证损失）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_auc_at_best_loss = val_roc_auc
            torch.save(model.state_dict(), f'./best_model/best_model_{argse.dataset}.pt')
            print(f"新的最佳模型已保存! (验证损失: {val_loss:.4f}, 验证AUC: {val_roc_auc:.4f}%)")

        # 记录最佳AUC（仅用于追踪，不保存模型）
        if val_roc_auc > best_val_auc:
            best_val_auc = val_roc_auc

        # 记录结果
        train_losses.append(train_loss if not math.isinf(train_loss) and not math.isnan(train_loss) else None)
        val_losses.append(val_loss)
        val_aucs.append(val_roc_auc)
        test_aucs.append(test_roc_auc)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # 输出更详细的训练摘要
        print(f"Epoch {epoch:03d} 摘要:")
        print(f"  训练损失: {train_loss:.4f} (atom: {train_metrics['atom_loss']:.4f}, motif: {train_metrics['motif_loss']:.4f})")
        print(f"  训练准确率: {train_acc:.4f}%")
        print(f"  验证损失: {val_loss:.4f} (atom: {val_metrics['atom_loss']:.4f}, motif: {val_metrics['motif_loss']:.4f})")
        print(f"  验证ROC_AUC: {val_roc_auc:.4f}%")
        print(f"  验证准确率: {val_acc:.4f}%")
        print(f"  测试ROC_AUC: {test_roc_auc:.4f}%")
        print(f"  门控均值: {train_metrics['gate']:.4f}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)

        # UMAP可视化出来
        # if (epoch+1) % 5==0:
        #     # 需要两个东西：信息嵌入 + label
        #     # "atom_level": atom_level,
        #     # "y_atom": y_atom
        #     _, _, val_metrics = evaluate(model, total_loader, criterion, argse.device,True)
        #     # utils.plot_embeddings(val_metrics["contrastive_data"],method="umap")
        #     X = val_metrics["atom_level"]
        #     Y = val_metrics["y_atom"][:,].flatten()
        #     HIV_XY.append([X,Y])
        #     # utils.task_visual(X, Y, epoch)
        #     # scatter_result[epoch+1] = (X,Y)

    # ==========================================测试最终模型==========================================

    total_metrics = evaluate_total(model, total_loader, argse.device)

    print("\nTesting best model...")
    model.load_state_dict(torch.load(f'./select_parameters/best_model_{argse.dataset}_{i}.pt'))
    test_loss, test_auc, test_metrics = evaluate(model, test_loader, criterion, argse.device)

    print(f"\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test ROC-AUC: {test_auc:.5f}%")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}%")

    # 绘制学习曲线
    plt.figure(figsize=(12, 12))

    # 子图 1: 损失曲线
    plt.subplot(3, 1, 1)
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="orange")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    # 子图 2: 验证 AUC 曲线
    plt.subplot(3, 1, 2)
    plt.plot(val_aucs, label="Validation ROC-AUC", color="red")
    plt.plot(test_aucs, label="Test ROC-AUC", color="green")
    plt.title("Validation ROC-AUC (%)")
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC (%)")
    plt.legend()
    plt.grid()

    # 子图 3: 准确率曲线
    plt.subplot(3, 1, 3)
    plt.plot(train_accs, label="Train Accuracy", color="blue")
    plt.plot(val_accs, label="Validation Accuracy", color="orange")
    plt.title(f"Training and Validation Accuracy (%)//{argse.alpha}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid()

    # 显示图表
    plt.tight_layout()
    plt.show()

    # 保存学习曲线为图片文件
    plt.savefig("./learning_curves.png")
    return test_auc,test_loss,test_metrics['accuracy'],total_metrics

def set_rng_seed(seed):
    random.seed(seed)  # 为 Python 设置随机种子
    np.random.seed(seed)  # 为 NumPy 设置随机种子
    torch.manual_seed(seed)  # 为 PyTorch 设置随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用 GPU，也设置 GPU 的随机种子
    torch.backends.cudnn.deterministic = True  # 禁用 cuDNN 的非确定性算法
    torch.backends.cudnn.benchmark = False  # 禁用 cuDNN 的自动优化
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置 Python 的哈希种子
    # torch.use_deterministic_algorithms(True)  # 强制使用确定性算法


if __name__ == "__main__":

    import os
    set_rng_seed(47)
    argse = parse_args()
    argse.dataset = "BBBP"
    results = dict()
    argse.is_contrastive = True

    train_loader, val_loader, test_loader,total_loader,sample_data, pos_weight = Data_loader(argse)
    test_auc,test_loss,test_acc,total_metrics = main(train_loader, val_loader, test_loader,total_loader,sample_data, pos_weight, argse,Pair_MLP=True)





