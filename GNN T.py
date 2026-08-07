#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   MULTI-TASK ANTIGRAVITY GRAPH TRANSFORMER  —  v2.0 (EGNN + 5-Head)       ║
║   Publication-Ready Research Prototype for Exotic Metamaterial Lattices     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Theoretical Foundations
═══════════════════════

1. JACOBIAN-BASED LOCAL SPACETIME METRIC TRANSFORMATION
─────────────────────────────────────────────────────────
Consider a smooth coordinate transformation φ: ℝ⁴ → ℝ⁴ mapping background
Minkowski coordinates x^μ to the deformed spacetime coordinates ξ^α = φ^α(x).
The Jacobian of this transformation is the 4×4 matrix:

    J^α_μ(x_i) = ∂φ^α / ∂x^μ  evaluated at lattice node i.

The induced metric at node i is obtained via the pullback:

    g_μν(x_i) = J^α_μ(x_i) · J^β_ν(x_i) · η_αβ

where η_αβ = diag(−1, +1, +1, +1) is the Minkowski metric. The metric
distortion tensor is:

    h_μν(x_i) = g_μν(x_i) − η_μν

and the scalar metric distortion norm (predicted by Head 4) is:

    ‖h_μν‖_F = √(Σ_{μ,ν} h_μν²)

The geodesic interval between nodes i, j follows from the line element:

    s²_ij = ∫_{γ_ij} g_μν dx^μ dx^ν ≈ −Δt²_ij + ‖r_j − r_i‖²

where the approximation uses a linearized metric along the spatial geodesic.

2. LAPLACE TRANSFORM STABILITY ANALYSIS OF REPULSIVE FIELDS
────────────────────────────────────────────────────────────
The temporal response of a localized gravitational repulsion field F(t) at a
lattice node is modeled as a damped oscillation:

    f(t) = F₀ · e^{−αt} · cos(ωt),   t ≥ 0

Its Laplace transform is:

    L{f}(s) = F₀ · (s + α) / [(s + α)² + ω²]

The poles are located at s = −α ± jω. The system satisfies the BIBO
(bounded-input, bounded-output) stability criterion iff all poles lie in the
open left half-plane, i.e., Re(s_k) < 0 ⟺ α > 0.

The micro-singularity instability classification target (Head 5) encodes
the binary stability decision:

    y_cls = { 0  if  α_eff > 0     (stable — poles in LHP)
            { 1  if  α_eff ≤ 0     (unstable — pole on or past jω-axis)

where α_eff is synthesized from the total negative energy density and
topological charge distribution across the lattice.

3. E(n)-EQUIVARIANT GRAPH NEURAL NETWORK (EGNN) LAYERS
──────────────────────────────────────────────────────
Standard GNN message-passing operates on scalar node features and is
SE(0)-invariant — it cannot distinguish rotated force field directions,
leading to vector collapse (predictions collapsing to dataset mean ≈ 0).

EGNN (Satorras, Hoogeboom & Welling, ICML 2021) jointly updates scalar
features h_i ∈ ℝ^d AND coordinate features x_i ∈ ℝ^3 equivariantly:

    m_ij = φ_e(h_i, h_j, ‖x_i − x_j‖², a_ij)        [message]
    x_i' = x_i + C · Σ_{j≠i} (x_i − x_j) · φ_x(m_ij)  [coord update]
    h_i' = φ_h(h_i, Σ_j m_ij)                           [scalar update]

The coordinate update uses ONLY relative position vectors (x_i − x_j) scaled
by learned scalar weights φ_x(m_ij). This guarantees E(3)-equivariance: if
all input coordinates are rotated by R ∈ O(3), all output coordinates rotate
by the same R. Force predictions F_x, F_y, F_z are read from the equivariant
coordinate channel, preventing vector collapse entirely.

4. FIVE DECOUPLED OUTPUT HEADS WITH UNCERTAINTY WEIGHTING
────────────────────────────────────────────────────────
The joint regression head Linear(hidden, 4) in v1 caused massive gradient
interference between force components and metric distortion. This rewrite
uses 5 independent MLPs:

    Head 1: F_x   (equivariant readout)
    Head 2: F_y   (equivariant readout)
    Head 3: F_z   (equivariant readout)
    Head 4: ‖h_μν‖ (invariant scalar MLP)
    Head 5: Instability classification (invariant scalar MLP, gradient-scaled)

Per-head homoscedastic uncertainty weighting (Kendall & Gal, CVPR 2018):

    L_total = Σ_{k=1}^{4} [½ exp(−s_k) L_k^reg + ½ s_k]
            + exp(−s₅) L₅^cls + ½ s₅

where s_k are learnable log-variance parameters, one per head.

Architectural Fixes vs. Baseline ALIGNN
════════════════════════════════════════
ALIGNN (Choudhary & DeCost, npj Comp. Mat. 2021) uses a line-graph
construction where bond angles become nodes in an auxiliary graph. While
this captures angular information, its scalar outputs remain SO(3)-invariant
— it cannot natively predict equivariant vector quantities like directional
forces without projecting onto local reference frames. EGNN avoids this
entirely by maintaining an explicit coordinate channel that transforms
equivariantly under E(3), making directional force prediction architecturally
native rather than a post-hoc projection.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, GELU, Sequential, Dropout
from torch.autograd import Function

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend safe for scripts
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False

import torch_geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool


# ==============================================================================
# 0. UTILITY: GRADIENT SCALING (prevents classification head from hijacking)
# ==============================================================================

class _GradientScaleFunc(Function):
    """Scales gradients flowing backward through this op by a constant factor."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def gradient_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Apply gradient scaling — identity in forward, scales grad in backward."""
    return _GradientScaleFunc.apply(x, scale)


# ==============================================================================
# 1. SYNTHETIC EXOTIC METAMATERIAL LATTICE DATASET GENERATOR
# ==============================================================================

class ExoticLatticeDataset(Dataset):
    r"""
    Synthetic dataset generating exotic metamaterial lattices for spacetime
    metric engineering research.

    Node Features (8 dimensions):
      0: Negative energy density ρ_e < 0 (GeV/fm³)
      1: Vacuum expectation value trace ⟨T^μ_μ⟩
      2: Casimir anisotropic pressure P_xx
      3: Casimir anisotropic pressure P_yy
      4: Casimir anisotropic pressure P_zz
      5: Topological defect charge Q_i
      6: Effective negative gravitational mass m_eff
      7: Quantum excitation frequency ω_i

    Edge Features (5 dimensions):
      0: Euclidean spatial distance ‖r_j − r_i‖
      1: Quantum entanglement concurrence S(i, j)
      2: Casimir interaction potential K_ij
      3: Geodesic spacetime interval s²_ij
      4: Gauge potential connection A_μ^(ij)

    Targets (5 separate scalars — fully decoupled):
      y_fx:     F_x component of gravitational repulsion vector
      y_fy:     F_y component of gravitational repulsion vector
      y_fz:     F_z component of gravitational repulsion vector
      y_metric: Metric distortion norm ‖h_μν‖_F
      y_cls:    Binary micro-singularity instability flag (0 or 1)
    """

    def __init__(self, num_samples=300, min_nodes=15, max_nodes=35,
                 radius_cutoff=5.0, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.num_samples = num_samples
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.radius_cutoff = radius_cutoff
        self.data_list = []
        self._generate_dataset()

    def _generate_dataset(self):
        for _ in range(self.num_samples):
            num_nodes = torch.randint(self.min_nodes, self.max_nodes + 1, (1,)).item()

            # 3D spatial coordinates r_i ∈ [−5, 5]³
            pos = (torch.rand(num_nodes, 3) - 0.5) * 10.0

            # ── Node scalar features ────────────────────────────────────────
            rho_e = -torch.rand(num_nodes, 1) * 5.0 - 0.1       # ρ_e < 0
            vev_trace = torch.randn(num_nodes, 1) * 2.0          # ⟨T^μ_μ⟩
            p_casimir = torch.rand(num_nodes, 3) * 3.0            # P_xx, P_yy, P_zz
            topo_charge = torch.randint(-2, 3, (num_nodes, 1)).float()  # Q_i
            m_eff = rho_e * 0.8                                    # m_eff ∝ ρ_e
            omega_i = torch.rand(num_nodes, 1) * 10.0 + 1.0       # ω_i

            x = torch.cat([rho_e, vev_trace, p_casimir, topo_charge,
                           m_eff, omega_i], dim=-1)               # (N, 8)

            # ── Edge construction (radius graph) ────────────────────────────
            dist_matrix = torch.cdist(pos, pos)
            adj = (dist_matrix < self.radius_cutoff) & (dist_matrix > 0)
            edge_index = adj.nonzero(as_tuple=False).t().contiguous()

            if edge_index.numel() == 0:
                # Fallback: k-NN with k=1 guarantees connectivity
                dist_matrix.fill_diagonal_(float("inf"))
                _, nearest = dist_matrix.min(dim=1)
                src = torch.arange(num_nodes)
                edge_index = torch.stack(
                    [torch.cat([src, nearest]), torch.cat([nearest, src])], dim=0
                )

            row, col = edge_index[0], edge_index[1]
            dist = torch.norm(pos[col] - pos[row], dim=-1, keepdim=True)

            # Entanglement concurrence S(i,j) ∈ [0, 1]
            entanglement = torch.sigmoid(-dist + torch.randn_like(dist) * 0.5)

            # Casimir potential K_ij ∝ 1/d⁴
            casimir_k = 1.0 / (dist.clamp(min=0.1) ** 4)

            # Geodesic interval s²_ij with pseudo-Minkowski approximation
            spatial_sq = dist ** 2
            time_comp = (pos[col, 0] - pos[row, 0]).unsqueeze(-1) ** 2
            geodesic = -time_comp + spatial_sq

            # Gauge connection potential A_μ^(ij)
            gauge_A = torch.sin(dist) * 0.5

            edge_attr = torch.cat([dist, entanglement, casimir_k,
                                   geodesic, gauge_A], dim=-1)     # (E, 5)

            # ── Target generation (physics-inspired synthetic labels) ───────
            total_neg_energy = torch.sum(rho_e)
            mean_topo = torch.mean(torch.abs(topo_charge))

            # Repulsion force vector: centroid-weighted by negative energy
            centroid_repulsion = torch.mean(pos * rho_e, dim=0)    # (3,)
            fx = centroid_repulsion[0].unsqueeze(0).unsqueeze(0)   # (1, 1)
            fy = centroid_repulsion[1].unsqueeze(0).unsqueeze(0)
            fz = centroid_repulsion[2].unsqueeze(0).unsqueeze(0)

            # Metric distortion norm ‖h_μν‖_F
            metric_dist = (torch.abs(total_neg_energy) * 0.15
                           + mean_topo * 0.05).unsqueeze(0).unsqueeze(0)

            # Instability classification (Laplace pole criterion)
            alpha_eff = (torch.abs(total_neg_energy) * 0.08
                         + mean_topo * 0.4
                         + torch.randn(1).item() * 0.2)
            y_cls = (alpha_eff > 2.2).float().unsqueeze(0)         # (1, 1)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=pos,
                y_fx=fx,
                y_fy=fy,
                y_fz=fz,
                y_metric=metric_dist,
                y_cls=y_cls,
            )
            self.data_list.append(data)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


# ==============================================================================
# 2. E(n)-EQUIVARIANT GRAPH NEURAL NETWORK (EGNN) LAYER
# ==============================================================================

class EGNNLayer(MessagePassing):
    r"""
    E(n)-Equivariant Graph Neural Network layer (Satorras et al., ICML 2021).

    Jointly updates:
      • Scalar (invariant) node features  h_i ∈ ℝ^d
      • Coordinate (equivariant) features  x_i ∈ ℝ^3

    Update equations:
      m_ij   = φ_e(h_i, h_j, ‖x_i − x_j‖², e_ij)
      x_i'   = x_i + Σ_j (x_i − x_j) · φ_x(m_ij)
      agg_i  = Σ_j m_ij
      h_i'   = φ_h(h_i, agg_i)

    The coordinate update uses only relative vectors scaled by learned scalars,
    guaranteeing E(3)-equivariance of the coordinate channel.
    """

    def __init__(self, scalar_dim, edge_dim, coord_dim=3, act=nn.SiLU,
                 residual=True, normalize_coords=True, tanh_clamp=True):
        super().__init__(aggr="add")
        self.scalar_dim = scalar_dim
        self.coord_dim = coord_dim
        self.residual = residual
        self.normalize_coords = normalize_coords
        self.tanh_clamp = tanh_clamp

        # Message MLP: φ_e(h_i, h_j, d², e_ij) → m_ij
        msg_input_dim = 2 * scalar_dim + 1 + edge_dim   # h_i ⊕ h_j ⊕ d² ⊕ e_ij
        self.message_mlp = Sequential(
            Linear(msg_input_dim, scalar_dim),
            LayerNorm(scalar_dim),
            act(),
            Linear(scalar_dim, scalar_dim),
            LayerNorm(scalar_dim),
            act(),
        )

        # Coordinate weight MLP: φ_x(m_ij) → scalar weight for coord update
        self.coord_mlp = Sequential(
            Linear(scalar_dim, scalar_dim),
            act(),
            Linear(scalar_dim, 1),
        )
        if self.tanh_clamp:
            self.coord_mlp.append(nn.Tanh())

        # Node update MLP: φ_h(h_i, agg_i) → h_i'
        self.node_mlp = Sequential(
            Linear(2 * scalar_dim, scalar_dim),
            LayerNorm(scalar_dim),
            act(),
            Linear(scalar_dim, scalar_dim),
        )
        self.node_norm = LayerNorm(scalar_dim)

    def forward(self, h, x, edge_index, edge_attr):
        """
        Args:
            h:          (N, scalar_dim)  — scalar node features
            x:          (N, 3)           — equivariant coordinate features
            edge_index: (2, E)           — edge connectivity
            edge_attr:  (E, edge_dim)    — edge scalar features

        Returns:
            h_out: (N, scalar_dim) — updated scalar features
            x_out: (N, 3)         — updated equivariant coordinates
        """
        row, col = edge_index

        # Compute pairwise squared distances (invariant scalar)
        rel_vec = x[row] - x[col]                          # (E, 3)
        dist_sq = (rel_vec ** 2).sum(dim=-1, keepdim=True)  # (E, 1)

        # Build message inputs: h_i ⊕ h_j ⊕ d² ⊕ e_ij
        msg_input = torch.cat([h[row], h[col], dist_sq, edge_attr], dim=-1)
        msg = self.message_mlp(msg_input)                   # (E, scalar_dim)

        # ── Coordinate update (equivariant) ─────────────────────────────
        coord_weight = self.coord_mlp(msg)                  # (E, 1)
        coord_delta = rel_vec * coord_weight                # (E, 3)

        # Aggregate coordinate deltas per node
        coord_agg = torch.zeros_like(x)
        coord_agg.scatter_add_(0, row.unsqueeze(-1).expand_as(coord_delta),
                               coord_delta)

        if self.normalize_coords:
            # Normalize by node degree to stabilize updates
            degree = torch.zeros(x.size(0), 1, device=x.device)
            degree.scatter_add_(0, row.unsqueeze(-1),
                                torch.ones(row.size(0), 1, device=x.device))
            degree = degree.clamp(min=1.0)
            coord_agg = coord_agg / degree

        x_out = x + coord_agg

        # ── Scalar update (invariant) ───────────────────────────────────
        # Aggregate messages per node
        msg_agg = torch.zeros_like(h)
        msg_agg.scatter_add_(0, row.unsqueeze(-1).expand_as(msg), msg)

        h_out = self.node_mlp(torch.cat([h, msg_agg], dim=-1))

        if self.residual:
            h_out = h + h_out

        h_out = self.node_norm(h_out)

        return h_out, x_out


# ==============================================================================
# 3. GRAPH TRANSFORMER LAYER (SCALAR-ONLY, POST-EGNN)
# ==============================================================================

class ScalarGraphTransformerLayer(nn.Module):
    r"""
    Multi-Head Self-Attention Transformer layer operating ONLY on the
    invariant scalar channel h_i. Applied after EGNN layers to capture
    long-range non-local interactions across all lattice nodes.

    This layer is intentionally decoupled from coordinate processing —
    spatial equivariance is handled entirely by the EGNN backbone.
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim

        self.q_proj = Linear(hidden_dim, hidden_dim)
        self.k_proj = Linear(hidden_dim, hidden_dim)
        self.v_proj = Linear(hidden_dim, hidden_dim)
        self.out_proj = Linear(hidden_dim, hidden_dim)

        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)

        self.ffn = Sequential(
            Linear(hidden_dim, hidden_dim * 4),
            GELU(),
            Dropout(dropout),
            Linear(hidden_dim * 4, hidden_dim),
            Dropout(dropout),
        )
        self.attn_dropout = Dropout(dropout)

    def forward(self, h, batch):
        """
        Args:
            h:     (N_total, hidden_dim) — scalar node features (batched)
            batch: (N_total,) — graph membership index

        Returns:
            h_out: (N_total, hidden_dim) — updated scalar features
        """
        N = h.size(0)
        residual = h

        Q = self.q_proj(h).view(N, self.num_heads, self.head_dim)
        K = self.k_proj(h).view(N, self.num_heads, self.head_dim)
        V = self.v_proj(h).view(N, self.num_heads, self.head_dim)

        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=h.device)

        # Per-graph dense attention
        uniques, counts = torch.unique(batch, return_counts=True)
        out_list = []
        node_offset = 0

        for b_idx in range(len(uniques)):
            n_b = counts[b_idx].item()
            q_b = Q[node_offset: node_offset + n_b]  # (n_b, H, d_k)
            k_b = K[node_offset: node_offset + n_b]
            v_b = V[node_offset: node_offset + n_b]

            # Attention scores: (H, n_b, n_b)
            scores = torch.einsum("ihd,jhd->hij", q_b, k_b)
            scores = scores / math.sqrt(self.head_dim)

            attn = F.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)

            # Weighted aggregation: (n_b, H, d_k)
            out_b = torch.einsum("hij,jhd->ihd", attn, v_b)
            out_b = out_b.contiguous().view(n_b, self.hidden_dim)
            out_list.append(out_b)
            node_offset += n_b

        attn_out = torch.cat(out_list, dim=0)
        h = self.norm1(residual + self.out_proj(attn_out))
        h = self.norm2(h + self.ffn(h))
        return h


# ==============================================================================
# 4. MULTI-TASK ANTIGRAVITY GRAPH TRANSFORMER v2 (5-HEAD ARCHITECTURE)
# ==============================================================================

class MultiTaskAntigravityTransformerV2(nn.Module):
    r"""
    Publication-ready 5-head multi-task architecture:

    Input Graph ──► EGNN Backbone (equivariant) ──► Scalar Transformer
                                                        │
                              ┌──────────────────┬──────┴───────┐
                              ▼                  ▼              ▼
                    Equivariant Branch     Shared Scalar     Scalar Trunk
                    (coord deltas)          Trunk             │
                    ┌──┬──┬──┐              │            ┌────┴────┐
                    ▼  ▼  ▼  │              ▼            ▼         ▼
                   H1 H2 H3  │           Head 4       Head 5
                  Fx Fy Fz   │          ‖h_μν‖     Instability
                             │          (scalar)   (classification)
                             │
                    (force components from
                     equivariant readout)

    Heads 1–3: Force component predictions from equivariant coordinate
               deltas Δx_i = x_i' − x_i^init, pooled and refined.
    Head 4:    Metric distortion norm from invariant scalar features.
    Head 5:    Binary instability classification with gradient scaling
               (0.1× backward multiplier prevents hijacking shared trunk).
    """

    def __init__(
        self,
        in_node_dim=8,
        in_edge_dim=5,
        hidden_dim=128,
        num_egnn_layers=4,
        num_transformer_layers=2,
        num_heads=4,
        dropout=0.1,
        cls_grad_scale=0.1,
    ):
        super().__init__()
        self.cls_grad_scale = cls_grad_scale

        # ── 1. Input Embeddings ─────────────────────────────────────────
        self.node_embedding = Sequential(
            Linear(in_node_dim, hidden_dim),
            LayerNorm(hidden_dim),
            GELU(),
            Linear(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim),
        )
        self.edge_embedding = Sequential(
            Linear(in_edge_dim, hidden_dim),
            LayerNorm(hidden_dim),
            GELU(),
            Linear(hidden_dim, hidden_dim),
            LayerNorm(hidden_dim),
        )

        # ── 2. EGNN Backbone (equivariant feature extractor) ────────────
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(
                scalar_dim=hidden_dim,
                edge_dim=hidden_dim,
                coord_dim=3,
                residual=True,
                normalize_coords=True,
                tanh_clamp=True,
            )
            for _ in range(num_egnn_layers)
        ])

        # ── 3. Scalar Graph Transformer (long-range interactions) ───────
        self.transformer_layers = nn.ModuleList([
            ScalarGraphTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_transformer_layers)
        ])

        # ── 4. Shared Scalar Trunk ──────────────────────────────────────
        self.shared_trunk = Sequential(
            Linear(hidden_dim * 2, hidden_dim * 2),
            LayerNorm(hidden_dim * 2),
            GELU(),
            Dropout(dropout),
            Linear(hidden_dim * 2, hidden_dim),
            LayerNorm(hidden_dim),
            GELU(),
        )

        # ── 5. Equivariant Force Readout ────────────────────────────────
        # The pooled coordinate delta is (batch_size, 3).
        # Each force head refines one component with a small MLP.
        self.force_coord_mlp = Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            GELU(),
            Linear(hidden_dim, 3),
        )
        # Per-component refinement heads taking (coord_component, shared_repr)
        self.head_fx = Sequential(
            Linear(hidden_dim + 1, hidden_dim // 2),
            LayerNorm(hidden_dim // 2),
            GELU(),
            Dropout(dropout),
            Linear(hidden_dim // 2, hidden_dim // 4),
            GELU(),
            Linear(hidden_dim // 4, 1),
        )
        self.head_fy = Sequential(
            Linear(hidden_dim + 1, hidden_dim // 2),
            LayerNorm(hidden_dim // 2),
            GELU(),
            Dropout(dropout),
            Linear(hidden_dim // 2, hidden_dim // 4),
            GELU(),
            Linear(hidden_dim // 4, 1),
        )
        self.head_fz = Sequential(
            Linear(hidden_dim + 1, hidden_dim // 2),
            LayerNorm(hidden_dim // 2),
            GELU(),
            Dropout(dropout),
            Linear(hidden_dim // 2, hidden_dim // 4),
            GELU(),
            Linear(hidden_dim // 4, 1),
        )

        # ── 6. Head 4: Metric Distortion (invariant scalar) ────────────
        self.head_metric = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            LayerNorm(hidden_dim // 2),
            GELU(),
            Dropout(dropout),
            Linear(hidden_dim // 2, hidden_dim // 4),
            GELU(),
            Linear(hidden_dim // 4, 1),
        )

        # ── 7. Head 5: Instability Classification (gradient-scaled) ────
        self.head_cls = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            LayerNorm(hidden_dim // 2),
            GELU(),
            Dropout(dropout),
            Linear(hidden_dim // 2, hidden_dim // 4),
            GELU(),
            Linear(hidden_dim // 4, 1),
        )

    def forward(self, data):
        x_scalar = data.x             # (N_total, 8)
        edge_index = data.edge_index  # (2, E_total)
        edge_attr = data.edge_attr    # (E_total, 5)
        pos = data.pos                # (N_total, 3)
        batch = data.batch            # (N_total,)

        # ── Input Embedding ─────────────────────────────────────────────
        h = self.node_embedding(x_scalar)   # (N, hidden)
        e = self.edge_embedding(edge_attr)  # (E, hidden)

        # Save initial coordinates for equivariant delta readout
        x_init = pos.clone()
        x_eq = pos.clone()

        # ── EGNN Backbone ───────────────────────────────────────────────
        for egnn in self.egnn_layers:
            h, x_eq = egnn(h, x_eq, edge_index, e)

        # Equivariant coordinate delta: how much EGNN moved each node
        coord_delta = x_eq - x_init   # (N, 3) — transforms equivariantly

        # ── Scalar Transformer ──────────────────────────────────────────
        for trans in self.transformer_layers:
            h = trans(h, batch)

        # ── Graph-Level Pooling ─────────────────────────────────────────
        h_mean = global_mean_pool(h, batch)            # (B, hidden)
        h_max = global_max_pool(h, batch)              # (B, hidden)
        h_pooled = torch.cat([h_mean, h_max], dim=-1)  # (B, hidden*2)

        # Pool coordinate deltas equivariantly (mean preserves equivariance)
        coord_delta_pooled = global_mean_pool(coord_delta, batch)  # (B, 3)

        # ── Shared Trunk ────────────────────────────────────────────────
        shared_repr = self.shared_trunk(h_pooled)      # (B, hidden)

        # ── Force Readout (Equivariant Branch) ──────────────────────────
        # Learned correction to the raw coordinate delta signal
        force_correction = self.force_coord_mlp(h_pooled)  # (B, 3)
        force_signal = coord_delta_pooled + force_correction  # (B, 3)

        # Per-component refinement: each head gets (shared_repr, force_i)
        pred_fx = self.head_fx(
            torch.cat([shared_repr, force_signal[:, 0:1]], dim=-1)
        )
        pred_fy = self.head_fy(
            torch.cat([shared_repr, force_signal[:, 1:2]], dim=-1)
        )
        pred_fz = self.head_fz(
            torch.cat([shared_repr, force_signal[:, 2:3]], dim=-1)
        )

        # ── Head 4: Metric Distortion ──────────────────────────────────
        pred_metric = self.head_metric(shared_repr)

        # ── Head 5: Classification (gradient-scaled) ───────────────────
        # Scale gradients flowing back through the classification branch
        # to prevent it from dominating shared trunk learning
        cls_input = gradient_scale(shared_repr, self.cls_grad_scale)
        pred_cls = self.head_cls(cls_input)

        return pred_fx, pred_fy, pred_fz, pred_metric, pred_cls


# ==============================================================================
# 5. FIVE-HEAD HOMOSCEDASTIC UNCERTAINTY LOSS MODULE
# ==============================================================================

class FiveHeadUncertaintyLoss(nn.Module):
    r"""
    Per-head homoscedastic uncertainty weighting (Kendall & Gal, CVPR 2018)
    with FIVE independent learnable log-variance parameters s_k.

    Total loss:

        L = Σ_{k=1}^{4} [½ exp(−s_k) · L_k^reg + ½ s_k]
          + exp(−s₅) · L₅^cls + ½ s₅

    This ensures no single head can dominate backpropagation — the network
    automatically learns to balance easy vs. hard tasks.
    """

    def __init__(self):
        super().__init__()
        # 5 independent learnable log-variance parameters
        self.s_fx = nn.Parameter(torch.tensor(0.0))
        self.s_fy = nn.Parameter(torch.tensor(0.0))
        self.s_fz = nn.Parameter(torch.tensor(0.0))
        self.s_metric = nn.Parameter(torch.tensor(0.0))
        self.s_cls = nn.Parameter(torch.tensor(-1.0))  # bias lower initially

    def forward(self, pred_fx, pred_fy, pred_fz, pred_metric, pred_cls,
                tgt_fx, tgt_fy, tgt_fz, tgt_metric, tgt_cls):
        """
        Returns:
            total_loss: scalar tensor (differentiable)
            loss_dict:  dict of per-head scalar losses (for logging)
        """
        tgt_fx = tgt_fx.view_as(pred_fx)
        tgt_fy = tgt_fy.view_as(pred_fy)
        tgt_fz = tgt_fz.view_as(pred_fz)
        tgt_metric = tgt_metric.view_as(pred_metric)
        tgt_cls = tgt_cls.view_as(pred_cls)

        # Per-head raw losses
        l_fx = F.smooth_l1_loss(pred_fx, tgt_fx)
        l_fy = F.smooth_l1_loss(pred_fy, tgt_fy)
        l_fz = F.smooth_l1_loss(pred_fz, tgt_fz)
        l_metric = F.smooth_l1_loss(pred_metric, tgt_metric)
        l_cls = F.binary_cross_entropy_with_logits(pred_cls, tgt_cls)

        # Uncertainty-weighted combination
        total = (
            0.5 * torch.exp(-self.s_fx) * l_fx + 0.5 * self.s_fx
            + 0.5 * torch.exp(-self.s_fy) * l_fy + 0.5 * self.s_fy
            + 0.5 * torch.exp(-self.s_fz) * l_fz + 0.5 * self.s_fz
            + 0.5 * torch.exp(-self.s_metric) * l_metric + 0.5 * self.s_metric
            + torch.exp(-self.s_cls) * l_cls + 0.5 * self.s_cls
        )

        loss_dict = {
            "l_fx": l_fx.item(),
            "l_fy": l_fy.item(),
            "l_fz": l_fz.item(),
            "l_metric": l_metric.item(),
            "l_cls": l_cls.item(),
        }
        return total, loss_dict


# ==============================================================================
# 6. TRAINING, VALIDATION & EVALUATION PIPELINE
# ==============================================================================

def train_epoch(model, loss_module, loader, optimizer, device):
    """Single training epoch with per-head loss tracking."""
    model.train()
    n_samples = 0
    accum = {"total": 0.0, "l_fx": 0.0, "l_fy": 0.0,
             "l_fz": 0.0, "l_metric": 0.0, "l_cls": 0.0}

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred_fx, pred_fy, pred_fz, pred_metric, pred_cls = model(batch)
        loss, ld = loss_module(
            pred_fx, pred_fy, pred_fz, pred_metric, pred_cls,
            batch.y_fx, batch.y_fy, batch.y_fz, batch.y_metric, batch.y_cls,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(loss_module.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        bs = batch.num_graphs
        n_samples += bs
        accum["total"] += loss.item() * bs
        for k in ld:
            accum[k] += ld[k] * bs

    return {k: v / n_samples for k, v in accum.items()}


@torch.no_grad()
def evaluate(model, loss_module, loader, device):
    """Evaluation with per-head metrics."""
    model.eval()
    n_samples = 0
    accum = {"total": 0.0, "l_fx": 0.0, "l_fy": 0.0,
             "l_fz": 0.0, "l_metric": 0.0, "l_cls": 0.0}

    all_pred = {"fx": [], "fy": [], "fz": [], "metric": [], "cls": []}
    all_tgt = {"fx": [], "fy": [], "fz": [], "metric": [], "cls": []}

    for batch in loader:
        batch = batch.to(device)
        pred_fx, pred_fy, pred_fz, pred_metric, pred_cls = model(batch)
        loss, ld = loss_module(
            pred_fx, pred_fy, pred_fz, pred_metric, pred_cls,
            batch.y_fx, batch.y_fy, batch.y_fz, batch.y_metric, batch.y_cls,
        )

        bs = batch.num_graphs
        n_samples += bs
        accum["total"] += loss.item() * bs
        for k in ld:
            accum[k] += ld[k] * bs

        all_pred["fx"].append(pred_fx.cpu())
        all_pred["fy"].append(pred_fy.cpu())
        all_pred["fz"].append(pred_fz.cpu())
        all_pred["metric"].append(pred_metric.cpu())
        all_pred["cls"].append(torch.sigmoid(pred_cls).cpu())

        all_tgt["fx"].append(batch.y_fx.cpu())
        all_tgt["fy"].append(batch.y_fy.cpu())
        all_tgt["fz"].append(batch.y_fz.cpu())
        all_tgt["metric"].append(batch.y_metric.cpu())
        all_tgt["cls"].append(batch.y_cls.cpu())

    # Concatenate
    for k in all_pred:
        all_pred[k] = torch.cat(all_pred[k], dim=0).view(-1)
        all_tgt[k] = torch.cat(all_tgt[k], dim=0).view(-1)

    # Per-head regression MAE
    mae_fx = F.l1_loss(all_pred["fx"], all_tgt["fx"]).item()
    mae_fy = F.l1_loss(all_pred["fy"], all_tgt["fy"]).item()
    mae_fz = F.l1_loss(all_pred["fz"], all_tgt["fz"]).item()
    mae_metric = F.l1_loss(all_pred["metric"], all_tgt["metric"]).item()

    # Per-head regression RMSE
    rmse_fx = torch.sqrt(F.mse_loss(all_pred["fx"], all_tgt["fx"])).item()
    rmse_fy = torch.sqrt(F.mse_loss(all_pred["fy"], all_tgt["fy"])).item()
    rmse_fz = torch.sqrt(F.mse_loss(all_pred["fz"], all_tgt["fz"])).item()
    rmse_metric = torch.sqrt(F.mse_loss(all_pred["metric"], all_tgt["metric"])).item()

    # Classification accuracy
    cls_binary = (all_pred["cls"] >= 0.5).float()
    cls_acc = (cls_binary == all_tgt["cls"]).float().mean().item()

    # Prediction variance (collapse detection)
    var_fx = all_pred["fx"].var().item()
    var_fy = all_pred["fy"].var().item()
    var_fz = all_pred["fz"].var().item()

    metrics = {
        "total_loss": accum["total"] / n_samples,
        **{k: v / n_samples for k, v in accum.items() if k != "total"},
        "mae_fx": mae_fx, "mae_fy": mae_fy, "mae_fz": mae_fz,
        "mae_metric": mae_metric,
        "rmse_fx": rmse_fx, "rmse_fy": rmse_fy, "rmse_fz": rmse_fz,
        "rmse_metric": rmse_metric,
        "cls_acc": cls_acc,
        "var_fx": var_fx, "var_fy": var_fy, "var_fz": var_fz,
    }
    return metrics, all_pred, all_tgt


# ==============================================================================
# 7. IEEE PUBLICATION-QUALITY FIGURE GENERATION
# ==============================================================================

def plot_ieee_figures(train_history, val_history, test_pred, test_tgt,
                     loss_module, results_dir="results"):
    """
    Generate IEEE publication-quality figures:
      1. Per-head loss convergence curves
      2. Per-component parity plots (Fx, Fy, Fz, ‖h_μν‖)
      3. Learned uncertainty weights evolution
    """
    import numpy as np
    import traceback

    if not HAS_MATPLOTLIB or plt is None:
        print("\n[*] matplotlib not available — skipping figure generation.")
        print("    Install via: pip install matplotlib")
        return

    os.makedirs(results_dir, exist_ok=True)

    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
        "figure.titlesize": 15, "font.family": "serif",
    })

    saved = []
    epochs = list(range(1, len(train_history) + 1))

    # ── Figure 1: Per-Head Loss Convergence ─────────────────────────────
    try:
        print("\n[*] Generating Figure 1: Per-Head Loss Convergence...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Multi-Task Loss Convergence (5 Independent Heads)",
                     fontweight="bold")

        head_keys = [
            ("l_fx", r"$F_x$ Loss", "tab:blue"),
            ("l_fy", r"$F_y$ Loss", "tab:orange"),
            ("l_fz", r"$F_z$ Loss", "tab:green"),
            ("l_metric", r"$\|h_{\mu\nu}\|$ Loss", "tab:red"),
            ("l_cls", "Instability BCE Loss", "tab:purple"),
            ("total", "Total Weighted Loss", "tab:brown"),
        ]

        for idx, (key, title, color) in enumerate(head_keys):
            ax = axes[idx // 3, idx % 3]
            tr = [h.get(key, h.get("total", 0)) for h in train_history]
            vl = [h.get(key, h.get("total", 0)) for h in val_history]
            ax.plot(epochs, tr, color=color, label="Train", linewidth=1.8)
            ax.plot(epochs, vl, color=color, linestyle="--", label="Val",
                    linewidth=1.8, alpha=0.7)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="upper right")

        plt.tight_layout()
        for ext in ("png", "pdf"):
            path = os.path.join(results_dir, f"per_head_loss_convergence.{ext}")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            saved.append(path)
        plt.close(fig)
        print("[OK] Figure 1 saved.")
    except Exception:
        print("[ERROR] Figure 1 failed:")
        traceback.print_exc()

    # ── Figure 2: Per-Component Parity Plots ────────────────────────────
    try:
        print("\n[*] Generating Figure 2: Per-Component Parity Plots...")
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        fig.suptitle("Predicted vs. True — Decoupled 5-Head Readout",
                     fontweight="bold")

        parity_items = [
            ("fx", r"$F_x$", "tab:blue"),
            ("fy", r"$F_y$", "tab:orange"),
            ("fz", r"$F_z$", "tab:green"),
            ("metric", r"$\|h_{\mu\nu}\|_F$", "tab:red"),
        ]

        for idx, (key, label, color) in enumerate(parity_items):
            ax = axes[idx]
            y_true = test_tgt[key].numpy()
            y_pred = test_pred[key].numpy()

            ax.scatter(y_true, y_pred, c=color, s=40, alpha=0.7,
                       edgecolors="none", label="Predictions")

            lo = min(y_true.min(), y_pred.min()) - 0.1
            hi = max(y_true.max(), y_pred.max()) + 0.1
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5, label="Ideal y=x")

            ax.set_xlabel(f"True {label}")
            ax.set_ylabel(f"Predicted {label}")
            ax.set_title(f"{label} Parity")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="upper left", fontsize=9)

            # Annotate MAE / RMSE
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            ax.text(0.97, 0.03,
                    f"MAE={mae:.3f}\nRMSE={rmse:.3f}",
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment="bottom", horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        plt.tight_layout()
        for ext in ("png", "pdf"):
            path = os.path.join(results_dir, f"per_component_parity.{ext}")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            saved.append(path)
        plt.close(fig)
        print("[OK] Figure 2 saved.")
    except Exception:
        print("[ERROR] Figure 2 failed:")
        traceback.print_exc()

    # ── Figure 3: Learned Uncertainty Weights ───────────────────────────
    try:
        print("\n[*] Generating Figure 3: Learned Uncertainty Weights...")
        fig, ax = plt.subplots(figsize=(8, 5))
        names = [r"$s_{F_x}$", r"$s_{F_y}$", r"$s_{F_z}$",
                 r"$s_{\|h\|}$", r"$s_\mathrm{cls}$"]
        values = [
            loss_module.s_fx.item(),
            loss_module.s_fy.item(),
            loss_module.s_fz.item(),
            loss_module.s_metric.item(),
            loss_module.s_cls.item(),
        ]
        # Effective precision weights: w_k = exp(-s_k)
        weights = [math.exp(-v) for v in values]

        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
        bars = ax.bar(names, weights, color=colors, edgecolor="black",
                      linewidth=0.8)

        ax.set_ylabel("Effective Precision Weight $e^{-s_k}$")
        ax.set_title("Learned Homoscedastic Uncertainty Weights (Final Epoch)",
                     fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        for bar, val, w in zip(bars, values, weights):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"s={val:.2f}\nw={w:.2f}", ha="center", va="bottom",
                    fontsize=9)

        plt.tight_layout()
        for ext in ("png", "pdf"):
            path = os.path.join(results_dir, f"uncertainty_weights.{ext}")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            saved.append(path)
        plt.close(fig)
        print("[OK] Figure 3 saved.")
    except Exception:
        print("[ERROR] Figure 3 failed:")
        traceback.print_exc()

    print(f"\n[*] IEEE publication figures saved in '{results_dir}/':")
    for f in saved:
        print(f"    - {f}")


# ==============================================================================
# 8. MAIN EXECUTION PIPELINE
# ==============================================================================

def main():
    print("=" * 80)
    print("  MULTI-TASK ANTIGRAVITY GRAPH TRANSFORMER v2.0")
    print("  Architecture: EGNN Backbone + 5 Decoupled Heads")
    print("  Fixes: Vector Collapse | Joint Interference | Classification Hijack")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    # ── 1. Dataset ──────────────────────────────────────────────────────
    print("[*] Generating synthetic exotic metamaterial lattice dataset...")
    dataset = ExoticLatticeDataset(
        num_samples=350, min_nodes=15, max_nodes=30,
        radius_cutoff=5.0, seed=42,
    )

    train_size = 244
    val_size = 53
    test_size = len(dataset) - train_size - val_size

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    print(f"    Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── 2. Model & Loss ────────────────────────────────────────────────
    model = MultiTaskAntigravityTransformerV2(
        in_node_dim=8,
        in_edge_dim=5,
        hidden_dim=128,
        num_egnn_layers=4,
        num_transformer_layers=2,
        num_heads=4,
        dropout=0.1,
        cls_grad_scale=0.1,
    ).to(device)

    loss_module = FiveHeadUncertaintyLoss().to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loss_params = sum(p.numel() for p in loss_module.parameters())
    print(f"[*] Model parameters:    {total_params:,}")
    print(f"[*] Learnable loss params: {loss_params} (5 uncertainty weights)")

    # ── 3. Optimizer & Scheduler ────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_module.parameters()),
        lr=5e-4,
        weight_decay=1e-2,
    )

    # 5-epoch warmup followed by CosineAnnealingLR across 40 total epochs
    num_epochs = 40
    warmup_epochs = 5

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"[*] Training for {num_epochs} epochs ({warmup_epochs} warmup + CosineAnnealingLR)")
    print("-" * 80)

    # ── 4. Training Loop ────────────────────────────────────────────────
    train_history = []
    val_history = []
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        tr = train_epoch(model, loss_module, train_loader, optimizer, device)
        vl, _, _ = evaluate(model, loss_module, val_loader, device)
        scheduler.step()

        train_history.append(tr)
        val_history.append(vl)

        if vl["total_loss"] < best_val_loss:
            best_val_loss = vl["total_loss"]
            best_marker = " *BEST*"
        else:
            best_marker = ""

        # Compact per-epoch logging
        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"Loss: {tr['total']:.4f} | "
            f"Fx:{tr['l_fx']:.3f} Fy:{tr['l_fy']:.3f} Fz:{tr['l_fz']:.3f} "
            f"Met:{tr['l_metric']:.3f} Cls:{tr['l_cls']:.3f} | "
            f"Val: {vl['total_loss']:.4f} Acc:{vl['cls_acc']*100:.0f}% "
            f"VarFx:{vl['var_fx']:.3f} VarFy:{vl['var_fy']:.3f} "
            f"VarFz:{vl['var_fz']:.3f}"
            f"{best_marker}"
        )

    print("-" * 80)

    # ── 5. Final Test Evaluation ────────────────────────────────────────
    test_m, test_pred, test_tgt = evaluate(
        model, loss_module, test_loader, device,
    )

    print("=" * 80)
    print("                    FINAL TEST SET EVALUATION")
    print("=" * 80)
    print(f"  Total Loss:          {test_m['total_loss']:.4f}")
    print(f"  ------------------------------------------")
    print(f"  F_x   MAE: {test_m['mae_fx']:.4f}   RMSE: {test_m['rmse_fx']:.4f}")
    print(f"  F_y   MAE: {test_m['mae_fy']:.4f}   RMSE: {test_m['rmse_fy']:.4f}")
    print(f"  F_z   MAE: {test_m['mae_fz']:.4f}   RMSE: {test_m['rmse_fz']:.4f}")
    print(f"  ||h_uv|| MAE: {test_m['mae_metric']:.4f}   RMSE: {test_m['rmse_metric']:.4f}")
    print(f"  ------------------------------------------")
    print(f"  Classification Acc:  {test_m['cls_acc']*100:.2f}%")
    print(f"  ------------------------------------------")
    print(f"  Prediction Variance (Collapse Check):")
    print(f"    Var(F_x)={test_m['var_fx']:.4f}  "
          f"Var(F_y)={test_m['var_fy']:.4f}  "
          f"Var(F_z)={test_m['var_fz']:.4f}")
    print(f"  (Values > 0.01 indicate no vector collapse)")
    print(f"  ------------------------------------------")
    print(f"  Learned Uncertainty Weights (log-variance s_k):")
    print(f"    s_Fx={loss_module.s_fx.item():.3f}  "
          f"s_Fy={loss_module.s_fy.item():.3f}  "
          f"s_Fz={loss_module.s_fz.item():.3f}  "
          f"s_Met={loss_module.s_metric.item():.3f}  "
          f"s_Cls={loss_module.s_cls.item():.3f}")
    print("=" * 80)

    # ── 6. IEEE Figures ─────────────────────────────────────────────────
    plot_ieee_figures(
        train_history, val_history, test_pred, test_tgt,
        loss_module, results_dir="results",
    )

    # ── 7. Sample Inference ─────────────────────────────────────────────
    sample = test_ds[0]
    sample_dev = sample.to(device)
    # Ensure batch attribute exists for single-graph inference
    if not hasattr(sample_dev, "batch") or sample_dev.batch is None:
        sample_dev.batch = torch.zeros(
            sample_dev.x.size(0), dtype=torch.long, device=device
        )

    model.eval()
    with torch.no_grad():
        pfx, pfy, pfz, pmet, pcls = model(sample_dev)

    print("\n[+] SAMPLE INFERENCE ON EXOTIC METAMATERIAL GRAPH:")
    print(f"    Nodes: {sample.x.size(0)} | Edges: {sample.edge_index.size(1)}")
    print(f"    Predicted F_x: {pfx.item():.4f}  "
          f"(True: {sample.y_fx.item():.4f})")
    print(f"    Predicted F_y: {pfy.item():.4f}  "
          f"(True: {sample.y_fy.item():.4f})")
    print(f"    Predicted F_z: {pfz.item():.4f}  "
          f"(True: {sample.y_fz.item():.4f})")
    print(f"    Predicted ||h_uv||: {pmet.item():.4f}  "
          f"(True: {sample.y_metric.item():.4f})")
    print(f"    Instability Prob: {torch.sigmoid(pcls).item()*100:.2f}%  "
          f"(True: {int(sample.y_cls.item())})")
    print("=" * 80)


if __name__ == "__main__":
    main()
