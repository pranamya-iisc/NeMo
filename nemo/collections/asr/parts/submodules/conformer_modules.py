# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Optional

import torch
from torch import nn as nn
from torch.nn import LayerNorm

from nemo.collections.asr.parts.submodules.adapters.attention_adapter_mixin import AttentionAdapterModuleMixin
from nemo.collections.asr.parts.submodules.batchnorm import FusedBatchNorm1d
from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    MultiHeadAttention,
    RelPositionMultiHeadAttention,
    RelPositionMultiHeadAttentionLongformer,
)
from nemo.collections.asr.parts.utils.activations import Swish
from nemo.collections.common.parts.utils import activation_registry
from nemo.core.classes.mixins import AccessMixin

__all__ = ['ConformerConvolution', 'ConformerFeedForward', 'ConformerMoEFeedForward', 'ConformerLayer']


class ConformerLayer(torch.nn.Module, AttentionAdapterModuleMixin, AccessMixin):
    """A single block of the Conformer encoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        self_attention_model (str): type of the attention layer and positional encoding
            'rel_pos': relative positional embedding and Transformer-XL
            'rel_pos_local_attn': relative positional embedding and Transformer-XL with local attention using
                overlapping chunks. Attention context is determined by att_context_size parameter.
            'abs_pos': absolute positional embedding and Transformer
            Default is rel_pos.
        global_tokens (int): number of tokens to be used for global attention.
            Only relevant if self_attention_model is 'rel_pos_local_attn'.
            Defaults to 0.
        global_tokens_spacing (int): how far apart the global tokens are
            Defaults to 1.
        global_attn_separate (bool): whether the q, k, v layers used for global tokens should be separate.
            Defaults to False.
        n_heads (int): number of heads for multi-head attention
        conv_kernel_size (int): kernel size for depthwise convolution in convolution module
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        use_bias (bool): Apply bias to all Linear and Conv1d layers from each ConformerLayer to improve activation flow and stabilize training of huge models.
            Defaults to True.
    """

    def __init__(
        self,
        d_model,
        d_ff,
        self_attention_model='rel_pos',
        global_tokens=0,
        global_tokens_spacing=1,
        global_attn_separate=False,
        n_heads=4,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        conv_context_size=None,
        dropout=0.1,
        dropout_att=0.1,
        pos_bias_u=None,
        pos_bias_v=None,
        att_context_size=[-1, -1],
        use_bias=True,
        use_pytorch_sdpa=False,
        use_pytorch_sdpa_backends=None,
        moe_config=None,
    ):
        super(ConformerLayer, self).__init__()

        self.use_pytorch_sdpa = use_pytorch_sdpa
        if use_pytorch_sdpa_backends is None:
            use_pytorch_sdpa_backends = []
        self.use_pytorch_sdpa_backends = use_pytorch_sdpa_backends
        self.self_attention_model = self_attention_model
        self.n_heads = n_heads
        self.fc_factor = 0.5

        moe_enabled = moe_config is not None and moe_config.get('enabled', False)
        apply_to_ff = moe_config.get('apply_to_ff', 'both') if moe_enabled else 'none'

        def _make_ff(use_moe: bool):
            if use_moe and moe_enabled:
                return ConformerMoEFeedForward(
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    use_bias=use_bias,
                    num_experts=moe_config.get('num_experts', 4),
                    top_k=moe_config.get('top_k', 2),
                    variant=moe_config.get('variant', 'top_k'),
                    num_langs=moe_config.get('num_langs', 0),
                    lang_emb_dim=moe_config.get('lang_emb_dim', 64),
                    aux_loss_coef=moe_config.get('aux_loss_coef', 1e-2),
                )
            return ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)

        # first feed forward module
        self.norm_feed_forward1 = LayerNorm(d_model)
        self.feed_forward1 = _make_ff(apply_to_ff in ('both', 'ff1'))

        # convolution module
        self.norm_conv = LayerNorm(d_model)
        self.conv = ConformerConvolution(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            norm_type=conv_norm_type,
            conv_context_size=conv_context_size,
            use_bias=use_bias,
        )

        # multi-headed self-attention module
        self.norm_self_att = LayerNorm(d_model)
        MHA_max_cache_len = att_context_size[0]

        if self_attention_model == 'rel_pos':
            self.self_attn = RelPositionMultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                max_cache_len=MHA_max_cache_len,
                use_bias=use_bias,
                use_pytorch_sdpa=self.use_pytorch_sdpa,
                use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
            )
        elif self_attention_model == 'rel_pos_local_attn':
            self.self_attn = RelPositionMultiHeadAttentionLongformer(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                max_cache_len=MHA_max_cache_len,
                att_context_size=att_context_size,
                global_tokens=global_tokens,
                global_tokens_spacing=global_tokens_spacing,
                global_attn_separate=global_attn_separate,
                use_bias=use_bias,
            )
        elif self_attention_model == 'abs_pos':
            self.self_attn = MultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                max_cache_len=MHA_max_cache_len,
                use_bias=use_bias,
                use_pytorch_sdpa=self.use_pytorch_sdpa,
                use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
            )
        else:
            raise ValueError(
                f"'{self_attention_model}' is not not a valid value for 'self_attention_model', "
                f"valid values can be from ['rel_pos', 'rel_pos_local_attn', 'abs_pos']"
            )

        # second feed forward module
        self.norm_feed_forward2 = LayerNorm(d_model)
        self.feed_forward2 = _make_ff(apply_to_ff in ('both', 'ff2'))

        self.dropout = nn.Dropout(dropout)
        self.norm_out = LayerNorm(d_model)
        self.moe_aux_loss: Optional[torch.Tensor] = None

    def forward(
        self,
        x,
        att_mask=None,
        pos_emb=None,
        pad_mask=None,
        cache_last_channel=None,
        cache_last_time=None,
        lang_id: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x (torch.Tensor): input signals (B, T, d_model)
            att_mask (torch.Tensor): attention masks(B, T, T)
            pos_emb (torch.Tensor): (L, 1, d_model)
            pad_mask (torch.tensor): padding mask
            cache_last_channel (torch.tensor) : cache for MHA layers (B, T_cache, d_model)
            cache_last_time (torch.tensor) : cache for convolutional layers (B, d_model, T_cache)
            lang_id (torch.Tensor): (B,) integer language indices for MoE routing, or None
        Returns:
            x (torch.Tensor): (B, T, d_model)
            cache_last_channel (torch.tensor) : next cache for MHA layers (B, T_cache, d_model)
            cache_last_time (torch.tensor) : next cache for convolutional layers (B, d_model, T_cache)
        """
        moe_loss = None
        residual = x
        x = self.norm_feed_forward1(x)
        if isinstance(self.feed_forward1, ConformerMoEFeedForward):
            x = self.feed_forward1(x, lang_id=lang_id, pad_mask=pad_mask)
            moe_loss = self.feed_forward1.aux_loss
        else:
            x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_self_att(residual)
        if self.self_attention_model == 'rel_pos':
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask, pos_emb=pos_emb, cache=cache_last_channel)
        elif self.self_attention_model == 'rel_pos_local_attn':
            x = self.self_attn(query=x, key=x, value=x, pad_mask=pad_mask, pos_emb=pos_emb, cache=cache_last_channel)
        elif self.self_attention_model == 'abs_pos':
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask, cache=cache_last_channel)
        else:
            x = None

        if x is not None and cache_last_channel is not None:
            (x, cache_last_channel) = x

        residual = residual + self.dropout(x)

        if self.is_adapter_available():
            # Call the MHA adapters
            pack_input = {
                'x': residual,
                'loc': 'mha',
                'att_mask': att_mask,
                'pos_emb': pos_emb,
            }
            pack_input = self.forward_enabled_adapters(pack_input)
            residual = pack_input['x']

        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask=pad_mask, cache=cache_last_time)
        if cache_last_time is not None:
            (x, cache_last_time) = x
        residual = residual + self.dropout(x)

        x = self.norm_feed_forward2(residual)
        if isinstance(self.feed_forward2, ConformerMoEFeedForward):
            x = self.feed_forward2(x, lang_id=lang_id, pad_mask=pad_mask)
            ff2_loss = self.feed_forward2.aux_loss
            moe_loss = ff2_loss if moe_loss is None else moe_loss + ff2_loss
        else:
            x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor

        self.moe_aux_loss = moe_loss
        x = self.norm_out(residual)

        if self.is_adapter_available():
            # Call the adapters
            pack_input = {
                'x': x,
                'loc': 'post',
            }
            pack_input = self.forward_enabled_adapters(pack_input)
            x = pack_input['x']

        if self.is_access_enabled(getattr(self, "model_guid", None)) and self.access_cfg.get(
            'save_encoder_tensors', False
        ):
            self.register_accessible_tensor(name='encoder', tensor=x)
        if cache_last_channel is None:
            return x
        else:
            return x, cache_last_channel, cache_last_time


class ConformerConvolution(nn.Module):
    """The convolution module for the Conformer model.
    Args:
        d_model (int): hidden dimension
        kernel_size (int): kernel size for depthwise convolution
        pointwise_activation (str): name of the activation function to be used for the pointwise conv.
            Note that Conformer uses a special key `glu_` which is treated as the original default from
            the paper.
        use_bias (bool): Use bias in all Linear and Conv1d layers improve activation flow and stabilize training of huge models.
            Defaults to True
    """

    def __init__(
        self,
        d_model,
        kernel_size,
        norm_type='batch_norm',
        conv_context_size=None,
        pointwise_activation='glu_',
        use_bias=True,
    ):
        super(ConformerConvolution, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.use_bias = use_bias

        if conv_context_size is None:
            conv_context_size = (kernel_size - 1) // 2

        if pointwise_activation in activation_registry:
            self.pointwise_activation = activation_registry[pointwise_activation]()
            dw_conv_input_dim = d_model * 2

            if hasattr(self.pointwise_activation, 'inplace'):
                self.pointwise_activation.inplace = True
        else:
            self.pointwise_activation = pointwise_activation
            dw_conv_input_dim = d_model

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.use_bias,
        )

        self.depthwise_conv = CausalConv1D(
            in_channels=dw_conv_input_dim,
            out_channels=dw_conv_input_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=conv_context_size,
            groups=dw_conv_input_dim,
            bias=self.use_bias,
        )

        if norm_type == 'batch_norm':
            self.batch_norm = nn.BatchNorm1d(dw_conv_input_dim)
        elif norm_type == 'instance_norm':
            self.batch_norm = nn.InstanceNorm1d(dw_conv_input_dim)
        elif norm_type == 'layer_norm':
            self.batch_norm = nn.LayerNorm(dw_conv_input_dim)
        elif norm_type == 'fused_batch_norm':
            self.batch_norm = FusedBatchNorm1d(dw_conv_input_dim)
        elif norm_type.startswith('group_norm'):
            num_groups = int(norm_type.replace("group_norm", ""))
            self.batch_norm = nn.GroupNorm(num_groups=num_groups, num_channels=d_model)
        else:
            raise ValueError(f"conv_norm_type={norm_type} is not valid!")

        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=dw_conv_input_dim,
            out_channels=d_model,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.use_bias,
        )

    def forward(self, x, pad_mask=None, cache=None):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)

        # Compute the activation function or use GLU for original Conformer
        if self.pointwise_activation == 'glu_':
            x = nn.functional.glu(x, dim=1)
        else:
            x = self.pointwise_activation(x)

        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x, cache=cache)
        if cache is not None:
            x, cache = x

        if self.norm_type == "layer_norm":
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.batch_norm(x)

        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        if cache is None:
            return x
        else:
            return x, cache

    def reset_parameters_conv(self):
        pw1_max = pw2_max = self.d_model**-0.5
        dw_max = self.kernel_size**-0.5

        with torch.no_grad():
            nn.init.uniform_(self.pointwise_conv1.weight, -pw1_max, pw1_max)
            nn.init.uniform_(self.pointwise_conv2.weight, -pw2_max, pw2_max)
            nn.init.uniform_(self.depthwise_conv.weight, -dw_max, dw_max)
            if self.use_bias:
                nn.init.uniform_(self.pointwise_conv1.bias, -pw1_max, pw1_max)
                nn.init.uniform_(self.pointwise_conv2.bias, -pw2_max, pw2_max)
                nn.init.uniform_(self.depthwise_conv.bias, -dw_max, dw_max)


class ConformerFeedForward(nn.Module):
    """
    feed-forward module of Conformer model.
    use_bias (bool): Apply bias to all Linear and Conv1d layers improve activation flow and stabilize training of huge models.
    """

    def __init__(self, d_model, d_ff, dropout, activation=Swish(), use_bias=True):
        super(ConformerFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_bias = use_bias
        self.linear1 = nn.Linear(d_model, d_ff, bias=self.use_bias)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model, bias=self.use_bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    def reset_parameters_ff(self):
        ffn1_max = self.d_model**-0.5
        ffn2_max = self.d_ff**-0.5
        with torch.no_grad():
            nn.init.uniform_(self.linear1.weight, -ffn1_max, ffn1_max)
            nn.init.uniform_(self.linear2.weight, -ffn2_max, ffn2_max)
            if self.use_bias:
                nn.init.uniform_(self.linear1.bias, -ffn1_max, ffn1_max)
                nn.init.uniform_(self.linear2.bias, -ffn2_max, ffn2_max)


class ConformerMoEFeedForward(nn.Module):
    """Mixture-of-Experts drop-in replacement for ConformerFeedForward.

    Routing variants:
      - 'top_k': each token is dispatched to the top-K scoring experts.
      - 'switch': each token is dispatched to the single top-1 expert (Switch Transformer).

    When num_langs > 0, a per-language embedding is concatenated to the router input,
    enabling language-conditioned routing for multilingual conformer models.
    Reference: https://arxiv.org/abs/2305.15663

    After each forward, self.aux_loss contains a load-balancing loss scalar that the
    caller should add to the total training loss (weighted by moe_aux_loss_weight).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        use_bias: bool = True,
        num_experts: int = 4,
        top_k: int = 2,
        variant: str = 'top_k',
        num_langs: int = 0,
        lang_emb_dim: int = 64,
        aux_loss_coef: float = 1e-2,
    ):
        super().__init__()
        if variant not in ('top_k', 'switch'):
            raise ValueError(f"MoE variant must be 'top_k' or 'switch', got '{variant}'")
        self.num_experts = num_experts
        self.effective_k = 1 if variant == 'switch' else top_k
        self.variant = variant
        self.aux_loss_coef = aux_loss_coef

        router_in_dim = d_model + (lang_emb_dim if num_langs > 0 else 0)
        self.router = nn.Linear(router_in_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias) for _ in range(num_experts)]
        )

        self.lang_emb: Optional[nn.Embedding] = nn.Embedding(num_langs, lang_emb_dim) if num_langs > 0 else None
        self.aux_loss: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        lang_id: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            lang_id: (B,) integer language indices for routing conditioning, or None
            pad_mask: (B, T) bool mask from ConformerEncoder — True = padding frame, False = valid.
                      Used to exclude padding tokens from load-balancing loss statistics only.
                      Expert dispatch still processes all positions (consistent with non-MoE FF).
        Returns:
            output: (B, T, d_model)
        Side-effect: self.aux_loss is set to the scalar load-balancing auxiliary loss.
        """
        B, T, D = x.shape
        N = B * T

        # valid_flat: (N,) True = valid token; None when no mask is available
        valid_flat = (~pad_mask.view(N)) if pad_mask is not None else None

        if self.lang_emb is not None and lang_id is not None:
            lang_vec = self.lang_emb(lang_id).unsqueeze(1).expand(-1, T, -1)  # (B, T, emb_dim)
            router_in = torch.cat([x, lang_vec], dim=-1).view(N, -1)
        else:
            router_in = x.view(N, D)

        router_logits = self.router(router_in)  # (N, E)
        router_probs = torch.softmax(router_logits, dim=-1)

        top_gates, top_indices = torch.topk(router_probs, self.effective_k, dim=-1)  # (N, k)
        top_gates = top_gates / top_gates.sum(dim=-1, keepdim=True)  # renormalize within top-k

        # Load-balancing auxiliary loss: num_experts * sum(f_i * p_i)
        # f_i: fraction of valid tokens dispatched to expert i (top-1 hard assignment, no-grad)
        # p_i: mean router probability for expert i over valid tokens (differentiable)
        # Padding frames are excluded so batch imbalance doesn't bias the loss signal.
        with torch.no_grad():
            top1_hot = torch.zeros(N, self.num_experts, device=x.device, dtype=router_probs.dtype)
            top1_hot.scatter_(1, top_indices[:, 0:1], 1.0)
            f = top1_hot[valid_flat].mean(0) if valid_flat is not None else top1_hot.mean(0)
        p = router_probs[valid_flat].mean(0) if valid_flat is not None else router_probs.mean(0)
        self.aux_loss = self.aux_loss_coef * self.num_experts * (f * p).sum()

        x_flat = x.view(N, D)
        output = torch.zeros(N, D, dtype=x.dtype, device=x.device)

        for e_idx in range(self.num_experts):
            # Aggregate gate weights across top-k slots for expert e_idx
            mask_e = (top_indices == e_idx)  # (N, k)
            gate_e = (top_gates * mask_e.to(top_gates.dtype)).sum(-1)  # (N,)
            token_mask = gate_e > 0
            if not token_mask.any():
                continue
            e_out = self.experts[e_idx](x_flat[token_mask])  # (n_e, D)
            tok_idx = token_mask.nonzero(as_tuple=False).view(-1)  # (n_e,)
            output.scatter_add_(
                0,
                tok_idx.unsqueeze(-1).expand(tok_idx.size(0), D),
                gate_e[token_mask].unsqueeze(-1) * e_out,
            )

        return output.view(B, T, D)
