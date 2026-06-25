# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""
BESTOW cross-attention adapter with optional multilingual extensions.

Reference: "BESTOW: Efficient and Streamable Speech Language Model with the
Best of Two Worlds in GPT and T5" (arXiv:2406.19954).

The adapter fuses speech encoder outputs into LLM input embeddings via stacked
transformer blocks, each containing:
  1. Causal self-attention on text embeddings.
  2. Cross-attention where text is query and speech is key/value.
  3. A feed-forward network (dense or sparse MoE for multilingual).

A global residual adds the original text embeddings to the adapter output to
preserve the LLM's pre-trained text representations.

Multilingual extensions
-----------------------
Two orthogonal improvements for multilingual scenarios:

Language-ID conditioning
    An optional learnable embedding table maps a per-batch language ID to a
    vector that is added to speech features before cross-attention. This gives
    the adapter an explicit cue about which language is being spoken, allowing
    it to specialise its alignment pattern per language family without changing
    the architecture capacity.

Sparse Mixture-of-Experts FFN (MoE)
    The dense FFN in each adapter block can be replaced with ``BESTOWMoEFFN``,
    which maintains N expert FFNs and routes each token to the top-K experts via
    a learned linear router. During training, a Switch-Transformer-style
    load-balancing auxiliary loss encourages uniform expert utilisation,
    preventing expert collapse. The router learns language-specific patterns
    from the data without requiring explicit language labels.

    Typical multilingual settings: num_experts=8, top_k=2.
    MoE auxiliary loss coefficient (moe_aux_loss_coeff) is typically 0.01–0.1.

For streaming inference (BESTOW-S), a Wait-K policy restricts each text
position t to attend only to speech frames 0..(wait_k + t) * stride - 1.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Sparse MoE FFN
# ---------------------------------------------------------------------------


class BESTOWMoEFFN(nn.Module):
    """
    Sparse Mixture-of-Experts feed-forward network for multilingual BESTOW.

    Each token is routed to its top-K experts (out of ``num_experts``).  The
    final output is the weighted sum of those expert outputs, where weights are
    the normalised router scores for the selected experts.

    Load-balancing auxiliary loss
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Following Switch Transformer (Fedus et al., 2022), we compute::

        aux_loss = num_experts * sum_i( f_i * P_i )

    where ``f_i`` is the fraction of tokens routed to expert i (hard assignment
    of the top-1 expert), and ``P_i`` is the mean router probability for expert
    i (soft, differentiable).  Multiplying hard and soft terms makes both
    quantities approach 1/num_experts at the optimum.

    Args:
        hidden_dim: Input and output feature dimension.
        ffn_dim: Inner dimension of each expert FFN (typically 4 × hidden_dim).
        num_experts: Total number of expert FFNs (default 8).
        top_k: Number of experts each token is dispatched to (default 2).
        dropout: Dropout probability inside expert FFNs.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert top_k <= num_experts, "top_k must not exceed num_experts"
        self.num_experts = num_experts
        self.top_k = top_k

        # Linear router: maps each token to a score per expert.
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)

        # Expert FFNs — identical architecture, independently parameterised.
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, ffn_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, hidden_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, T, H) input tensor.

        Returns:
            output: (B, T, H) weighted sum of top-K expert outputs.
            aux_loss: Scalar load-balancing loss; add to main loss ×  moe_aux_loss_coeff.
        """
        B, T, H = x.shape
        x_flat = x.reshape(-1, H)  # (N, H), N = B*T

        router_logits = self.router(x_flat)          # (N, E)
        router_probs = F.softmax(router_logits, dim=-1)  # (N, E)

        topk_weights, topk_indices = router_probs.topk(self.top_k, dim=-1)  # (N, K)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # renormalise

        # Compute all expert outputs and gather top-K — avoids a Python loop over N.
        # all_expert_out: (E, N, H)
        all_expert_out = torch.stack([expert(x_flat) for expert in self.experts], dim=0)

        # Gather: for each of the K selected experts, pick the corresponding row.
        # topk_indices: (N, K) → expand to (K, N) for advanced indexing into (E, N, H).
        N = x_flat.shape[0]
        n_idx = torch.arange(N, device=x.device).unsqueeze(0).expand(self.top_k, -1)  # (K, N)
        gathered = all_expert_out[topk_indices.t(), n_idx]  # (K, N, H)
        gathered = gathered.permute(1, 0, 2)                # (N, K, H)

        # Weighted sum over selected experts.
        output_flat = (gathered * topk_weights.unsqueeze(-1)).sum(dim=1)  # (N, H)

        # Load-balancing auxiliary loss (Switch Transformer style).
        # f_i: fraction of tokens whose top-1 expert is i (hard, not differentiable).
        # P_i: mean router probability for expert i (soft, differentiable).
        top1_indices = topk_indices[:, 0]                                  # (N,)
        f_i = F.one_hot(top1_indices, num_classes=self.num_experts).float().mean(dim=0)  # (E,)
        P_i = router_probs.mean(dim=0)                                     # (E,)
        aux_loss = self.num_experts * (f_i * P_i).sum()

        return output_flat.reshape(B, T, H), aux_loss


# ---------------------------------------------------------------------------
# Adapter layer
# ---------------------------------------------------------------------------


class BESTOWAdapterLayer(nn.Module):
    """
    Single transformer block in the BESTOW cross-attention adapter.

    Applies, in order:
      1. Pre-norm causal self-attention on text embeddings.
      2. Pre-norm cross-attention (text queries speech keys/values).
      3. Pre-norm feed-forward network (dense ``nn.Sequential`` or
         ``BESTOWMoEFFN`` for multilingual).

    Returns ``(output, aux_loss)`` where ``aux_loss`` is a zero scalar for
    dense FFN layers and the MoE load-balancing loss otherwise.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.ffn_norm = nn.LayerNorm(hidden_dim)
        if use_moe:
            self.ffn = BESTOWMoEFFN(
                hidden_dim=hidden_dim,
                ffn_dim=ffn_dim,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, hidden_dim),
                nn.Dropout(dropout),
            )
        self._use_moe = use_moe

    def forward(
        self,
        x: Tensor,
        speech: Tensor,
        causal_mask: Tensor | None,
        speech_key_padding_mask: Tensor | None,
        streaming_attn_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: Text embeddings (B, T, H).
            speech: Projected speech embeddings (B, S, H).
            causal_mask: (T, T) additive upper-triangular mask for causal self-attention.
            speech_key_padding_mask: (B, S) bool, True at padded speech positions.
            streaming_attn_mask: (T, S) additive mask for Wait-K streaming cross-attention.

        Returns:
            Tuple of:
              - Updated text embeddings (B, T, H).
              - Scalar MoE aux loss (zero tensor for dense FFN layers).
        """
        residual = x
        x = self.self_attn_norm(x)
        x, _ = self.self_attn(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = x + residual

        residual = x
        x = self.cross_attn_norm(x)
        x, _ = self.cross_attn(
            query=x,
            key=speech,
            value=speech,
            key_padding_mask=speech_key_padding_mask,
            attn_mask=streaming_attn_mask,
            need_weights=False,
        )
        x = x + residual

        residual = x
        x = self.ffn_norm(x)
        if self._use_moe:
            x, aux_loss = self.ffn(x)
        else:
            x = self.ffn(x)
            aux_loss = x.new_zeros(1).squeeze()
        x = x + residual

        return x, aux_loss


# ---------------------------------------------------------------------------
# Cross-attention adapter (main public API)
# ---------------------------------------------------------------------------


class BESTOWCrossAttentionAdapter(nn.Module):
    """
    BESTOW cross-attention adapter that fuses speech context into LLM text embeddings.

    Speech encoder outputs serve as keys/values; text embeddings are queries.
    Compared to concatenation-based approaches (e.g. SALM), cross-attention reduces
    LLM self-attention complexity from O((T_text + T_speech)²) to
    O(T_text·T_speech + T_text²), yielding significant speedups when speech is long.

    A linear projection aligns the speech encoder hidden dimension with the LLM hidden
    dimension before cross-attention. When the two dims match, no projection is applied.

    Multilingual extensions (all optional)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``num_languages`` > 0
        Enables language-ID conditioning. A learnable embedding table (num_languages, H)
        maps each sample's language ID (int) to a vector that is broadcast over the
        speech time axis and added to projected speech features before cross-attention.
        This gives the cross-attention explicit phonological cues without adding
        parameters inside the attention mechanism.

    ``use_moe=True``
        Replaces the dense FFN in every adapter block with ``BESTOWMoEFFN``
        (num_experts experts, top_k routing).  The adapter's ``forward`` returns a
        non-zero ``aux_loss`` that the model's training_step should add to the CE loss
        weighted by ``moe_aux_loss_coeff`` (config field on the BESTOW model, default 0.01).

    Args:
        text_dim: LLM hidden dimension.
        speech_dim: Speech encoder output dimension.
        num_heads: Number of attention heads (shared by self- and cross-attention).
        num_layers: Number of stacked adapter transformer blocks (default 2, per paper).
        ffn_expansion: FFN hidden-size multiplier relative to text_dim (default 4).
        dropout: Dropout applied in attention and FFN sub-layers.
        use_moe: Replace dense FFN with sparse MoE FFN (default False).
        num_experts: Number of MoE expert FFNs per layer (used when use_moe=True).
        top_k: Number of experts each token is dispatched to (used when use_moe=True).
        num_languages: Vocabulary size for language-ID conditioning (0 = disabled).
    """

    def __init__(
        self,
        text_dim: int,
        speech_dim: int,
        num_heads: int,
        num_layers: int = 2,
        ffn_expansion: int = 4,
        dropout: float = 0.0,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        num_languages: int = 0,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.speech_dim = speech_dim
        self._use_moe = use_moe
        self._num_languages = num_languages

        self.speech_proj = (
            nn.Linear(speech_dim, text_dim, bias=False) if speech_dim != text_dim else nn.Identity()
        )

        # Optional language conditioning: language_id → embedding added to speech features.
        # This shifts speech representations toward language-specific acoustic subspaces.
        self.lang_embed = nn.Embedding(num_languages, text_dim) if num_languages > 0 else None

        self.layers = nn.ModuleList(
            [
                BESTOWAdapterLayer(
                    hidden_dim=text_dim,
                    num_heads=num_heads,
                    ffn_dim=text_dim * ffn_expansion,
                    dropout=dropout,
                    use_moe=use_moe,
                    num_experts=num_experts,
                    top_k=top_k,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(text_dim)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> Tensor:
        """Upper-triangular additive mask: 0 on/below diagonal, -inf above."""
        return torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1,
        )

    @staticmethod
    def _streaming_mask(T: int, S: int, wait_k: int, stride: int, device: torch.device) -> Tensor:
        """
        Additive (T, S) cross-attention mask for Wait-K streaming.

        Text position t may attend to speech frames 0 .. (wait_k + t) * stride - 1.
        Frames at or beyond the boundary receive -inf (blocked), others receive 0.
        """
        t_idx = torch.arange(T, device=device).unsqueeze(1)  # (T, 1)
        s_idx = torch.arange(S, device=device).unsqueeze(0)  # (1, S)
        boundary = (wait_k + t_idx) * stride                 # (T, 1)
        return torch.where(
            s_idx < boundary,
            torch.zeros(1, device=device),
            torch.full((1,), float('-inf'), device=device),
        )  # (T, S)

    def forward(
        self,
        text_embeds: Tensor,
        speech_embeds: Tensor,
        speech_padding_mask: Tensor | None = None,
        language_ids: Tensor | None = None,
        wait_k: int | None = None,
        stride: int = 4,
    ) -> tuple[Tensor, Tensor]:
        """
        Fuse speech context into text embeddings.

        Args:
            text_embeds: (B, T, H_text) LLM input embeddings.
            speech_embeds: (B, S, H_speech) speech encoder outputs.
            speech_padding_mask: (B, S) bool, True at padded/invalid speech positions.
            language_ids: (B,) int64 language IDs, or None when language conditioning
                          is disabled or language labels are unavailable.
            wait_k: Wait-K initial context for streaming (K in the paper).
                    None = offline mode (full speech context at every position).
            stride: Speech encoder frames per generated text token (L in the paper,
                    default 4). Only used when wait_k is not None.

        Returns:
            Tuple of:
              - (B, T, H_text) adapted text embeddings.
              - Scalar MoE auxiliary loss (zero when use_moe=False).
        """
        B, T, _ = text_embeds.shape
        S = speech_embeds.shape[1]
        device = text_embeds.device

        # Project speech to LLM hidden dimension.
        speech = self.speech_proj(speech_embeds)   # (B, S, H_text)

        # Language-ID conditioning: add language embedding to every speech frame.
        if self.lang_embed is not None and language_ids is not None:
            lang_vec = self.lang_embed(language_ids)  # (B, H_text)
            speech = speech + lang_vec.unsqueeze(1)   # broadcast over S

        causal_mask = self._causal_mask(T, device)
        streaming_mask = (
            self._streaming_mask(T, S, wait_k, stride, device) if wait_k is not None else None
        )

        x = text_embeds
        total_aux_loss = text_embeds.new_zeros(1).squeeze()
        for layer in self.layers:
            x, layer_aux_loss = layer(x, speech, causal_mask, speech_padding_mask, streaming_mask)
            total_aux_loss = total_aux_loss + layer_aux_loss

        x = self.output_norm(x)
        # Global residual: add original text embeddings to preserve pre-trained LLM representations.
        return x + text_embeds, total_aux_loss
