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
BESTOW: Efficient and Streamable Speech Language Model with the Best of Two Worlds in GPT and T5.

Reference: arXiv:2406.19954

Architecture overview
---------------------
Unlike SALM, which inserts encoded speech embeddings into the text token sequence via a
placeholder token, BESTOW fuses speech into the LLM via a lightweight cross-attention
adapter placed **before** the LLM layers.  Speech is the key/value source; text tokens
are the queries.  The LLM then receives adapted text embeddings whose length equals only
the text sequence length (not text + speech), reducing LLM self-attention cost from
O((T_text + T_speech)²) to O(T_text * T_speech + T_text²).

Data flow (training)
--------------------
  audios → perception (ASR encoder + modality adapter) → speech_embs  (B, S, H_speech)
  input_ids → embed_tokens → text_embs                                 (B, T, H_llm)
  cross_attention_adapter(text_embs, speech_embs) → adapted_embs      (B, T, H_llm)
  adapted_embs[:, :-1] → LLM → logits                                 (B, T-1, V)
  loss = CE(logits, input_ids[:, 1:], masked by loss_mask[:, 1:])

Streaming (BESTOW-S)
--------------------
Set ``model.streaming.enabled = true`` and optionally adjust ``k_min``/``k_max`` (training)
or pass ``wait_k`` to ``generate()`` at inference time.  The Wait-K policy masks the
cross-attention so text position t attends only to speech frames 0 .. (wait_k + t) * stride - 1,
where stride (L=4 per the paper) is the number of speech encoder frames per text token.

During training, ``wait_k`` is sampled uniformly from [k_min, k_max] per batch, enabling
a single checkpoint to serve multiple latency budgets at inference.
"""
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import torch
from lhotse import CutSet
from lightning import LightningModule
from omegaconf import DictConfig
from peft import PeftModel
from torch import Tensor
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    loss_parallel,
    parallelize_module,
)
from transformers import GenerationConfig

from nemo.collections.common.prompts import PromptFormatter
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.salm_dataset import left_collate_vectors
from nemo.collections.speechlm2.modules.bestow_adapter import BESTOWCrossAttentionAdapter
from nemo.collections.speechlm2.parts.encoder_chunking import encode_audio_with_optional_chunking
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.pretrained import (
    load_pretrained_hf,
    maybe_load_pretrained_models,
    setup_speech_encoder,
)
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, MaskType, NeuralType
from nemo.utils import logging


def _pad_speech_embeddings(
    speech_emb_list: list[Tensor],
) -> tuple[Tensor, Tensor]:
    """Pad a list of per-sample speech embeddings into a batch tensor with a boolean padding mask.

    Args:
        speech_emb_list: List of B tensors, each (T_i, H_speech).

    Returns:
        padded: (B, T_max, H_speech) zero-padded batch tensor.
        padding_mask: (B, T_max) bool, True at positions that are padding (should be ignored).
    """
    B = len(speech_emb_list)
    H = speech_emb_list[0].shape[1]
    device = speech_emb_list[0].device
    dtype = speech_emb_list[0].dtype

    lengths = torch.tensor([e.shape[0] for e in speech_emb_list], device=device, dtype=torch.long)
    T_max = int(lengths.max().item()) if B > 0 else 0

    padded = torch.zeros(B, T_max, H, device=device, dtype=dtype)
    for i, emb in enumerate(speech_emb_list):
        L = emb.shape[0]
        padded[i, :L] = emb

    # True where position exceeds valid length → these should be masked out in cross-attention.
    padding_mask = torch.arange(T_max, device=device).unsqueeze(0) >= lengths.unsqueeze(1)  # (B, T_max)
    return padded, padding_mask


class BESTOW(LightningModule, HFHubMixin):
    """
    BESTOW speech language model.

    Wraps a pretrained HuggingFace causal LM with a BESTOW cross-attention adapter that
    fuses speech encoder outputs into the LLM input stream without expanding the token sequence.

    Required config fields
    ----------------------
    pretrained_llm: str
        HuggingFace model name or local path for the LLM backbone.
    pretrained_asr: str
        NeMo model name or .nemo path for the speech encoder.
    pretrained_weights: bool
        Whether to load pretrained weights (True) or random-init (False).
    perception: DictConfig
        Config for AudioPerceptionModule (preprocessor, encoder, modality_adapter, output_dim).
    adapter: DictConfig
        Cross-attention adapter config with fields:
          num_heads (int)         – attention heads (must divide LLM hidden_size)
          num_layers (int)        – number of adapter blocks (default 2)
          ffn_expansion (int)     – FFN hidden-size multiplier (default 4)
          dropout (float)         – dropout probability (default 0.0)
    optimizer: DictConfig
        Hydra-style optimizer config (passed to configure_optimizers).
    prompt_format: str
        PromptFormatter class name (e.g. 'llama3', 'gemma').

    Optional config fields
    ----------------------
    tokenizer_path: str
        Override tokenizer source (defaults to pretrained_llm).
    freeze_params: list[str]
        Regex patterns for frozen parameters.
    prevent_freeze_params: list[str]
        Regex patterns that override freeze_params (keep trainable).
    lr_scheduler: DictConfig
        Hydra-style LR scheduler config.
    encoder_chunk_size_seconds: float
        Split long audio into chunks of this duration before encoding.
    streaming:
        enabled: bool        – enable streaming Wait-K training (default False)
        k_min: int           – minimum wait-K value sampled during training (default 3)
        k_max: int           – maximum wait-K value sampled during training (default 12)
        stride: int          – speech frames per text token, L in the paper (default 4)
    trust_remote_code: bool
        Passed to from_pretrained for models with custom code (default False).
    """

    def __init__(self, cfg) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to BESTOW as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        self.cfg = DictConfig(cfg)

        tokenizer_src = self.cfg.get("tokenizer_path", None) or self.cfg.pretrained_llm
        self.tokenizer = AutoTokenizer(
            tokenizer_src, use_fast=True, trust_remote_code=self.cfg.get("trust_remote_code", False)
        )

        self.llm = load_pretrained_hf(
            self.cfg.pretrained_llm,
            pretrained_weights=self.cfg.pretrained_weights,
            trust_remote_code=self.cfg.get("trust_remote_code", False),
        )
        # Extract embed_tokens outside the LLM to avoid FSDP/TP hook interference
        # (same pattern as SALM; see salm.py for the detailed comment).
        self.embed_tokens = self.llm.model.embed_tokens
        del self.llm.model.embed_tokens

        maybe_install_lora(self)
        setup_speech_encoder(self, pretrained_weights=self.cfg.pretrained_weights)
        maybe_load_pretrained_models(self)

        # Cross-attention adapter: speech_dim is the perception module's output dim
        # (already projected to a fixed size inside AudioPerceptionModule.proj).
        speech_dim = self.cfg.perception.output_dim
        text_dim = self.llm.config.hidden_size
        adapter_cfg = self.cfg.adapter
        self.cross_attention_adapter = BESTOWCrossAttentionAdapter(
            text_dim=text_dim,
            speech_dim=speech_dim,
            num_heads=adapter_cfg.num_heads,
            num_layers=adapter_cfg.get("num_layers", 2),
            ffn_expansion=adapter_cfg.get("ffn_expansion", 4),
            dropout=adapter_cfg.get("dropout", 0.0),
            use_moe=adapter_cfg.get("use_moe", False),
            num_experts=adapter_cfg.get("num_experts", 8),
            top_k=adapter_cfg.get("top_k", 2),
            num_languages=adapter_cfg.get("num_languages", 0),
        )

        self._use_fsdp = False
        self._use_tp = False

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def text_vocab_size(self) -> int:
        return self.embed_tokens.num_embeddings

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        pad_id = self.tokenizer.pad
        if pad_id is None:
            pad_id = self.tokenizer.unk_id
        if pad_id is None:
            warnings.warn(
                "The text tokenizer has no <pad> or <unk> token; using id 0 for padding (may cause silent bugs)."
            )
            pad_id = 0
        return pad_id

    @property
    def token_equivalent_duration(self) -> float:
        return self.perception.token_equivalent_duration

    @property
    def sampling_rate(self) -> int:
        return self.perception.preprocessor.featurizer.sample_rate

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(self, input_embeds: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        """
        Run the LLM on already-adapted text embeddings.

        The cross-attention adapter must be applied **before** calling this
        method (see ``prepare_inputs``).

        Args:
            input_embeds: (B, T, H) adapted text embeddings.
            attention_mask: (B, T) bool, True at non-padding positions.

        Returns:
            dict with key ``"logits"`` of shape (B, T, vocab_size).
        """
        out = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return {"logits": out["logits"]}

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def prepare_inputs(self, batch: dict) -> dict[str, Tensor]:
        """
        Encode audio, embed text tokens, apply the cross-attention adapter.

        Batch keys expected:
          audios     – (B, T_samples) float32 waveform batch.
          audio_lens – (B,) int64 valid sample counts.
          input_ids  – (B, T_text) int64 text token ids (left-padded).
          loss_mask  – (B, T_text) bool, True where loss should be computed.

        Returns a dict with:
          input_embeds   – (B, T_text-1, H) adapter-transformed text embeddings.
          attention_mask – (B, T_text-1) bool attention mask.
          target_ids     – (B, T_text-1) int64 targets (-100 at masked positions).
        """
        # 1. Encode audio → list of (T_i, H_speech) tensors.
        speech_emb_list = encode_audio_with_optional_chunking(
            self.perception,
            batch["audios"],
            batch["audio_lens"],
            chunk_size_seconds=self.cfg.get("encoder_chunk_size_seconds", None),
            sampling_rate=self.sampling_rate,
        )

        # 2. Pad speech embeddings into a batch tensor + padding mask.
        speech_embs, speech_padding_mask = _pad_speech_embeddings(speech_emb_list)

        # 3. Embed text tokens.
        input_ids = batch["input_ids"]
        text_embs = self.embed_tokens(input_ids)                         # (B, T, H_llm)
        attention_mask = (input_ids != self.text_pad_id)                 # (B, T)
        target_ids = input_ids.where(batch["loss_mask"], -100)           # (B, T)

        # 4. Resolve Wait-K for streaming training.
        wait_k = None
        streaming_cfg = self.cfg.get("streaming", {})
        if self.training and streaming_cfg.get("enabled", False):
            k_min = int(streaming_cfg.get("k_min", 3))
            k_max = int(streaming_cfg.get("k_max", 12))
            wait_k = int(torch.randint(k_min, k_max + 1, (1,)).item())

        stride = int(streaming_cfg.get("stride", 4))

        # 5. Apply cross-attention adapter.
        # language_ids come from the batch when language conditioning is enabled.
        language_ids = batch.get("language_ids", None)
        adapted_embs, moe_aux_loss = self.cross_attention_adapter(
            text_embs, speech_embs, speech_padding_mask,
            language_ids=language_ids,
            wait_k=wait_k,
            stride=stride,
        )                                                                # (B, T, H_llm)

        # 6. Optionally truncate for TP divisibility (same pattern as SALM).
        if self._use_tp:
            tp_world_size = self.device_mesh["tensor_parallel"].size()
            if (remainder := (adapted_embs.shape[1] - 1) % tp_world_size) != 0:
                adapted_embs = adapted_embs[:, :-remainder]
                attention_mask = attention_mask[:, :-remainder]
                target_ids = target_ids[:, :-remainder]

        # 7. Shift for teacher-forcing: input is all but last, target is all but first.
        return {
            "input_embeds": adapted_embs[:, :-1],
            "attention_mask": attention_mask[:, :-1],
            "target_ids": target_ids[:, 1:],
            "moe_aux_loss": moe_aux_loss,
        }

    # ------------------------------------------------------------------
    # Training / validation / test
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int):
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm):
            if is_frozen(m):
                m.eval()

        inputs = self.prepare_inputs(batch)
        forward_outputs = self(inputs["input_embeds"], attention_mask=inputs["attention_mask"])
        num_frames = (inputs["target_ids"] != -100).long().sum()
        with loss_parallel():
            ce_loss = (
                torch.nn.functional.cross_entropy(
                    forward_outputs["logits"].flatten(0, 1),
                    inputs["target_ids"].flatten(0, 1),
                    reduction="sum",
                    ignore_index=-100,
                )
                / num_frames
            )
        moe_coeff = self.cfg.adapter.get("moe_aux_loss_coeff", 0.01)
        moe_aux_loss = inputs["moe_aux_loss"]
        loss = ce_loss + moe_coeff * moe_aux_loss

        B, T = inputs["input_embeds"].shape[:2]
        ans = {
            "loss": loss,
            "ce_loss": ce_loss,
            "moe_aux_loss": moe_aux_loss,
            "learning_rate": torch.as_tensor(
                self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0
            ),
            "batch_size": B,
            "sequence_length": T,
            "num_frames": num_frames.to(torch.float32),
            "target_to_input_ratio": num_frames / (B * T),
            "padding_ratio": (batch["input_ids"] != self.text_pad_id).long().sum() / batch["input_ids"].numel(),
        }
        self.log("loss", loss, on_step=True, prog_bar=True)
        self.log_dict({k: v for k, v in ans.items() if k != "loss"}, on_step=True)
        return ans

    def on_validation_epoch_start(self) -> None:
        self._partial_val_losses: dict[str, list] = defaultdict(list)
        self._partial_accuracies: dict[str, list] = defaultdict(list)

    def on_validation_epoch_end(self) -> None:
        val_losses = []
        for name, vals in self._partial_val_losses.items():
            val_loss = torch.stack(vals).mean()
            self.log(f"val_loss_{name}", val_loss, on_epoch=True, sync_dist=True)
            val_losses.append(val_loss)
        if val_losses:
            self.log("val_loss", torch.stack(val_losses).mean(), on_epoch=True, sync_dist=True)

        accuracies = []
        for name, accs in self._partial_accuracies.items():
            val_acc = torch.stack(accs).mean()
            self.log(f"val_acc_{name}", val_acc, on_epoch=True, sync_dist=True)
            accuracies.append(val_acc)
        if accuracies:
            self.log("val_acc", torch.stack(accuracies).mean(), on_epoch=True, sync_dist=True)

        self._partial_val_losses.clear()
        self._partial_accuracies.clear()

    def validation_step(self, batch: dict, batch_idx: int):
        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue
            inputs = self.prepare_inputs(dataset_batch)
            forward_outputs = self(inputs["input_embeds"], attention_mask=inputs["attention_mask"])
            num_frames = (inputs["target_ids"] != -100).long().sum()
            with loss_parallel():
                loss = (
                    torch.nn.functional.cross_entropy(
                        forward_outputs["logits"].flatten(0, 1),
                        inputs["target_ids"].flatten(0, 1),
                        reduction="sum",
                        ignore_index=-100,
                    )
                    / num_frames
                )

            preds = forward_outputs["logits"].argmax(dim=-1).view(-1)
            refs = inputs["target_ids"].reshape(-1)
            preds = preds[refs != -100]
            refs = refs[refs != -100]
            accuracy = preds.eq(refs).float().mean()

            self._partial_accuracies[name].append(accuracy)
            self._partial_val_losses[name].append(loss)

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end()

    def test_step(self, *args: Any, **kwargs: Any):
        return self.validation_step(*args, **kwargs)

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompts: list[list[dict]] | Tensor,
        audios: Optional[Tensor] = None,
        audio_lens: Optional[Tensor] = None,
        max_new_tokens: int = 128,
        wait_k: Optional[int] = None,
        stride: int = 4,
        language_ids: Optional[Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        enable_thinking: Optional[bool] = None,
        **generation_kwargs,
    ) -> Tensor:
        """
        Generate text from speech+text prompts using BESTOW cross-attention.

        Unlike SALM, BESTOW does not insert speech tokens into the text sequence.
        The cross-attention adapter is re-applied at every generation step so the
        LLM can attend to the appropriate speech context at each position.

        Args:
            prompts: Batch of prompts as list[list[dict]] (chat-format) or a pre-tokenized
                     int64 Tensor of shape (B, T_prompt).
            audios: (B, T_samples) float32 waveform batch. May be embedded inside
                    ``prompts`` via the ``"audio"`` turn key instead.
            audio_lens: (B,) int64 valid sample counts corresponding to ``audios``.
            max_new_tokens: Maximum number of tokens to generate per prompt.
            wait_k: Wait-K initial context for streaming inference.
                    None = offline (full speech context at every position).
                    int K = streaming: text position t attends to speech frames 0..(K+t)*stride-1.
            stride: Speech frames per text token (L=4 in the paper). Effective only
                    when ``wait_k`` is not None.
            generation_config: HuggingFace GenerationConfig for sampling strategy.
            enable_thinking: Optional flag forwarded to PromptFormatter.encode_dialog.
            **generation_kwargs: Unused (reserved for API compatibility).

        Returns:
            (B, max_new_tokens) int64 tensor of generated token ids (padded with text_pad_id).

        Note:
            This implementation re-applies the cross-attention adapter at every generation step,
            which is O(T²) in the generated sequence length. Production deployments should
            cache adapter KV states between steps for O(T) amortised cost per step.
        """
        # ------ Resolve prompts ------
        if isinstance(prompts, Tensor):
            tokens = prompts
        else:
            if (
                maybe_audio := _resolve_audios_in_prompt(
                    prompts, sampling_rate=self.sampling_rate, device=self.device
                )
            ) is not None:
                assert audios is None and audio_lens is None, (
                    "Audios cannot be provided via both ``prompts`` and ``audios``/``audio_lens``."
                )
                audios, audio_lens = maybe_audio
            formatter = PromptFormatter.resolve(self.cfg.prompt_format)(self.tokenizer)
            fmt_kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
            tokens = left_collate_vectors(
                [formatter.encode_dialog(turns=p, **fmt_kwargs)["input_ids"] for p in prompts],
                padding_value=self.text_pad_id,
            ).to(self.device)

        # ------ Encode audio ------
        if audios is not None:
            speech_emb_list = encode_audio_with_optional_chunking(
                self.perception,
                audios,
                audio_lens,
                chunk_size_seconds=self.cfg.get("encoder_chunk_size_seconds", None),
                sampling_rate=self.sampling_rate,
            )
            speech_embs, speech_padding_mask = _pad_speech_embeddings(speech_emb_list)
        else:
            speech_embs, speech_padding_mask = None, None

        B = tokens.shape[0]
        prompt_len = tokens.shape[1]
        generated = tokens                                    # (B, T_current)
        done = torch.zeros(B, dtype=torch.bool, device=self.device)

        for step in range(max_new_tokens):
            # Embed all tokens generated so far (prompt + previous outputs).
            text_embs = self.embed_tokens(generated)          # (B, T_current, H)

            if speech_embs is not None:
                adapted, _ = self.cross_attention_adapter(
                    text_embs, speech_embs, speech_padding_mask,
                    language_ids=language_ids,
                    wait_k=wait_k,
                    stride=stride,
                )
            else:
                adapted = text_embs

            attn_mask = (generated != self.text_pad_id)       # (B, T_current)
            out = self.llm(inputs_embeds=adapted, attention_mask=attn_mask, return_dict=True)
            next_token = out.logits[:, -1].argmax(dim=-1, keepdim=True)  # (B, 1) greedy

            # For sequences that already finished, emit padding.
            next_token = torch.where(
                done.unsqueeze(1),
                torch.full_like(next_token, self.text_pad_id),
                next_token,
            )
            generated = torch.cat([generated, next_token], dim=1)
            done = done | (next_token.squeeze(1) == self.text_eos_id)
            if done.all():
                break

        return generated[:, prompt_len:]                       # return only new tokens

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        return configure_optimizers(self)

    # ------------------------------------------------------------------
    # Distributed (FSDP2 + Tensor Parallel) — mirrors SALM.configure_model
    # ------------------------------------------------------------------

    def configure_model(self) -> None:
        device_mesh = self.device_mesh
        if device_mesh is None:
            return

        llm = self.llm
        if isinstance(llm, PeftModel):
            llm = llm.base_model.model

        if (tp_mesh := device_mesh["tensor_parallel"]).size() > 1:
            self._use_tp = True

            plan = {
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(),),
                    desired_input_layouts=(Shard(1),),
                    use_local_output=True,
                ),
                "norm": SequenceParallel(),
            }
            parallelize_module(llm, tp_mesh, plan)

            for transformer_block in llm.model.layers:
                plan = {
                    "input_layernorm": SequenceParallel(),
                    "self_attn.q_proj": ColwiseParallel(),
                    "self_attn.k_proj": ColwiseParallel(),
                    "self_attn.v_proj": ColwiseParallel(),
                    "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                    "post_attention_layernorm": SequenceParallel(),
                    "mlp": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "mlp.gate_proj": ColwiseParallel(),
                    "mlp.up_proj": ColwiseParallel(),
                    "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                }

                attn_layer = transformer_block.self_attn
                for attr in ("num_heads", "num_key_value_heads", "hidden_size"):
                    val = getattr(attn_layer, attr)
                    if val % tp_mesh.size() != 0:
                        logging.warning(
                            f"attn_layer.{attr}={val} is not divisible by {tp_mesh.size()=}: "
                            "set a different TP size to avoid errors."
                        )
                    setattr(attn_layer, attr, val // tp_mesh.size())

                parallelize_module(transformer_block, tp_mesh, plan)

            parallelize_module(
                llm.lm_head,
                tp_mesh,
                ColwiseParallel(
                    input_layouts=Shard(1),
                    output_layouts=Shard(-1),
                    use_local_output=False,
                ),
            )

        if (dp_mesh := device_mesh["data_parallel"]).size() > 1:
            assert dp_mesh.ndim == 1, "Hybrid-sharding not supported"
            self._use_fsdp = True
            fsdp_config = {"mesh": dp_mesh}
            for idx, layer in enumerate(llm.model.layers):
                llm.model.layers[idx] = fully_shard(layer, **fsdp_config)
            self.embed_tokens = fully_shard(self.embed_tokens, **fsdp_config)
            self.cross_attention_adapter = fully_shard(self.cross_attention_adapter, **fsdp_config)
            llm.lm_head = fully_shard(llm.lm_head, **fsdp_config)
            self.llm = fully_shard(self.llm, **fsdp_config)
            self.perception = fully_shard(self.perception, **fsdp_config)

    # ------------------------------------------------------------------
    # OOMptimizer schema
    # ------------------------------------------------------------------

    @property
    def oomptimizer_schema(self) -> dict:
        return {
            "cls": dict,
            "inputs": [
                {"name": "audios", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {
                    "name": "input_ids",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": self.text_vocab_size,
                },
                {"name": "loss_mask", "type": NeuralType(("B", "T"), MaskType()), "seq_length": "output"},
            ],
        }


# ------------------------------------------------------------------
# Audio resolution helper (same as in salm.py)
# ------------------------------------------------------------------


def _resolve_audios_in_prompt(
    prompts: list[list[dict]], sampling_rate: int, device: str | torch.device
) -> tuple[Tensor, Tensor] | None:
    from lhotse import Recording

    paths = []
    for conversation in prompts:
        for turn in conversation:
            if "audio" in turn:
                turn_audio = turn["audio"]
                if isinstance(turn_audio, (str, Path)):
                    turn_audio = [turn_audio]
                for p in turn_audio:
                    assert isinstance(p, (str, Path)), f"Invalid value under prompt key 'audio': {p}"
                    paths.append(p)
    if not paths:
        return None
    cuts = CutSet([Recording.from_file(p).to_cut() for p in paths])
    with torch.device("cpu"):
        audio, audio_lens = cuts.resample(sampling_rate).load_audio(collate=True)
    return (
        torch.as_tensor(audio).to(device, non_blocking=True),
        torch.as_tensor(audio_lens).to(device, non_blocking=True),
    )
