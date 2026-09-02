# Spec: MoE Conformer Layers in `EncDecRNNTBPEModelWithPrompt`

**Target:** `nemo/collections/asr/models/rnnt_bpe_models_prompt.py` — `EncDecRNNTBPEModelWithPrompt`
**Encoder:** `nemo/collections/asr/modules/conformer_encoder.py` — `ConformerEncoder`

---

## Problem Statement

<!-- TODO: describe motivation — why MoE in the conformer FF layers? capacity, multilingual routing, etc. --> I want to implement conformer based multilingual moe to nemotron architecture to improve it performance

## Goals

<!-- TODO: measurable outcomes (WER, parameter efficiency, throughput, routing entropy, etc.) --> Implement moe based routing for language specific routing to enhance moe features for conformer. You can find similar reference code here https://arxiv.org/abs/2305.15663

## Non-Goals

<!-- TODO: e.g. MoE in attention, decoder, joint — scope to encoder FF only? --> Yeah changes only in encoder FF as described the attached paper from goals section. Write a resultant config as well

---

## Approach

<!-- TODO: which MoE variant? (top-k, switch, expert-choice, soft-MoE)
     Which layers get MoE? (all FF, every-N, last-K layers?)
     New module vs. swap existing ConformerLayer FF sub-module? -->
     I want to make the approach as configurable and all the changes to be applied to layers as specified in the paper. I want to apply only to last K layers of conformer (K to be taken from config). And also the moe variant should also be taken from config. 

## Key Changes

<!-- TODO: files / classes / methods to touch -->

## Config Shape

<!-- TODO: sketch the new YAML knobs, e.g.
     model.encoder.feed_forward.moe.enabled
     model.encoder.feed_forward.moe.num_experts
     model.encoder.feed_forward.moe.top_k
     model.encoder.feed_forward.moe.apply_at_layers
     model.encoder.feed_forward.moe.aux_loss_weight -->

## Testing Plan

<!-- TODO: existing tests to run; new unit tests for routing / load-balancing loss -->

---

## Open Questions

<!-- TODO: e.g. expert load balancing loss integration with existing RNNT loss?
           distributed expert placement with DDP/FSDP2?
           checkpoint compatibility with dense pretrained weights? -->
