# Spec: Optimize `EncDecRNNTBPEModelWithPrompt`

**Target:** `nemo/collections/asr/models/rnnt_bpe_models_prompt.py` — `EncDecRNNTBPEModelWithPrompt`

---

## Problem Statement

<!-- TODO: describe what is slow / incorrect / expensive --> I am training a multilingual asr model using rnnt_bpe_with_prompt style architecture using nemotron3.5-asr style weights. I want to dynamic data balancing (I have a list of validation_sets (which is per locale)) Now based on the validation set which has lower score I will somehow proportianetly modify the weights of those language shars so that I get faster convergence

## Goals

<!-- TODO: list measurable outcomes (latency, memory, accuracy, etc.) -->
Validation will be per locale.. Based on per locale scores I want to modify the weights of per locale training shars... Lhotse there are group level weights so I want to modify those grp level weights

## Non-Goals

<!-- TODO: what this spec deliberately does not change --> Other training related details / parameters should not be impacted by this dynamic weighting concept.... (refer to original source code as well here https://github.com/lhotse-speech/lhotse) Make all of these changes configurable

---

## Approach

<!-- TODO: describe the optimization strategy --> Optimize the code 
Get language wise validation loss and based on language wise changes (modify the shar reweighting for each language... Can you also share the sample config for this training)

## Key Changes

<!-- TODO: list files / methods to touch -->

## Testing Plan

<!-- TODO: which existing tests to run; any new tests needed -->
<!-- Relevant test file: tests/collections/asr/test_asr_rnnt_encoder_model_bpe_prompt.py -->

---

## Open Questions

<!-- TODO -->
