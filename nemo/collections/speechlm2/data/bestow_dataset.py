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
BESTOW dataset.

Extends SALMDataset to extract per-sample language IDs from Lhotse cut metadata
and add them to the batch as a ``language_ids`` int64 tensor.

Language flow through the data pipeline
----------------------------------------
1. The YAML config's ``tags: language: "en"`` is processed by
   ``attach_tags(conversation, {"language": "en"})`` in ``cutset.py``.
2. ``attach_tags`` calls ``setattr(conversation, "language", "en")``.
   Because ``NeMoMultimodalConversation`` inherits ``CustomFieldMixin``,
   unknown attributes are stored in ``conversation.custom``, so the result is
   ``conversation.custom["language"] = "en"``.
3. ``SALMDataset.__getitem__`` stores the ``CutSet`` of conversations under the
   ``"conversations"`` key in the batch dict but never reads ``custom["language"]``.
4. ``BESTOWDataset.__getitem__`` (this class) post-processes that batch: it reads
   ``conversation.custom.get("language", "unknown")`` for each conversation and
   maps it to an integer using the ``language_to_id`` dict supplied at construction
   time.  The resulting tensor is stored as ``batch["language_ids"]``.

When ``language_to_id`` is empty or ``None`` (default BESTOW without language
conditioning), ``language_ids`` is not added to the batch and the model's
``prepare_inputs`` safely falls back to ``batch.get("language_ids", None) → None``.
"""
import torch

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.salm_dataset import SALMDataset


class BESTOWDataset(SALMDataset):
    """
    Dataset for BESTOW that optionally adds ``language_ids`` to each batch.

    When ``language_to_id`` is provided, the dataset reads the ``"language"`` key
    from each conversation's ``custom`` metadata (set by Lhotse's ``attach_tags``
    from the YAML ``tags`` field) and converts it to an integer tensor.  This
    integer is consumed by ``BESTOWCrossAttentionAdapter`` for language-ID
    conditioning of the cross-attention layers.

    Args:
        tokenizer: Model tokenizer (same as SALMDataset).
        language_to_id: Dict mapping ISO language code string → integer ID.
            Example: ``{"unknown": 0, "en": 1, "de": 2, "fr": 3, "es": 4}``.
            A missing language string defaults to the "unknown" entry (id 0) or 0
            if "unknown" is also absent.  Pass ``None`` or ``{}`` to disable
            language-ID conditioning (no ``language_ids`` key in batch).
        multispeaker_cfg: Forwarded to SALMDataset (optional SOT config).
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        language_to_id: dict[str, int] | None = None,
        multispeaker_cfg: dict | None = None,
    ) -> None:
        super().__init__(tokenizer=tokenizer, multispeaker_cfg=multispeaker_cfg)
        self.language_to_id: dict[str, int] = language_to_id or {}
        self._unknown_id: int = self.language_to_id.get("unknown", 0)

    def __getitem__(self, conversations):
        batch = super().__getitem__(conversations)

        # Parent returns None when all conversations fail to load.
        if batch is None or not self.language_to_id:
            return batch

        # Conversations in the batch are NeMoMultimodalConversation objects
        # stored under "conversations".  The language tag was attached to each
        # conversation's ``custom`` dict by Lhotse's attach_tags.
        lang_ids = []
        for conv in batch["conversations"]:
            lang = "unknown"
            if conv.custom and "language" in conv.custom:
                lang = conv.custom["language"]
            lang_ids.append(self.language_to_id.get(lang, self._unknown_id))

        batch["language_ids"] = torch.tensor(lang_ids, dtype=torch.long)
        return batch
