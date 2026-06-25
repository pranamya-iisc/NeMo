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
BESTOW training script.

Usage (single node, all GPUs):

  torchrun --nproc-per-node=$(nvidia-smi -L | wc -l) examples/speechlm2/bestow_train.py \
    --config-path conf --config-name bestow \
    data.train_ds.input_cfg.0.cuts_path=/path/to/train.jsonl.gz \
    data.validation_ds.datasets.val_set_0.input_cfg.0.cuts_path=/path/to/val.jsonl.gz

Override any field with standard Hydra syntax.  See examples/speechlm2/conf/bestow.yaml
and examples/speechlm2/conf/bestow_multilingual.yaml for full config reference.

Streaming training (BESTOW-S):

  ... model.streaming.enabled=true model.streaming.k_min=3 model.streaming.k_max=12

Language-conditioned MoE (multilingual):

  ... --config-name bestow_multilingual
"""
import os

import torch
from lightning.pytorch import Trainer, seed_everything
from omegaconf import OmegaConf

from nemo.collections.speechlm2 import BESTOW, BESTOWDataset, DataModule, SALMDataset
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

if torch.cuda.is_available():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


@hydra_runner(config_path="conf", config_name="bestow")
def train(cfg):
    OmegaConf.resolve(cfg)
    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend="nccl")
    seed_everything(cfg.data.train_ds.seed)
    torch.set_float32_matmul_precision("medium")

    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    with trainer.init_module():
        model = BESTOW(OmegaConf.to_container(cfg.model, resolve=True))

    # Use BESTOWDataset when language conditioning is configured (num_languages > 0
    # and a language_to_id mapping is provided in the adapter config).
    # Otherwise fall back to SALMDataset — the batch format is identical; BESTOW
    # simply ignores the audio_locator_tag (no placeholder substitution needed).
    language_to_id = OmegaConf.to_container(cfg.model.adapter.get("language_to_id", {}), resolve=True)
    if language_to_id:
        dataset = BESTOWDataset(
            tokenizer=model.tokenizer,
            language_to_id=language_to_id,
            multispeaker_cfg=cfg.data.get("multispeaker_cfg", None),
        )
    else:
        dataset = SALMDataset(tokenizer=model.tokenizer, multispeaker_cfg=cfg.data.get("multispeaker_cfg", None))
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
