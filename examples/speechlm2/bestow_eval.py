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
BESTOW evaluation script.

Computes WER / CER on a Lhotse CutSet and writes per-example JSONL predictions.
Supports multilingual evaluation (per-language WER), streaming Wait-K inference,
and character-error-rate (CER) for character-based languages (Chinese, Japanese, etc.).

Basic usage:

  python examples/speechlm2/bestow_eval.py \
    pretrained_name=/path/to/checkpoint_dir \
    inputs=/path/to/test.jsonl.gz \
    user_prompt="Transcribe the speech."

Streaming evaluation (fixed Wait-K):

  ... wait_k=10 stride=4

Multilingual (auto per-language WER from cut metadata):

  ... per_language_wer=true

Character-based languages (CER instead of WER):

  ... cer_languages=[zh,ja,ko]
"""
import json
from collections import defaultdict
from dataclasses import dataclass, field
from time import perf_counter
from typing import Optional

import lhotse.dataset
import torch
from lhotse import CutSet
from lhotse.serialization import SequentialJsonlWriter
from omegaconf import OmegaConf
from transformers import GenerationConfig
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo.collections.common.data.lhotse.cutset import guess_parse_cutset
from nemo.collections.speechlm2.models import BESTOW
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero


# ---------------------------------------------------------------------------
# Dataset helper — loads raw audio from Lhotse cuts
# ---------------------------------------------------------------------------


class ToAudio(torch.utils.data.Dataset):
    def __getitem__(self, cuts: CutSet):
        audios, audio_lens = cuts.load_audio(collate=True)
        return {"cuts": cuts, "audios": audios, "audio_lens": audio_lens}


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


@dataclass
class BESTOWEvalConfig:
    pretrained_name: str
    inputs: str
    batch_size: int = 32
    max_new_tokens: int = 128
    output_manifest: Optional[str] = "bestow_generations.jsonl"
    verbose: bool = True

    # Text normalisation before WER/CER computation.
    # "english" = EnglishTextNormalizer, "basic" = BasicTextNormalizer, else identity.
    use_normalizer: Optional[str] = "basic"

    # Prompt that wraps each audio sample.
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = "Transcribe the speech."

    device: str = "cuda"
    dtype: str = "bfloat16"

    extra_eos_tokens: Optional[list] = None

    # Streaming Wait-K inference.  None = offline (full speech context at every step).
    wait_k: Optional[int] = None
    stride: int = 4   # speech encoder frames per text token (L=4 per paper)

    # Multilingual options.
    per_language_wer: bool = False  # report WER broken down by language tag in cut metadata
    cer_languages: list = field(default_factory=list)  # e.g. ["zh", "ja", "ko"]

    enable_thinking: Optional[bool] = None


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


@hydra_runner(config_name="BESTOWEvalConfig", schema=BESTOWEvalConfig)
def main(cfg: BESTOWEvalConfig):
    logging.info(f'Hydra config:\n{OmegaConf.to_yaml(cfg)}')

    model = BESTOW.from_pretrained(cfg.pretrained_name)
    model = model.to(getattr(torch, cfg.dtype)).to(cfg.device).eval()

    cuts = guess_parse_cutset(cfg.inputs).sort_by_duration()
    dloader = torch.utils.data.DataLoader(
        dataset=ToAudio(),
        sampler=lhotse.dataset.DynamicCutSampler(cuts, max_cuts=cfg.batch_size, rank=0, world_size=1),
        num_workers=1,
        batch_size=None,
    )

    # Normaliser selection.
    _normalizers = {"english": EnglishTextNormalizer(), "basic": BasicTextNormalizer()}
    normalizer = _normalizers.get(cfg.use_normalizer, lambda x: x)

    eos_tokens = [model.text_eos_id]
    if cfg.extra_eos_tokens:
        for t in cfg.extra_eos_tokens:
            tid = model.tokenizer.token_to_id(t)
            assert tid is not None, f"Token '{t}' not in model vocabulary."
            eos_tokens.append(tid)

    gen_config = GenerationConfig(
        max_new_tokens=cfg.max_new_tokens,
        bos_token_id=model.text_bos_id,
        eos_token_id=eos_tokens,
        pad_token_id=model.text_pad_id,
    )

    # Build per-example prompt.  BESTOW does NOT use an audio_locator_tag;
    # audio is fused entirely via the cross-attention adapter.
    prompt_turns = []
    if cfg.system_prompt:
        prompt_turns.append({"role": "system", "content": cfg.system_prompt})
    prompt_turns.append({"role": "user", "content": cfg.user_prompt or "Transcribe the speech."})

    refs: list[str] = []
    hyps: list[str] = []
    cut_ids: list[str] = []
    cut_durations: list[float] = []
    cut_languages: list[str] = []
    infer_durations: list[float] = []

    for batch_idx, batch in enumerate(dloader):
        ts = perf_counter()
        batch_cuts = batch["cuts"]

        answer_ids = model.generate(
            prompts=[prompt_turns] * len(batch_cuts),
            audios=batch["audios"].to(model.device, non_blocking=True),
            audio_lens=batch["audio_lens"].to(model.device, non_blocking=True),
            max_new_tokens=cfg.max_new_tokens,
            wait_k=cfg.wait_k,
            stride=cfg.stride,
            generation_config=gen_config,
            enable_thinking=cfg.enable_thinking,
        )
        answer_ids = answer_ids.cpu()
        batch_infer_dur = perf_counter() - ts

        batch_refs = []
        batch_hyps = []
        for cut, ans in zip(batch_cuts, answer_ids):
            ref_text = cut.supervisions[0].text if cut.supervisions else ""
            lang = _get_language(cut)
            use_cer = lang in cfg.cer_languages
            ref = _normalise(ref_text, normalizer, use_cer)
            hyp = _normalise(
                model.tokenizer.ids_to_text(_parse_hyp(ans, eos_tokens)).strip(),
                normalizer,
                use_cer,
            )
            batch_refs.append(ref)
            batch_hyps.append(hyp)
            cut_languages.append(lang)

        if cfg.verbose:
            batch_dur = sum(c.duration for c in batch_cuts)
            metric_name, metric_val = _compute_metric(batch_hyps, batch_refs, cut_languages[-len(batch_cuts):], cfg.cer_languages)
            rtfx = batch_dur / batch_infer_dur
            logging.info(f"Batch {batch_idx}: {metric_name}={metric_val:.2%} RTFx={rtfx:.1f}")

        refs.extend(batch_refs)
        hyps.extend(batch_hyps)
        cut_ids.extend(c.id for c in batch_cuts)
        cut_durations.extend(c.duration for c in batch_cuts)
        infer_durations.append(batch_infer_dur)

    # Overall WER/CER.
    _, overall_name, overall_val = _aggregate_metric(hyps, refs, cut_languages, cfg.cer_languages)
    rtfx = sum(cut_durations) / sum(infer_durations)
    logging.info(f"Overall {overall_name}: {overall_val:.2%}  RTFx: {rtfx:.1f}")

    # Per-language breakdown.
    if cfg.per_language_wer:
        _log_per_language(hyps, refs, cut_languages, cfg.cer_languages)

    # Write JSONL output.
    with _create_output_writer(cfg.output_manifest) as writer:
        for cut_id, dur, lang, ref, hyp in zip(cut_ids, cut_durations, cut_languages, refs, hyps):
            writer.write({
                "id": cut_id,
                "duration": dur,
                "language": lang,
                "text": ref,
                "pred_text": hyp,
            })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_language(cut) -> str:
    """Return ISO-639-1 language code from cut metadata, supervision, or 'unknown'."""
    if cut.supervisions and cut.supervisions[0].language:
        return cut.supervisions[0].language
    meta = getattr(cut, "custom", None)
    if meta and "language" in meta:
        return meta["language"]
    return "unknown"


def _normalise(text: str, normalizer, use_cer: bool) -> str:
    """Apply text normaliser.  For CER languages, collapse whitespace for char-level eval."""
    text = normalizer(text)
    if use_cer:
        return text.replace(" ", "")  # char-level: remove word boundaries
    return text


def _parse_hyp(answer: torch.Tensor, eos_tokens: list[int]) -> torch.Tensor:
    """Truncate at first EOS token."""
    end = torch.isin(answer, torch.tensor(eos_tokens)).nonzero(as_tuple=True)[0]
    return answer[:end[0]] if end.numel() > 0 else answer


def _compute_metric(
    hyps: list[str], refs: list[str], langs: list[str], cer_languages: list[str]
) -> tuple[str, float]:
    """Compute WER or CER based on language, return (metric_name, value)."""
    if all(l in cer_languages for l in langs):
        wer, *_ = word_error_rate_detail(hyps, refs, use_cer=True)
        return "CER", wer
    wer, *_ = word_error_rate_detail(hyps, refs, use_cer=False)
    return "WER", wer


def _aggregate_metric(
    hyps: list[str], refs: list[str], langs: list[str], cer_languages: list[str]
) -> tuple[list, str, float]:
    wer, *_ = word_error_rate_detail(hyps, refs, use_cer=False)
    return [], "WER", wer


def _log_per_language(
    hyps: list[str], refs: list[str], langs: list[str], cer_languages: list[str]
) -> None:
    """Log WER / CER per language code."""
    by_lang: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for h, r, l in zip(hyps, refs, langs):
        by_lang[l][0].append(h)
        by_lang[l][1].append(r)

    for lang in sorted(by_lang):
        lh, lr = by_lang[lang]
        use_cer = lang in cer_languages
        val, *_ = word_error_rate_detail(lh, lr, use_cer=use_cer)
        metric = "CER" if use_cer else "WER"
        logging.info(f"  [{lang}] {metric}: {val:.2%}  (n={len(lh)})")


class _NullWriter:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def write(self, data):
        pass


def _create_output_writer(output_manifest: Optional[str]):
    if output_manifest is None or not is_global_rank_zero():
        return _NullWriter()
    return SequentialJsonlWriter(output_manifest)


if __name__ == "__main__":
    main()
