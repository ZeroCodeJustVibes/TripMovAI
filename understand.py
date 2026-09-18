"""Phase 3 - describe what is actually in the selected clips.

Two models, loaded one at a time so both fit comfortably in 10 GB of VRAM:
  * BLIP-2 (opt-2.7b, fp16) captions every selected photo and video keyframe
  * Whisper large-v2 transcribes the audio of every selected video

Results are cached on the timeline: an entry that already has a caption (or a
transcript, `""` counting as "checked, no speech") is skipped on re-runs.
"""

from __future__ import annotations

import gc
import logging
import math
from dataclasses import dataclass
from typing import Callable, Sequence

from mediautil import extract_keyframes, has_audio_stream, load_pil, measure_audio
from models import Timeline, TimelineEntry

log = logging.getLogger(__name__)

BLIP2_MODEL = "Salesforce/blip2-opt-2.7b"
WHISPER_MODEL = "large-v2"


@dataclass
class UnderstandConfig:
    blip_model: str = BLIP2_MODEL
    whisper_model: str = WHISPER_MODEL
    device: str = "cuda"
    keyframe_fps: float = 1.0
    max_keyframes: int = 8
    batch_size: int = 2
    max_new_tokens: int = 24
    num_beams: int = 3
    transcribe: bool = True


def _free() -> None:
    """Collect and hand the VRAM back. Callers must drop their references first
    - `del` on a local copy inside this function would free nothing."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _clean_caption(text: str) -> str:
    text = " ".join(str(text).split()).strip(" ,;:-")
    if not text:
        return ""
    # BLIP-2 sometimes repeats itself: "a lake a lake a lake"
    words = text.split()
    deduped: list[str] = []
    for w in words:
        if len(deduped) >= 4 and deduped[-4:] == [w] * 4:
            break
        deduped.append(w)
    text = " ".join(deduped)
    return text[0].upper() + text[1:] if text else ""


# ------------------------------------------------------------- captioning

class Captioner:
    """Thin wrapper around BLIP-2 that also reports a confidence per caption."""

    def __init__(self, cfg: UnderstandConfig):
        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        self.cfg = cfg
        self.torch = torch
        self.device = cfg.device
        # Compute dtype for the inputs. With 8-bit weights `param.dtype` reports
        # int8/float32 depending on the layer, and feeding fp32 pixels into the
        # fp16 compute path makes bitsandbytes warn on every batch.
        self.input_dtype = torch.float16 if cfg.device == "cuda" else torch.float32
        self.model = self._load(Blip2ForConditionalGeneration, torch)
        self.processor = Blip2Processor.from_pretrained(cfg.blip_model)

    def _load(self, model_cls, torch):
        """Load BLIP-2 with whatever fits: fp16 on the GPU if there is room,
        8-bit if not, CPU offload as a last resort before plain CPU.

        blip2-opt-2.7b is ~7.4 GB in fp16, so on a 10 GB card a desktop session
        or a second CUDA process is enough to push a straight fp16 load over."""
        cfg = self.cfg
        if cfg.device != "cuda" or not torch.cuda.is_available():
            log.info("loading %s on CPU (float32) - this will be slow", cfg.blip_model)
            return self._from_pretrained(model_cls, dtype=torch.float32).to("cpu").eval()

        free_bytes = torch.cuda.mem_get_info()[0]
        needed = 8.6e9  # weights plus beam-search activations
        if free_bytes >= needed:
            try:
                log.info("loading %s in fp16 (%.1f GB VRAM free)", cfg.blip_model, free_bytes / 1e9)
                model = self._from_pretrained(model_cls, dtype=torch.float16, device_map={"": 0})
                return model.eval()
            except torch.OutOfMemoryError as exc:
                log.warning("fp16 load hit OOM (%s) - retrying quantised", exc)
                _free()

        try:
            from transformers import BitsAndBytesConfig

            log.info("loading %s in 8-bit (%.1f GB VRAM free)", cfg.blip_model, free_bytes / 1e9)
            # BLIP-2 keeps some layers unquantised, so bitsandbytes logs a
            # "MatMul8bitLt: inputs will be cast" line on every batch. It casts
            # them correctly - the notice is just noise in the CLI output.
            logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)
            model = self._from_pretrained(
                model_cls,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map={"": 0},
            )
            return model.eval()
        except Exception as exc:  # noqa: BLE001 - bitsandbytes missing or OOM
            log.warning("8-bit load failed (%s) - falling back to CPU offload", exc)
            _free()

        log.info("loading %s with CPU offload", cfg.blip_model)
        model = self._from_pretrained(
            model_cls, dtype=torch.float16, device_map="auto",
            max_memory={0: f"{max(1, int(free_bytes / 1e9) - 1)}GiB", "cpu": "24GiB"},
        )
        return model.eval()

    def _from_pretrained(self, model_cls, dtype=None, **kwargs):
        if dtype is not None:
            try:
                return model_cls.from_pretrained(self.cfg.blip_model, dtype=dtype, **kwargs)
            except TypeError:  # transformers < 5 spells it torch_dtype
                return model_cls.from_pretrained(self.cfg.blip_model, torch_dtype=dtype, **kwargs)
        return model_cls.from_pretrained(self.cfg.blip_model, **kwargs)

    def caption_batch(self, images: Sequence) -> list[tuple[str, float]]:
        """-> [(caption, confidence 0..1)] for a batch of PIL images."""
        if not images:
            return []
        torch = self.torch
        device = getattr(self.model, "device", None) or next(self.model.parameters()).device
        inputs = self.processor(images=list(images), return_tensors="pt").to(device, self.input_dtype)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                num_beams=self.cfg.num_beams,
                length_penalty=1.0,
                return_dict_in_generate=True,
                output_scores=True,
            )
        texts = self.processor.batch_decode(out.sequences, skip_special_tokens=True)
        scores = getattr(out, "sequences_scores", None)
        results: list[tuple[str, float]] = []
        for i, text in enumerate(texts):
            if scores is not None:
                conf = float(min(1.0, math.exp(float(scores[i]))))
            else:  # greedy decoding has no sequences_scores
                conf = 0.5
            results.append((_clean_caption(text), round(conf, 4)))
        return results

    def close(self) -> None:
        self.model = None
        self.processor = None
        _free()


def _caption_with_fallback(captioner: "Captioner", images: list) -> list[tuple[str, float]]:
    """Caption a batch, retrying one image at a time if the batch runs out of VRAM."""
    import torch

    try:
        return captioner.caption_batch(images)
    except torch.OutOfMemoryError:
        if len(images) == 1:
            raise
        log.warning("batch of %d ran out of VRAM - falling back to one image at a time", len(images))
        _free()
        results: list[tuple[str, float]] = []
        for image in images:
            try:
                results.extend(captioner.caption_batch([image]))
            except Exception as exc:  # noqa: BLE001
                log.error("captioning a single image failed: %s", exc)
                results.append(("", 0.0))
        return results


def caption_entries(entries: list[TimelineEntry], cfg: UnderstandConfig,
                    progress_cb: Callable | None = None,
                    checkpoint: Callable | None = None) -> int:
    """Caption photos directly and videos via their best keyframe."""
    todo = [e for e in entries if not e.caption]
    if not todo:
        log.info("all selected clips already captioned - skipping BLIP-2")
        return 0

    try:
        captioner = Captioner(cfg)
    except Exception as exc:  # noqa: BLE001 - OOM, no CUDA, download failure
        log.error("could not load BLIP-2 (%s) - continuing without captions", exc)
        return 0

    done = 0
    try:
        photos = [e for e in todo if not e.is_video]
        videos = [e for e in todo if e.is_video]

        for start in range(0, len(photos), cfg.batch_size):
            chunk = photos[start : start + cfg.batch_size]
            images, keep = [], []
            for e in chunk:
                img = load_pil(e.filepath, max_side=512)
                if img is None:
                    e.caption = ""
                    continue
                images.append(img)
                keep.append(e)
            try:
                for e, (text, conf) in zip(keep, _caption_with_fallback(captioner, images)):
                    e.caption, e.caption_confidence = text, conf
                    done += 1
            except Exception as exc:  # noqa: BLE001
                log.error("captioning failed for %d photos: %s", len(keep), exc)
            if checkpoint:
                checkpoint()  # keep finished captions if a later batch dies
            if progress_cb:
                progress_cb(done, len(todo))

        for e in videos:
            try:
                frames = extract_keyframes(e.filepath, cfg.keyframe_fps, cfg.max_keyframes)
                best_text, best_conf = "", -1.0
                for i in range(0, len(frames), cfg.batch_size):
                    for text, conf in _caption_with_fallback(captioner, frames[i : i + cfg.batch_size]):
                        if text and conf > best_conf:
                            best_text, best_conf = text, conf
                e.caption = best_text
                e.caption_confidence = round(max(best_conf, 0.0), 4)
                done += 1
            except Exception as exc:  # noqa: BLE001
                log.error("captioning failed for %s: %s", e.name, exc)
            if checkpoint:
                checkpoint()
            if progress_cb:
                progress_cb(done, len(todo))
    finally:
        captioner.close()
    log.info("captioned %d clips", done)
    return done


# ---------------------------------------------------------- transcription

def _halve_weights(model):
    """Cast a Whisper model to fp16 *except* its LayerNorms.

    Whisper's LayerNorm runs `super().forward(x.float())`, so it needs fp32
    norm weights, while its Linear/Conv layers cast their weights to the input
    dtype themselves. A blanket `.half()` therefore raises "expected scalar
    type Float but found Half"; skipping the norms halves 6.2 GB to 3.1 GB.
    """
    import torch

    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            continue
        for param in module.parameters(recurse=False):
            if param.is_floating_point():
                param.data = param.data.half()
        for buffer in module.buffers(recurse=False):
            if buffer.is_floating_point():
                buffer.data = buffer.data.half()
    return model


def _load_whisper(cfg: UnderstandConfig):
    """Whisper on the GPU in fp16, degrading to a smaller model then to CPU.

    `whisper.load_model(..., device="cuda")` keeps fp32 weights - 6.2 GB for
    large-v2, which will not sit next to anything else on a 10 GB card. Loading
    on CPU and halving first brings it down to 3.1 GB.
    """
    import torch
    import whisper

    if cfg.device != "cuda" or not torch.cuda.is_available():
        log.info("loading Whisper %s on CPU", cfg.whisper_model)
        return whisper.load_model(cfg.whisper_model, device="cpu")

    ladder = [cfg.whisper_model] + [m for m in ("medium", "small", "base") if m != cfg.whisper_model]
    for name in ladder:
        try:
            log.info("loading Whisper %s on cuda (fp16, %.1f GB free)",
                     name, torch.cuda.mem_get_info()[0] / 1e9)
            model = _halve_weights(whisper.load_model(name, device="cpu")).to("cuda")
            if name != cfg.whisper_model:
                log.warning("using Whisper %s instead of %s - not enough VRAM", name, cfg.whisper_model)
            return model
        except torch.OutOfMemoryError:
            log.warning("Whisper %s did not fit in VRAM", name)
            _free()
    log.warning("no Whisper model fits in VRAM - transcribing on CPU (slow)")
    return whisper.load_model(cfg.whisper_model, device="cpu")


def transcribe_entries(entries: list[TimelineEntry], cfg: UnderstandConfig,
                       progress_cb: Callable | None = None,
                       checkpoint: Callable | None = None) -> int:
    """Whisper transcripts for selected videos. `""` means "checked, silent"."""
    todo = [e for e in entries if e.is_video and e.transcript is None]
    if not todo:
        return 0

    try:
        model = _load_whisper(cfg)
    except Exception as exc:  # noqa: BLE001
        log.error("could not load Whisper (%s) - continuing without transcripts", exc)
        return 0

    done = 0
    try:
        for e in todo:
            try:
                if not has_audio_stream(e.filepath):
                    e.transcript = ""
                    e.extra.pop("audio_mean_db", None)
                else:
                    levels = measure_audio(e.filepath)   # cached for Phase 6's mixing
                    if levels:
                        e.extra["audio_mean_db"] = round(levels["mean_db"], 2)
                        e.extra["audio_max_db"] = round(levels["max_db"], 2)
                    result = model.transcribe(
                        e.filepath,
                        fp16=(str(next(model.parameters()).device).startswith("cuda")),
                        condition_on_previous_text=False,
                        verbose=None,
                    )
                    # drop segments Whisper itself thinks are silence/hallucination
                    kept = [
                        seg.get("text", "")
                        for seg in result.get("segments", [])
                        if seg.get("no_speech_prob", 0.0) < 0.6 and seg.get("avg_logprob", -5.0) > -1.0
                    ]
                    e.transcript = " ".join(t.strip() for t in kept).strip()
                    if e.transcript:
                        e.extra["speech_segments"] = [
                            {"start": round(float(s.get("start", 0.0)), 2),
                             "end": round(float(s.get("end", 0.0)), 2)}
                            for s in result.get("segments", [])
                            if s.get("no_speech_prob", 0.0) < 0.6 and s.get("avg_logprob", -5.0) > -1.0
                        ]
                done += 1
            except Exception as exc:  # noqa: BLE001
                log.error("transcription failed for %s: %s", e.name, exc)
                e.transcript = ""
            if checkpoint:
                checkpoint()
            if progress_cb:
                progress_cb(done, len(todo))
    finally:
        model = None
        _free()
    speaking = sum(1 for e in todo if e.transcript)
    log.info("transcribed %d videos (%d contain speech)", done, speaking)
    return done


# -------------------------------------------------------------- pipeline

def run(timeline: Timeline, cfg: UnderstandConfig,
        progress: dict[str, Callable] | None = None,
        checkpoint: Callable | None = None) -> Timeline:
    """Caption then transcribe. `checkpoint` is called as results land, so a
    crash in the Whisper stage cannot throw away an hour of captioning."""
    progress = progress or {}
    entries = timeline.selected
    if not entries:
        log.warning("no selected clips to understand")
        return timeline
    caption_entries(entries, cfg, progress.get("caption"), checkpoint)
    if cfg.transcribe:
        transcribe_entries(entries, cfg, progress.get("whisper"), checkpoint)
    return timeline


def stats(timeline: Timeline) -> dict[str, object]:
    sel = timeline.selected
    return {
        "captioned": sum(1 for e in sel if e.caption),
        "mean_confidence": round(
            sum(e.caption_confidence or 0.0 for e in sel) / max(1, sum(1 for e in sel if e.caption)), 3),
        "videos_with_speech": sum(1 for e in sel if e.is_video and e.transcript),
    }
