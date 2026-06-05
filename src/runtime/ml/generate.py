from __future__ import annotations
import numpy as np
from typing import Optional, Callable, Generator
from dataclasses import dataclass, field
import time

THINK_START = 151667
THINK_END = 151668
IM_END = 151645
PAD = 151643

@dataclass
class GenerationResult:
    text: str = ""
    thinking: str = ""
    all_tokens: list[int] = field(default_factory=list)
    think_tokens: list[int] = field(default_factory=list)
    answer_tokens: list[int] = field(default_factory=list)
    tokens_per_second: float = 0.0
    prompt_tokens: int = 0
    total_tokens: int = 0

@dataclass
class StreamToken:
    token_id: int
    text: str
    phase: str

def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.where(np.isneginf(x) | ~np.isfinite(x), -1e30, x)
    mx = np.max(x)
    ex = np.exp(x - mx)
    s = np.sum(ex)
    if s == 0:
        s = 1.0
    return ex / s

def _apply_repetition_penalty(logits: np.ndarray, token_ids: list[int],
                               penalty: float, window: int = 64) -> np.ndarray:
    if penalty == 1.0 or not token_ids:
        return logits
    recent = token_ids[-window:]
    seen = set()
    for tid in recent:
        if tid not in seen:
            seen.add(tid)
            if logits[tid] > 0:
                logits[tid] = logits[tid] / penalty
            else:
                logits[tid] = logits[tid] * penalty
    return logits

def sample_token(logits: np.ndarray, temperature: float = 1.0,
                 top_k: int = 0, top_p: float = 1.0,
                 repetition_penalty: float = 1.0,
                 repetition_window: int = 64,
                 recent_tokens: Optional[list[int]] = None,
                 rng: Optional[np.random.Generator] = None) -> int:
    if rng is None:
        rng = np.random.default_rng()
    logits = logits.astype(np.float64)

    if recent_tokens is not None and repetition_penalty != 1.0:
        logits = _apply_repetition_penalty(logits, recent_tokens, repetition_penalty, repetition_window)

    if temperature <= 0:
        return int(np.argmax(logits))
    logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, len(logits))
        threshold = np.sort(logits)[-top_k]
        logits[logits < threshold] = -np.inf
    if top_p < 1.0:
        sorted_idx = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_idx]
        probs = _softmax(sorted_logits)
        cum_probs = np.cumsum(probs)
        cutoff = sorted_idx[cum_probs > top_p]
        logits[cutoff] = -np.inf
    probs = _softmax(logits)
    if not np.isfinite(probs).all():
        return int(np.argmax(logits))
    probs = np.nan_to_num(probs, nan=0.0)
    total = probs.sum()
    if total <= 0:
        return int(np.argmax(logits))
    probs = probs / total
    return int(rng.choice(len(probs), p=probs))

def _run_generate(model, prompt_ids: list[int], max_new_tokens: int,
                  temperature: float, top_k: int, top_p: float,
                  repetition_penalty: float, repetition_window: int,
                  stop_ids: set[int], seed: Optional[int],
                  callback: Optional[Callable]) -> GenerationResult:
    from runtime.ml.forward import forward

    rng = np.random.default_rng(seed)
    token_ids = list(prompt_ids)
    kv_caches = []
    all_tokens = []
    think_tokens = []
    answer_tokens = []
    in_thinking = False
    thinking_done = False
    t_start = time.time()

    logits = forward(model, token_ids, kv_caches)
    next_id = sample_token(logits[-1], temperature, top_k, top_p,
                            repetition_penalty, repetition_window,
                            token_ids, rng)
    token_ids.append(next_id)
    all_tokens.append(next_id)

    if next_id == THINK_START:
        in_thinking = True

    phase = "thinking" if in_thinking else "answer"
    if callback:
        callback(0, next_id, phase)

    for step in range(1, max_new_tokens):
        logits = forward(model, [next_id], kv_caches)
        next_id = sample_token(logits[-1], temperature, top_k, top_p,
                                repetition_penalty, repetition_window,
                                token_ids, rng)
        token_ids.append(next_id)
        all_tokens.append(next_id)

        if in_thinking:
            think_tokens.append(next_id)
            if next_id == THINK_END:
                in_thinking = False
                thinking_done = True
        else:
            if next_id == THINK_START:
                in_thinking = True
            elif next_id in stop_ids:
                phase = "eos"
                if callback:
                    callback(step, next_id, phase)
                break
            else:
                answer_tokens.append(next_id)

        if in_thinking:
            phase = "thinking"
        elif next_id in stop_ids:
            phase = "eos"
        else:
            phase = "answer"

        if callback:
            callback(step, next_id, phase)

    elapsed = time.time() - t_start
    tps = len(all_tokens) / elapsed if elapsed > 0 else 0.0
    thinking_text = model.tokenizer.decode(think_tokens, skip_special=True) if think_tokens else ""
    answer_text = model.tokenizer.decode(answer_tokens, skip_special=True)

    return GenerationResult(
        text=answer_text.strip(),
        thinking=thinking_text.strip(),
        all_tokens=all_tokens,
        think_tokens=think_tokens,
        answer_tokens=answer_tokens,
        tokens_per_second=tps,
        prompt_tokens=len(prompt_ids),
        total_tokens=len(prompt_ids) + len(all_tokens),
    )

def _run_stream(model, prompt_ids: list[int], max_new_tokens: int,
                temperature: float, top_k: int, top_p: float,
                repetition_penalty: float, repetition_window: int,
                stop_ids: set[int], seed: Optional[int]
                ) -> Generator[StreamToken, None, GenerationResult]:
    from runtime.ml.forward import forward

    rng = np.random.default_rng(seed)
    token_ids = list(prompt_ids)
    kv_caches = []
    all_tokens = []
    think_tokens = []
    answer_tokens = []
    in_thinking = False
    thinking_done = False
    t_start = time.time()

    logits = forward(model, token_ids, kv_caches)
    next_id = sample_token(logits[-1], temperature, top_k, top_p,
                            repetition_penalty, repetition_window,
                            token_ids, rng)
    token_ids.append(next_id)
    all_tokens.append(next_id)

    if next_id == THINK_START:
        in_thinking = True

    phase = "thinking" if in_thinking else "answer"
    yield StreamToken(next_id, model.tokenizer.decode([next_id], skip_special=True), phase)

    for step in range(1, max_new_tokens):
        logits = forward(model, [next_id], kv_caches)
        next_id = sample_token(logits[-1], temperature, top_k, top_p,
                                repetition_penalty, repetition_window,
                                token_ids, rng)
        token_ids.append(next_id)
        all_tokens.append(next_id)

        if in_thinking:
            think_tokens.append(next_id)
            if next_id == THINK_END:
                in_thinking = False
                thinking_done = True
        else:
            if next_id == THINK_START:
                in_thinking = True
            elif next_id in stop_ids:
                yield StreamToken(next_id, model.tokenizer.decode([next_id], skip_special=True), "eos")
                break
            else:
                answer_tokens.append(next_id)

        if in_thinking:
            phase = "thinking"
        elif next_id in stop_ids:
            phase = "eos"
        else:
            phase = "answer"

        yield StreamToken(next_id, model.tokenizer.decode([next_id], skip_special=True), phase)

    elapsed = time.time() - t_start
    tps = len(all_tokens) / elapsed if elapsed > 0 else 0.0
    thinking_text = model.tokenizer.decode(think_tokens, skip_special=True) if think_tokens else ""
    answer_text = model.tokenizer.decode(answer_tokens, skip_special=True)

    return GenerationResult(
        text=answer_text.strip(),
        thinking=thinking_text.strip(),
        all_tokens=all_tokens,
        think_tokens=think_tokens,
        answer_tokens=answer_tokens,
        tokens_per_second=tps,
        prompt_tokens=len(prompt_ids),
        total_tokens=len(prompt_ids) + len(all_tokens),
    )

def generate(model, prompt_ids: list[int], max_new_tokens: int = 512,
             temperature: float = 0.6, top_k: int = 50, top_p: float = 0.95,
             repetition_penalty: float = 1.0, repetition_window: int = 64,
             stop_ids: Optional[list[int]] = None,
             callback: Optional[Callable[[int, int, str], None]] = None,
             seed: Optional[int] = None) -> GenerationResult:
    eos_ids = set(stop_ids or [])
    eos_ids.update([IM_END, PAD])
    return _run_generate(model, prompt_ids, max_new_tokens, temperature, top_k, top_p,
                         repetition_penalty, repetition_window, eos_ids, seed, callback)

def stream_generate(model, prompt_ids: list[int], max_new_tokens: int = 512,
                    temperature: float = 0.6, top_k: int = 50, top_p: float = 0.95,
                    repetition_penalty: float = 1.0, repetition_window: int = 64,
                    stop_ids: Optional[list[int]] = None,
                    seed: Optional[int] = None) -> Generator[StreamToken, None, GenerationResult]:
    eos_ids = set(stop_ids or [])
    eos_ids.update([IM_END, PAD])
    return _run_stream(model, prompt_ids, max_new_tokens, temperature, top_k, top_p,
                       repetition_penalty, repetition_window, eos_ids, seed)

def chat(model, messages: list[dict[str, str]], max_new_tokens: int = 1024,
         temperature: float = 0.6, top_k: int = 50, top_p: float = 0.95,
         repetition_penalty: float = 1.0,
         seed: Optional[int] = None,
         show_thinking: bool = False,
         stream: bool = False) -> GenerationResult:
    ids = model.tokenizer.apply_chat_template(messages, add_generation_prompt=True, think=True)
    eos_id = model.tokenizer.eos_id
    stop_ids = [eos_id] if eos_id is not None else []

    if stream:
        def cb(step, tid, phase):
            tok = model.tokenizer.decode([tid], skip_special=True)
            if phase == "thinking" and show_thinking:
                print(tok, end="", flush=True)
            elif phase == "answer":
                print(tok, end="", flush=True)
            elif phase == "eos":
                print()
    else:
        cb = None

    return generate(model, ids, max_new_tokens, temperature, top_k, top_p,
                    repetition_penalty, 64, stop_ids, seed=seed, callback=cb)
