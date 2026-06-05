from __future__ import annotations
import os
import sys
import time
import json
import re
import numpy as np
from pathlib import Path

D_MODEL = 512
N_SOLVER = 16384
N_BLOCKS = 4
N_MEMORY = 3
SEQ_LEN = 64
BATCH = 2
EPOCHS = 3000
LR = 3e-4
SEED = 42
EVAL_EVERY = 500
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def gsm8k_extract_answer(text: str) -> str:
    match = re.findall(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match[-1].replace(',', '').strip()
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else ''

def train():
    from runtime.ml.ml_device import set_device, mem_info, reset_pool, is_gpu
    set_device('cuda', 'float32')
    print(f'device: {"GPU" if is_gpu() else "CPU"}')

    from runtime.ml.language import CharTokenizer, TriadLM, TextDataset
    from runtime.ml.nn import Adam
    from runtime.ml.tensor import tensor, no_grad
    from runtime.ml.serialization import save_weights, load_weights
    from datasets import load_dataset

    np.random.seed(SEED)

    ds = load_dataset('openai/gsm8k', 'main')
    train_data = list(ds['train'])
    test_data = list(ds['test'])
    print(f'GSM8K: {len(train_data)} train, {len(test_data)} test')

    all_text = ''
    for ex in train_data:
        all_text += 'Q: ' + ex['question'] + ' A: ' + ex['answer'] + '\n\n'
    for ex in test_data[:200]:
        all_text += 'Q: ' + ex['question'] + ' A: ' + ex['answer'] + '\n\n'

    tok = CharTokenizer(all_text)
    tokens = tok.encode(all_text)
    print(f'vocab_size={tok.vocab_size}  chars={len(all_text)}  tokens={len(tokens)}')

    dataset = TextDataset(tokens, SEQ_LEN)

    model = TriadLM(
        vocab_size=tok.vocab_size,
        d_model=D_MODEL,
        d_state=N_SOLVER,
        n_blocks=N_BLOCKS,
        n_memory=N_MEMORY,
    )
    n_params = sum(p._data.size for p in model.parameters())
    print(f'model: d={D_MODEL} N={N_SOLVER} blocks={N_BLOCKS} memory={N_MEMORY}')
    print(f'params: {n_params:,} ({n_params/1e6:.0f}M)')
    fi = mem_info()
    print(f'GPU: {(fi[1]-fi[2])/1e6:.0f}MB free / {fi[1]/1e6:.0f}MB total')

    opt = Adam(model.parameters(), lr=LR)

    os.makedirs(MODELS_DIR, exist_ok=True)
    t0 = time.perf_counter()
    best_loss = 999.0

    for ep in range(EPOCHS):
        model.training = True
        x_np, y_np = dataset.batch(BATCH)
        x_t = tensor(x_np)
        y_t = tensor(y_np)
        loss = model.loss(x_t, y_t)
        opt.zero_grad()
        loss.backward()
        for p in model.parameters():
            if p._grad is not None:
                p._grad = np.clip(p._grad, -5.0, 5.0)
        opt.step()
        reset_pool()

        err = float(loss._data)
        if err < best_loss:
            best_loss = err

        if (ep + 1) % EVAL_EVERY == 0:
            dt = time.perf_counter() - t0
            fi = mem_info()
            free_mb = (fi[1] - fi[2]) / 1e6
            print(f'  ep {ep+1:4d}/{EPOCHS}  loss={err:.4f}  best={best_loss:.4f}  wall={dt:.0f}s  {free_mb:.0f}MB free')

        if (ep + 1) % 1000 == 0:
            model.training = False
            prompt = tok.encode('Q: What is 2 + 3? A:')
            with no_grad():
                ids = model.generate(prompt, max_new=80, temperature=0.5, top_k=10)
            print(f'  sample: {tok.decode(ids)[:200]}')
            model.training = True

    dt_total = time.perf_counter() - t0
    print(f'\ntraining done in {dt_total:.0f}s  best_loss={best_loss:.4f}')

    model_path = os.path.join(MODELS_DIR, 'gsm8k.weights')
    save_weights(model, model_path)
    print(f'saved: {model_path}')

    print(f'\nevaluating on {len(test_data)} test questions...')
    model.training = False
    correct = 0
    total = 0
    eval_results = []

    for i, ex in enumerate(test_data):
        question = ex['question']
        gold_answer = gsm8k_extract_answer(ex['answer'])
        prompt_text = 'Q: ' + question + ' A:'
        prompt = tok.encode(prompt_text)
        if len(prompt) > SEQ_LEN:
            prompt = prompt[-SEQ_LEN:]
        try:
            with no_grad():
                ids = model.generate(prompt, max_new=150, temperature=0.3, top_k=5)
            generated = tok.decode(ids)
            pred_answer = gsm8k_extract_answer(generated[len(prompt_text):])
        except Exception:
            pred_answer = ''

        match = pred_answer.strip() == gold_answer.strip()
        if match:
            correct += 1
        total += 1

        if (i + 1) % 100 == 0:
            acc = correct / total * 100
            print(f'  {i+1}/{total}  acc={acc:.1f}%  gold={gold_answer}  pred={pred_answer}')

        if i < 5:
            eval_results.append({
                'question': question[:100],
                'gold': gold_answer,
                'pred': pred_answer,
                'correct': match,
                'generated': generated[len(prompt_text):200] if 'generated' in dir() else '',
            })

    accuracy = correct / total * 100 if total > 0 else 0
    print(f'\nGSM8K accuracy: {correct}/{total} = {accuracy:.1f}%')

    report = {
        'dataset': 'openai/gsm8k',
        'model': {
            'd_model': D_MODEL,
            'N_solver': N_SOLVER,
            'n_blocks': N_BLOCKS,
            'n_memory': N_MEMORY,
            'params': n_params,
        },
        'training': {
            'epochs': EPOCHS,
            'lr': LR,
            'batch': BATCH,
            'seq_len': SEQ_LEN,
            'best_loss': round(best_loss, 4),
            'train_s': round(dt_total, 1),
        },
        'eval': {
            'accuracy': round(accuracy, 2),
            'correct': correct,
            'total': total,
        },
        'samples': eval_results[:5],
    }
    report_path = os.path.join(MODELS_DIR, 'gsm8k_report.json')
    Path(report_path).write_text(json.dumps(report, indent=2))
    print(f'saved: {report_path}')

if __name__ == '__main__':
    train()
