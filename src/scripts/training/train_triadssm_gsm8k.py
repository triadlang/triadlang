from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from triad import ntri as np

D_MODEL = 333
D_STATE = 333
N_BLOCKS = 33
N_MEMORY = 3
COUPLING = True
SEQ_LEN = 256
BATCH = 3
EPOCHS = 2000
LR = 3e-4
WARMUP_EPOCHS = 200
EVAL_EVERY = 200
GEN_EVERY = 400
GEN_TOKENS = 120
TEMPERATURE = 0.7
TOP_K = 20
GRAD_CLIP = 5.0
SEED = 42

DATASET_DIR = '/home/qrv0/workspace/triadlang/archives/dataset'
MODELS_DIR = '/home/qrv0/workspace/triadlang/archives/models'

SRC = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.abspath(SRC))

from runtime.ml.language import CharTokenizer, TextDataset
from runtime.ml.ml_device import asnumpy, is_gpu, mem_info, reset_pool, set_device
from runtime.ml.nn import Adam, Embedding, LayerNorm, Module, TriadSSMBlock, triad
from runtime.ml.serialization import load_weights, save_weights
from runtime.ml.tensor import TriadTensor, no_grad, tensor


class TriadSSMLM(Module):

    def __init__(self, vocab_size: int, d_model: int = 256, d_state: int = 64,
                 n_blocks: int = 4, n_memory: int = 3, coupling: bool = True):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.tok_emb = Embedding(vocab_size, d_model)
        self.blocks = [
            TriadSSMBlock(d_model, d_state=d_state, n_memory=n_memory,
                          coupling=coupling, noise=0.0)
            for _ in range(n_blocks)
        ]
        self.ln_f = LayerNorm(d_model)
        self.lm_head = triad(d_model, vocab_size, bias=False)
        self.training = True

    def forward(self, idx: TriadTensor) -> TriadTensor:
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def loss(self, idx: TriadTensor, targets: TriadTensor) -> TriadTensor:
        logits = self.forward(idx)
        B, T, C = logits.shape
        logits_2d = logits.reshape(B * T, C)
        targets_np = targets._data.astype(int).reshape(B * T)
        shifted = logits_2d._data - logits_2d._data.max(axis=1, keepdims=True)
        exp_s = np.exp(shifted)
        probs = exp_s / exp_s.sum(axis=1, keepdims=True)
        log_probs = np.log(probs + 1e-12)
        nll = -log_probs[np.arange(B * T), targets_np]
        loss_val = np.mean(nll)
        out = TriadTensor(loss_val)
        out._requires_grad = True
        out._children = [logits_2d]

        def _back(g):
            grad = probs.copy()
            grad[np.arange(B * T), targets_np] -= 1.0
            grad /= (B * T)
            grad *= g
            logits_2d._grad = grad if logits_2d._grad is None else logits_2d._grad + grad
        out._grad_fn = _back
        return out

    def generate(self, idx: list[int], max_new: int = 50,
                 temperature: float = 1.0, top_k: int = 0) -> list[int]:
        was_training = self.training
        self.training = False
        for _ in range(max_new):
            ctx = idx[-SEQ_LEN:]
            x = tensor(np.array([ctx], dtype=np.int64))
            with no_grad():
                logits = self.forward(x)
            logits_np = logits._data[0, -1] / max(temperature, 1e-8)
            if top_k > 0:
                top_vals = np.sort(logits_np)[-top_k:]
                logits_np = np.where(logits_np >= top_vals[0], logits_np, -1e9)
            pr = np.exp(logits_np - logits_np.max())
            pr = pr / pr.sum()

            pr_np = asnumpy(pr) if hasattr(pr, 'get') else np.asarray(pr)
            next_id = int(np.random.choice(len(pr_np), p=pr_np))
            idx.append(next_id)
        self.training = was_training
        return idx

def gsm8k_extract_answer(text: str) -> str:
    match = re.findall(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match[-1].replace(',', '').strip()
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else ''

def load_local_gsm8k(dataset_dir: str):
    import fastparquet
    main_train = os.path.join(dataset_dir, 'main', 'train-00000-of-00001.parquet')
    main_test = os.path.join(dataset_dir, 'main', 'test-00000-of-00001.parquet')
    socratic_train = os.path.join(dataset_dir, 'socratic', 'train-00000-of-00001.parquet')

    def _load(path):
        if not os.path.exists(path):
            return []
        pf = fastparquet.ParquetFile(path)
        df = pf.to_pandas()
        return [{'question': str(r['question']), 'answer': str(r['answer'])}
                for _, r in df.iterrows()]

    train = _load(main_train)
    test = _load(main_test)
    socratic = _load(socratic_train)

    train.extend(socratic)
    return train, test

def train():
    np.random.seed(SEED)

    set_device('cuda', 'float32')
    print(f'  device: {"GPU" if is_gpu() else "CPU"}')
    mi = mem_info()
    if is_gpu():
        print(f'  GPU mem: {(mi[1]-mi[2])/1e6:.0f}MB free / {mi[1]/1e6:.0f}MB total')

    os.makedirs(MODELS_DIR, exist_ok=True)

    header('loading local GSM8K dataset')
    train_data, test_data = load_local_gsm8k(DATASET_DIR)
    print(f'  train={len(train_data)}  test={len(test_data)}')

    all_text = ''
    for ex in train_data:
        all_text += 'Q: ' + ex['question'] + ' A: ' + ex['answer'] + '\n\n'

    tok = CharTokenizer(all_text)
    tokens = tok.encode(all_text)
    print(f'  vocab_size={tok.vocab_size}  chars={len(all_text)}  tokens={len(tokens)}')

    dataset = TextDataset(tokens, SEQ_LEN)
    print(f'  dataset samples={len(dataset)}')

    model = TriadSSMLM(
        vocab_size=tok.vocab_size,
        d_model=D_MODEL,
        d_state=D_STATE,
        n_blocks=N_BLOCKS,
        n_memory=N_MEMORY,
        coupling=COUPLING,
    )
    n_params = sum(p._data.size for p in model.parameters())
    print(f'\n  model: d_model={D_MODEL} d_state={D_STATE} n_blocks={N_BLOCKS} n_memory={N_MEMORY}')
    print(f'  params: {n_params:,}')

    opt = Adam(model.parameters(), lr=LR)

    header('training loop')
    t0 = time.perf_counter()
    best_loss = float('inf')
    history = []

    for ep in range(EPOCHS):

        if ep < WARMUP_EPOCHS:
            opt.lr = LR * (ep + 1) / WARMUP_EPOCHS
        else:
            frac = (ep - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
            opt.lr = LR * (0.5 + 0.5 * np.cos(np.pi * frac))

        model.training = True
        x_np, y_np = dataset.batch(BATCH)
        x_t = tensor(x_np)
        y_t = tensor(y_np)
        loss = model.loss(x_t, y_t)
        opt.zero_grad()
        loss.backward()
        for p in model.parameters():
            if p._grad is not None:
                p._grad = np.clip(p._grad, -GRAD_CLIP, GRAD_CLIP)
        opt.step()
        reset_pool()

        err = float(loss._data)
        history.append(err)
        if err < best_loss:
            best_loss = err

        if (ep + 1) % EVAL_EVERY == 0:
            dt = time.perf_counter() - t0
            mi = mem_info()
            free_mb = (mi[1] - mi[2]) / 1e6
            print(f'  ep {ep+1:4d}/{EPOCHS}  loss={err:.4f}  best={best_loss:.4f}  lr={opt.lr:.2e}  wall={dt:.1f}s  {free_mb:.0f}MB free')

        if (ep + 1) % GEN_EVERY == 0:
            model.training = False
            prompt_text = 'Q: What is 2 + 3? A:'
            prompt = tok.encode(prompt_text)
            with no_grad():
                ids = model.generate(prompt.copy(), max_new=80,
                                     temperature=TEMPERATURE, top_k=TOP_K)
            print(f'  --- gen ep={ep+1} ---')
            print(f'  {tok.decode(ids)[:200]}')

    dt_total = time.perf_counter() - t0
    print(f'\n  training done in {dt_total:.1f}s  best_loss={best_loss:.4f}')

    weights_path = os.path.join(MODELS_DIR, 'triadssm_lm.weights')
    save_weights(model, weights_path)
    print(f'  saved weights: {weights_path}')

    model2 = TriadSSMLM(
        vocab_size=tok.vocab_size,
        d_model=D_MODEL,
        d_state=D_STATE,
        n_blocks=N_BLOCKS,
        n_memory=N_MEMORY,
        coupling=COUPLING,
    )
    load_weights(model2, weights_path)
    model2.training = False
    with no_grad():
        x_check = tensor(x_np[:1])
        np.random.seed(999)
        o1 = model.forward(x_check)._data
        np.random.seed(999)
        o2 = model2.forward(x_check)._data
    diff = float(np.max(np.abs(o1 - o2)))
    print(f'  load verify: max_diff={diff:.2e}')
    assert diff < 1e-10, f'load mismatch: {diff}'

    prompt = tok.encode('Q: Natalia sold clips to 48 friends in April, and then she sold half as many in May. How many altogether? A:')
    with no_grad():
        final_ids = model2.generate(prompt.copy(), max_new=GEN_TOKENS,
                                    temperature=0.5, top_k=TOP_K)
    final_text = tok.decode(final_ids)
    print('\n  final generation:')
    print(f'  {final_text[:300]}')

    header('quick eval on test set (first 50)')
    model2.training = False
    correct = 0
    total = 0
    for i, ex in enumerate(test_data[:50]):
        question = ex['question']
        gold = gsm8k_extract_answer(ex['answer'])
        prompt_text = 'Q: ' + question + ' A:'
        prompt = tok.encode(prompt_text)
        if len(prompt) > SEQ_LEN:
            prompt = prompt[-SEQ_LEN:]
        try:
            with no_grad():
                ids = model2.generate(prompt, max_new=100, temperature=0.3, top_k=10)
            generated = tok.decode(ids)
            pred = gsm8k_extract_answer(generated[len(prompt_text):])
        except (ValueError, RuntimeError, IndexError):
            pred = ''
        match = pred.strip() == gold.strip()
        if match:
            correct += 1
        total += 1
        if (i + 1) % 10 == 0:
            print(f'  {i+1}/{total}  acc={correct/total*100:.1f}%')

    acc = correct / total * 100 if total else 0
    print(f'  test accuracy: {correct}/{total} = {acc:.1f}%')

    meta = {
        'model_class': 'TriadSSMLM',
        'vocab_size': tok.vocab_size,
        'd_model': D_MODEL,
        'd_state': D_STATE,
        'n_blocks': N_BLOCKS,
        'n_memory': N_MEMORY,
        'coupling': COUPLING,
        'seq_len': SEQ_LEN,
        'batch': BATCH,
        'epochs': EPOCHS,
        'lr': LR,
        'seed': SEED,
        'best_loss': round(best_loss, 4),
        'final_loss': round(err, 4),
        'n_params': n_params,
        'train_s': round(dt_total, 1),
        'test_acc_50': round(acc, 2),
    }
    meta_path = os.path.join(MODELS_DIR, 'triadssm_lm.meta.json')
    Path(meta_path).write_text(json.dumps(meta, indent=2))
    print(f'  saved meta: {meta_path}')

    report_path = os.path.join(MODELS_DIR, 'triadssm_lm_report.json')
    report = {
        'meta': meta,
        'vocab': tok.itos,
        'sample_output': final_text,
    }
    Path(report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'  saved report: {report_path}')

def header(msg: str):
    print(f'\n{"=" * 60}')
    print(f'  {msg}')
    print(f'{"=" * 60}')

if __name__ == '__main__':
    train()

