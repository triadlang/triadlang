from __future__ import annotations
import os
import sys
import time
import json
import numpy as np
from pathlib import Path

TEXT = """
the triad field is a self-organizing system that combines three interacting substrates.
the first substrate is the complex wavefunction psi which evolves under the schrodinger equation.
the second substrate is the memory field which captures nonlocal temporal correlations.
the third substrate is the physical reservoir which provides high dimensional dynamics.
these three substrates are coupled through a three body interaction that produces emergent order.
the key equation governing the dynamics is the triad equation which reads:
i hbar partial t psi equals negative hbar squared over two m laplacian plus v external plus lambda psi squared plus v memory plus alpha minus delta to the sigma over two minus i gamma times psi plus eta
here lambda controls the nonlinearity and gamma is the dissipation rate and eta is the stochastic forcing.
the fluctuation dissipation theorem connects gamma and eta through the relation eta squared equals two gamma k b t.
the memory field introduces a nonlocal potential that depends on the history of the density.
the physical reservoir emulates edge of chaos dynamics where the system sits at the boundary between order and disorder.
at the edge of chaos the system has maximum computational capacity and can perform complex transformations.
the triad ssm block uses complex valued states with frequency omega and decay rate that determine the dynamics.
each block has a memory channel that stores past activation patterns and feeds them back into the computation.
the coupling term ensures that the three substrates interact through a physically motivated mechanism.
the mamba three architecture which inspired this design uses similar complex recurrence with structured state matrices.
our model extends mamba three by adding the memory field and the three body coupling from the triad equation.
the fidelity gate ensures that the surrogate model only accelerates when it can reproduce the solver output accurately.
when the surrogate error exceeds a threshold the system falls back to the native solver for that region.
the attractor observer monitors the long term statistics of the system to detect anomalies and drift.
this passive observer never feeds back into the dynamics preserving the physical integrity of the simulation.
save weights and load weights provide a simple serialization format based on json.
each parameter is stored as a list of floats with its name and shape metadata.
the training loop uses adam optimizer with learning rate scheduling and gradient clipping.
the loss function combines mean squared error on the trajectory data with a physics constraint term.
the physics term penalizes violations of the triad residual ensuring that the model stays physically consistent.
the full model stack includes the solver at layer zero the ssm at layer one the surrogate at layer two and the observer at layer three.
together these four layers form a complete simulation framework that can accelerate physical computations while maintaining accuracy.
the triad language model uses the ssm as its backbone instead of attention.
each ssm block processes the sequence one token at a time maintaining a complex hidden state.
the hidden state carries information from previous tokens enabling context dependent predictions.
the memory field within each block stores patterns of past activations in a nonlocal manner.
the coupling between substrates allows the model to capture multi-scale temporal dependencies.
character level tokenization treats each character as a separate token.
this gives the model fine grained control over text generation at the cost of longer sequences.
the embedding layer converts each character id into a dense vector representation.
the language model head projects the final hidden state back to vocabulary logits.
cross entropy loss compares the predicted logits against the actual next character.
the adam optimizer adapts the learning rate for each parameter individually.
gradient clipping prevents gradient explosion during backpropagation through time.
sampling with temperature scales the logits before converting to probabilities.
lower temperature makes the model more conservative preferring high probability tokens.
higher temperature makes the model more exploratory sampling from a flatter distribution.
top k sampling restricts the candidates to the k most likely tokens at each step.
the generate function performs autoregressive sampling producing one token at a time.
each generated token is appended to the context and fed back into the model.
the triad ssm recurrence is similar to the mamba architecture used in modern language models.
mamba achieves linear time complexity in sequence length unlike the quadratic cost of attention.
this makes the triad language model efficient for long sequences.
the complex valued state space provides richer representations than real valued alternatives.
the omega parameter controls the frequency of oscillation in the hidden state.
the decay parameter controls how quickly information about past tokens fades.
the memory parameter determines how strongly past patterns influence current predictions.
the coupling parameter controls the interaction between the complex state and memory channels.
the physical reservoir interpretation provides theoretical grounding for the architecture.
edge of chaos dynamics suggest that the model operates in a regime of maximum computational power.
the fluctuation dissipation theorem constrains the noise injection to be physically consistent.
together these principles ensure that the triad language model is not just a neural network but a physically motivated system.
the training process adjusts all parameters simultaneously through gradient descent.
the loss landscape is navigated by the adam optimizer which uses momentum and adaptive learning rates.
overfitting occurs when the model memorizes the training data instead of learning general patterns.
regularization techniques like dropout and weight decay can help prevent overfitting.
the current model uses gradient clipping as its primary form of regularization.
larger models with more blocks and higher dimensions can capture more complex patterns.
the triad language model can be scaled up by increasing d_model d_state n_blocks and n_memory.
inference is fast because the ssm recurrence does not require storing attention matrices.
the memory footprint grows linearly with sequence length making long context windows feasible.
the triad language model demonstrates that physically motivated architectures can perform language tasks.
future work includes byte pair encoding tokenization and training on larger text corpora.
the ultimate goal is to build a language model that is both performant and physically interpretable.
""".strip()

D_MODEL = 128
D_STATE = 64
N_BLOCKS = 4
N_MEMORY = 3
SEQ_LEN = 96
BATCH = 8
EPOCHS = 3000
LR = 5e-4
EVAL_EVERY = 300
GEN_EVERY = 600
GEN_TOKENS = 200
TEMPERATURE = 0.8
TOP_K = 20
SEED = 42
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def train():
    from runtime.ml.language import CharTokenizer, TriadLM, TextDataset
    from runtime.ml.nn import Adam
    from runtime.ml.tensor import tensor, no_grad
    from runtime.ml.serialization import save_weights, load_weights

    np.random.seed(SEED)

    tok = CharTokenizer(TEXT)
    tokens = tok.encode(TEXT) * 10
    print(f'vocab_size={tok.vocab_size}  tokens={len(tokens)}  unique_chars={tok.vocab_size}')
    print(f'vocab: {"".join(tok.itos[i] for i in sorted(tok.itos)[:min(40, tok.vocab_size)])}')

    dataset = TextDataset(tokens, SEQ_LEN)
    model = TriadLM(
        vocab_size=tok.vocab_size,
        d_model=D_MODEL,
        d_state=D_STATE,
        n_blocks=N_BLOCKS,
        n_memory=N_MEMORY,
    )
    n_params = sum(p._data.size for p in model.parameters())
    print(f'model: d_model={D_MODEL} N_solver={D_STATE} n_blocks={N_BLOCKS} n_memory={N_MEMORY}')
    print(f'params: {n_params:,}')
    print(f'architecture: full solver (FFT + P1+P2+P3) per block')

    opt = Adam(model.parameters(), lr=LR)

    os.makedirs(MODELS_DIR, exist_ok=True)

    warmup_epochs = 100
    t0 = time.perf_counter()
    best_loss = 999.0
    for ep in range(EPOCHS):
        if ep < warmup_epochs:
            opt.lr = LR * (ep + 1) / warmup_epochs
        elif ep > EPOCHS * 0.7:
            frac = (ep - EPOCHS * 0.7) / (EPOCHS * 0.3)
            opt.lr = LR * (1.0 - 0.8 * frac)
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

        err = float(loss._data)

        if (ep + 1) % EVAL_EVERY == 0:
            dt = time.perf_counter() - t0
            print(f'  epoch {ep+1:4d}/{EPOCHS}  loss={err:.4f}  wall={dt:.1f}s')

        if (ep + 1) % GEN_EVERY == 0:
            model.training = False
            prompt = tokens[:20]
            with no_grad():
                out_ids = model.generate(prompt.copy(), max_new=GEN_TOKENS,
                                         temperature=TEMPERATURE, top_k=TOP_K)
            generated = tok.decode(out_ids)
            prompt_str = tok.decode(prompt)
            print(f'  --- gen ep={ep+1} ---')
            print(f'  prompt: "{prompt_str}"')
            print(f'  output: "{generated}"')

    dt_total = time.perf_counter() - t0
    print(f'\ntraining done in {dt_total:.1f}s')

    model_path = os.path.join(MODELS_DIR, 'triadlm.weights')
    save_weights(model, model_path)
    print(f'saved: {model_path}')

    model2 = TriadLM(
        vocab_size=tok.vocab_size,
        d_model=D_MODEL,
        d_state=D_STATE,
        n_blocks=N_BLOCKS,
        n_memory=N_MEMORY,
    )
    load_weights(model2, model_path)
    model2.training = False
    with no_grad():
        x_check = tensor(x_np[:1])
        np.random.seed(999)
        o1 = model.forward(x_check)._data
        np.random.seed(999)
        o2 = model2.forward(x_check)._data
    diff = float(np.max(np.abs(o1 - o2)))
    print(f'load verify: max_diff={diff:.2e}')
    assert diff < 1e-10, f'load mismatch: {diff}'

    model2.training = False
    prompt = tokens[:10]
    with no_grad():
        final_ids = model2.generate(prompt.copy(), max_new=200,
                                    temperature=0.7, top_k=TOP_K)
    final_text = tok.decode(final_ids)
    print(f'\nfinal generation (200 tokens):')
    print(f'---')
    print(final_text)
    print(f'---')

    meta = {
        'vocab_size': tok.vocab_size,
        'd_model': D_MODEL,
        'd_state': D_STATE,
        'n_blocks': N_BLOCKS,
        'n_memory': N_MEMORY,
        'seq_len': SEQ_LEN,
        'batch': BATCH,
        'epochs': EPOCHS,
        'lr': LR,
        'seed': SEED,
        'final_loss': round(err, 4),
        'n_params': n_params,
        'train_s': round(dt_total, 1),
    }
    meta_path = os.path.join(MODELS_DIR, 'triadlm.meta.json')
    Path(meta_path).write_text(json.dumps(meta, indent=2))
    print(f'saved: {meta_path}')

    report_path = os.path.join(MODELS_DIR, 'triadlm_report.json')
    report = {
        'meta': meta,
        'vocab': tok.itos,
        'sample_output': final_text,
    }
    Path(report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'saved: {report_path}')

if __name__ == '__main__':
    train()
