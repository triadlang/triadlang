from __future__ import annotations

import importlib

from triad import ntri as np

_T = importlib.import_module('runtime.ml.tensor')
_nn = importlib.import_module('runtime.ml.nn')
_tl = importlib.import_module('runtime.ml.triad_layer')

class _FFN(_nn.Module):
    def __init__(self, d_model, mult):
        self.fc1 = _nn.triad(d_model, d_model * mult)
        self.fc2 = _nn.triad(d_model * mult, d_model)

    def forward(self, x):
        return self.fc2(_T.relu(self.fc1(x)))

class TriadBlock(_nn.Module):
    def __init__(self, d_model, n_heads, ffn_mult, **triad_kw):
        self.norm1 = _nn.Parameter(np.ones(d_model))
        self.triad = _tl.TriadLayer(d_model, n_heads=n_heads, **triad_kw)
        self.norm2 = _nn.Parameter(np.ones(d_model))
        self.ffn = _FFN(d_model, ffn_mult)

    def forward(self, x, y_state=None):
        h, y_out = self.triad(_T.layer_norm(x, self.norm1), y_state)

        x = h
        x = x + self.ffn(_T.layer_norm(x, self.norm2))
        return x, y_out

class TriadLM(_nn.Module):

    def __init__(self, vocab_size, d_model=512, n_layers=8, n_heads=8,
                 ffn_mult=5, use_checkpoint=False, **triad_kw):
        self.emb = _nn.Embedding(vocab_size, d_model)
        self.blocks = [TriadBlock(d_model, n_heads, ffn_mult, **triad_kw)
                       for _ in range(n_layers)]
        self.norm_f = _nn.Parameter(np.ones(d_model))
        self.head = _nn.triad(d_model, vocab_size, bias=False)
        self.n_layers = n_layers
        self.use_checkpoint = use_checkpoint

    def forward(self, ids, y_states=None):
        x = self.emb(ids)
        y_out = []
        for i, b in enumerate(self.blocks):
            ys = y_states[i] if y_states else None
            if self.use_checkpoint:
                x, y = _T.checkpoint(lambda xx, _b=b, _ys=ys: _b(xx, _ys), x,
                                     rng_modules=(b.triad,))
            else:
                x, y = b(x, ys)
            y_out.append(y)
        x = _T.layer_norm(x, self.norm_f)
        return self.head(x), y_out

class TransformerBlockLM(_nn.Module):
    def __init__(self, d_model, n_heads, ffn_mult):
        self.norm1 = _nn.Parameter(np.ones(d_model))
        self.attn = _nn.CausalSelfAttention(d_model, n_heads)
        self.norm2 = _nn.Parameter(np.ones(d_model))
        self.ffn = _FFN(d_model, ffn_mult)

    def forward(self, x):
        x = x + self.attn(_T.layer_norm(x, self.norm1))
        x = x + self.ffn(_T.layer_norm(x, self.norm2))
        return x

class TransformerLM(_nn.Module):

    def __init__(self, vocab_size, d_model=512, n_layers=8, n_heads=8,
                 ffn_mult=4, use_checkpoint=False):
        self.emb = _nn.Embedding(vocab_size, d_model)
        self.blocks = [TransformerBlockLM(d_model, n_heads, ffn_mult)
                       for _ in range(n_layers)]
        self.norm_f = _nn.Parameter(np.ones(d_model))
        self.head = _nn.triad(d_model, vocab_size, bias=False)
        self.use_checkpoint = use_checkpoint

    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks:
            if self.use_checkpoint:
                x = _T.checkpoint(lambda xx, _b=b: _b(xx), x)
            else:
                x = b(x)
        x = _T.layer_norm(x, self.norm_f)
        return self.head(x)
