
from __future__ import annotations

from typing import TYPE_CHECKING

from triad import ntri as np

if TYPE_CHECKING:
    from runtime.ml.generate import GenerationResult

EXTERNAL_STACK = True

_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
    torch = None

def _ensure_torch():
    if not _HAS_TORCH:
        raise RuntimeError(
            'PyTorch is not installed. This module (runtime.ml.forward) is '
            'an **external interop backend** and requires torch. '
            'Install with: pip install torch'
        )

_device = None

def _get_device() -> torch.device:
    _ensure_torch()
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device

def _t(arr: np.ndarray) -> torch.Tensor:
    _ensure_torch()
    return torch.as_tensor(arr, device=_get_device())

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight

def rms_norm_qwen35(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x_normed = x.float() * torch.rsqrt(variance + eps)
    return (x_normed * (1.0 + weight.float())).to(x.dtype)

def silu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(x)

def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm

def rope(x: torch.Tensor, positions: torch.Tensor,
         head_dim: int, base: float = 10000.0,
         partial_rotary_factor: float = 1.0) -> torch.Tensor:
    rotary_dim = int(head_dim * partial_rotary_factor)
    half = rotary_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=x.device) / rotary_dim))
    theta = positions[:, None] * freqs[None, :]
    cos_t = torch.cos(theta)[None, :, None, :]
    sin_t = torch.sin(theta)[None, :, None, :]
    cos_t = torch.cat([cos_t, cos_t], dim=-1)
    sin_t = torch.cat([sin_t, sin_t], dim=-1)
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    rotated_triad = x_rot * cos_t + rotated * sin_t
    if rotary_dim < head_dim:
        return torch.cat([rotated_triad, x_pass], dim=-1)
    return rotated_triad

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def _torch_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64,
                                   initial_state=None, output_final_state=False):
    initial_dtype = query.dtype
    query = _l2norm(query.float(), dim=-1)
    key = _l2norm(key.float(), dim=-1)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = torch.nn.functional.pad(query, (0, 0, 0, pad_size))
    key = torch.nn.functional.pad(key, (0, 0, 0, pad_size))
    value = torch.nn.functional.pad(value, (0, 0, 0, pad_size))
    beta = torch.nn.functional.pad(beta, (0, pad_size))
    g = torch.nn.functional.pad(g, (0, pad_size))
    total_seq = sequence_length + pad_size
    scale = query.shape[-1] ** -0.5
    query = query * scale
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask_tri = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g_cum = g.cumsum(dim=-1)
    decay_mask = ((g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask_tri, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask_diag = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    for i in range(total_seq // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_i @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state

def _torch_recurrent_gated_delta_rule(query, key, value, g, beta,
                                       initial_state=None, output_final_state=False):
    initial_dtype = query.dtype
    query = _l2norm(query.float(), dim=-1)
    key = _l2norm(key.float(), dim=-1)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = query.shape[-1] ** -0.5
    query = query * scale
    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim, dtype=value.dtype, device=value.device
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state

def _rms_norm_gated(hidden_states: torch.Tensor, gate: torch.Tensor,
                    weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states.float() * torch.rsqrt(variance + eps)
    hidden_states = (weight.float() * hidden_states).to(hidden_states.dtype)
    hidden_states = hidden_states * torch.nn.functional.silu(gate.float())
    return hidden_states.to(hidden_states.dtype)

def _causal_conv1d_forward(x: torch.Tensor, weight: torch.Tensor,
                            bias: torch.Tensor | None = None) -> torch.Tensor:
    _, conv_dim, seq_len = x.shape
    kernel_size = weight.shape[-1]
    padded = torch.nn.functional.pad(x, (kernel_size - 1, 0))
    out = torch.nn.functional.conv1d(padded, weight.unsqueeze(1), bias,
                                      padding=0, groups=conv_dim)
    out = torch.nn.functional.silu(out[:, :, :seq_len])
    return out

def _causal_conv1d_update(x: torch.Tensor, conv_state: torch.Tensor,
                           weight: torch.Tensor,
                           bias: torch.Tensor | None = None) -> torch.Tensor:
    _, conv_dim, seq_len = x.shape
    state_len = conv_state.shape[-1]
    x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
    conv_state.copy_(x_new[:, :, -state_len:])
    out = torch.nn.functional.conv1d(x_new, weight.unsqueeze(1), bias,
                                      padding=0, groups=conv_dim)
    out = torch.nn.functional.silu(out[:, :, -seq_len:])
    return out.to(x.dtype)

def forward_triad_attn_layer(x: torch.Tensor, layer_idx: int,
                               weights: dict[str, torch.Tensor],
                               config: dict,
                               prefix: str = "model",
                               layer_state: dict | None = None) -> tuple[torch.Tensor, dict | None]:
    p = f"{prefix}.layers.{layer_idx}.triad_attn"
    n_kv = config.get("triad_num_value_heads", 32)
    n_k = config.get("triad_num_key_heads", 16)
    head_k_dim = config.get("triad_key_head_dim", 128)
    head_v_dim = config.get("triad_value_head_dim", 128)
    conv_kernel = config.get("triad_conv_kernel_dim", 4)
    key_dim = n_k * head_k_dim
    value_dim = n_kv * head_v_dim

    inp_norm_w = weights[f"{prefix}.layers.{layer_idx}.input_layernorm.weight"]
    h = rms_norm_qwen35(x, inp_norm_w)
    seq_len = x.shape[0]

    mixed_qkv = h @ weights[f"{p}.in_proj_qkv.weight"].T
    mixed_qkv = mixed_qkv.reshape(1, seq_len, key_dim * 2 + value_dim).transpose(1, 2)

    z = (h @ weights[f"{p}.in_proj_z.weight"].T).reshape(1, seq_len, n_kv, head_v_dim)
    b = h @ weights[f"{p}.in_proj_b.weight"].T
    a = h @ weights[f"{p}.in_proj_a.weight"].T

    conv_w = weights[f"{p}.conv1d.weight"].T
    conv_b = weights.get(f"{p}.conv1d.bias", None)

    has_state = layer_state is not None and "conv_state" in layer_state

    if has_state and seq_len == 1:
        mixed_qkv = _causal_conv1d_update(mixed_qkv, layer_state["conv_state"], conv_w, conv_b)
    else:
        if has_state:
            mixed_qkv = torch.cat([layer_state["conv_state"], mixed_qkv], dim=-1)
        new_conv_state_raw = torch.nn.functional.pad(
            mixed_qkv, (conv_kernel - mixed_qkv.shape[-1], 0)
        )
        if layer_state is not None:
            layer_state["conv_state"] = new_conv_state_raw[:, :, -conv_kernel:].clone()
        mixed_qkv = _causal_conv1d_forward(mixed_qkv, conv_w, conv_b)
        if has_state:
            mixed_qkv = mixed_qkv[:, :, -seq_len:]

    mixed_qkv = mixed_qkv.transpose(1, 2)
    q, k, v = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
    q = q.reshape(1, seq_len, n_k, head_k_dim)
    k = k.reshape(1, seq_len, n_k, head_k_dim)
    v = v.reshape(1, seq_len, n_kv, head_v_dim)

    beta = b.reshape(1, seq_len, n_kv).sigmoid()
    A_log = weights[f"{p}.A_log"]
    dt_bias = weights[f"{p}.dt_bias"]
    g = -A_log.float().exp() * torch.nn.functional.softplus(a.float() + dt_bias.float())
    g = g.reshape(1, seq_len, n_kv)

    if n_kv // n_k > 1:
        q = q.repeat_interleave(n_kv // n_k, dim=2)
        k = k.repeat_interleave(n_kv // n_k, dim=2)

    has_recurrent = layer_state is not None and "recurrent_state" in layer_state
    init_state = layer_state.get("recurrent_state") if has_recurrent else None

    if has_recurrent and seq_len == 1:
        core_out, last_state = _torch_recurrent_gated_delta_rule(
            q, k, v, g, beta, init_state,
            output_final_state=layer_state is not None
        )
    else:
        core_out, last_state = _torch_chunk_gated_delta_rule(
            q, k, v, g, beta, chunk_size=64,
            initial_state=init_state,
            output_final_state=layer_state is not None
        )

    if layer_state is not None and last_state is not None:
        layer_state["recurrent_state"] = last_state

    norm_w = weights[f"{p}.norm.weight"]
    core_out_2d = core_out.reshape(-1, head_v_dim)
    z_2d = z.reshape(-1, head_v_dim)
    core_out_2d = _rms_norm_gated(core_out_2d, z_2d, norm_w)
    core_out = core_out_2d.reshape(1, seq_len, -1)

    out = core_out @ weights[f"{p}.out_proj.weight"].T
    return x + out[0], layer_state

def forward_attention_layer_qwen35(x: torch.Tensor, layer_idx: int,
                                    weights: dict[str, torch.Tensor],
                                    config: dict,
                                    positions: torch.Tensor,
                                    kv_cache: list | None = None,
                                    prefix: str = "model") -> tuple[torch.Tensor, list | None]:
    p = f"{prefix}.layers.{layer_idx}.self_attn"
    n_heads = config.get("num_attention_heads", 16)
    n_kv_heads = config.get("num_key_value_heads", 4)
    head_dim = config.get("head_dim", 256)
    rope_params = config.get("rope_parameters", {})
    rope_base = rope_params.get("rope_theta", 10000.0)
    partial_rotary = rope_params.get("partial_rotary_factor", 1.0)
    eps = config.get("rms_norm_eps", 1e-6)

    inp_norm_w = weights[f"{prefix}.layers.{layer_idx}.input_layernorm.weight"]
    h = rms_norm_qwen35(x, inp_norm_w)

    q_w = weights[f"{p}.q_proj.weight"]
    k_w = weights[f"{p}.k_proj.weight"]
    v_w = weights[f"{p}.v_proj.weight"]
    o_w = weights[f"{p}.o_proj.weight"]
    seq_len = x.shape[0]

    q_gated = (h @ q_w.T).reshape(seq_len, n_heads, head_dim * 2)
    q, gate = torch.chunk(q_gated, 2, dim=-1)
    gate = gate.reshape(seq_len, -1)

    k = (h @ k_w.T).reshape(seq_len, n_kv_heads, head_dim)
    v = (h @ v_w.T).reshape(seq_len, n_kv_heads, head_dim)

    q_nw = weights[f"{p}.q_norm.weight"]
    k_nw = weights[f"{p}.k_norm.weight"]
    q = rms_norm_qwen35(q.reshape(-1, head_dim), q_nw, eps).reshape(seq_len, n_heads, head_dim)
    k = rms_norm_qwen35(k.reshape(-1, head_dim), k_nw, eps).reshape(seq_len, n_kv_heads, head_dim)

    q = rope(q.reshape(1, seq_len, n_heads, head_dim), positions, head_dim, rope_base, partial_rotary)[0]
    k = rope(k.reshape(1, seq_len, n_kv_heads, head_dim), positions, head_dim, rope_base, partial_rotary)[0]

    if kv_cache is not None:
        if len(kv_cache) == 0:
            kv_cache.append(torch.zeros(1, n_kv_heads, 0, head_dim, dtype=k.dtype, device=k.device))
            kv_cache.append(torch.zeros(1, n_kv_heads, 0, head_dim, dtype=v.dtype, device=v.device))
        cached_k, cached_v = kv_cache[0], kv_cache[1]
        if cached_k.shape[-2] > 0:
            k_triad = torch.cat([cached_k[0], k], dim=0)
            v_triad = torch.cat([cached_v[0], v], dim=0)
        else:
            k_triad = k
            v_triad = v
        kv_cache[0] = k_triad[None]
        kv_cache[1] = v_triad[None]
    else:
        k_triad = k
        v_triad = v

    n_rep = n_heads // n_kv_heads
    if n_rep > 1:
        k_triad = k_triad.repeat_interleave(n_rep, dim=1)
        v_triad = v_triad.repeat_interleave(n_rep, dim=1)

    q_t = q.transpose(0, 1)[None]
    k_t = k_triad.transpose(0, 1)[None]
    v_t = v_triad.transpose(0, 1)[None]

    total_len = k_triad.shape[0]
    if kv_cache is not None:
        mask = torch.zeros(1, 1, seq_len, total_len, dtype=x.dtype, device=x.device)
        start = total_len - seq_len
        for i in range(seq_len):
            mask[0, 0, i, start + i + 1:] = -1e9
    else:
        causal = torch.triu(torch.triad((seq_len, seq_len), -1e9, dtype=x.dtype, device=x.device), diagonal=1)
        mask = causal[None, None]

    scale = head_dim ** -0.5
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale + mask
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v_t)

    out = out[0].transpose(0, 1).reshape(seq_len, -1)
    out = out * torch.sigmoid(gate)
    attn_out = out @ o_w.T
    return x + attn_out, kv_cache

def forward_attention_layer(x: torch.Tensor, layer_idx: int,
                            weights: dict[str, torch.Tensor],
                            config: dict,
                            positions: torch.Tensor,
                            kv_cache: list | None = None,
                            prefix: str = "model") -> tuple[torch.Tensor, list | None]:
    p = f"{prefix}.layers.{layer_idx}.self_attn"
    n_heads = config.get("num_attention_heads", config.get("n_head", 32))
    n_kv_heads = config.get("num_key_value_heads", n_heads)
    head_dim = config.get("head_dim", config.get("hidden_size", 1024) // n_heads)
    rope_base = config.get("rope_theta", 10000.0)

    inp_norm_w = weights[f"{prefix}.layers.{layer_idx}.input_layernorm.weight"]
    h = rms_norm(x, inp_norm_w)

    q_w = weights[f"{p}.q_proj.weight"]
    k_w = weights[f"{p}.k_proj.weight"]
    v_w = weights[f"{p}.v_proj.weight"]
    o_w = weights[f"{p}.o_proj.weight"]

    q = (h @ q_w.T).reshape(-1, n_heads, head_dim)
    k = (h @ k_w.T).reshape(-1, n_kv_heads, head_dim)
    v = (h @ v_w.T).reshape(-1, n_kv_heads, head_dim)
    seq_len = x.shape[0]

    if f"{p}.q_norm.weight" in weights:
        q_nw = weights[f"{p}.q_norm.weight"]
        k_nw = weights[f"{p}.k_norm.weight"]
        q = rms_norm(q.reshape(-1, head_dim), q_nw).reshape(seq_len, n_heads, head_dim)
        k = rms_norm(k.reshape(-1, head_dim), k_nw).reshape(seq_len, n_kv_heads, head_dim)

    q = rope(q.reshape(1, seq_len, n_heads, head_dim), positions, head_dim, rope_base)[0]
    k = rope(k.reshape(1, seq_len, n_kv_heads, head_dim), positions, head_dim, rope_base)[0]

    if kv_cache is not None:
        if len(kv_cache) == 0:
            kv_cache.append(torch.zeros(1, n_kv_heads, 0, head_dim, dtype=k.dtype, device=k.device))
            kv_cache.append(torch.zeros(1, n_kv_heads, 0, head_dim, dtype=v.dtype, device=v.device))
        cached_k, cached_v = kv_cache[0], kv_cache[1]
        if cached_k.shape[-2] > 0:
            k_triad = torch.cat([cached_k[0], k], dim=0)
            v_triad = torch.cat([cached_v[0], v], dim=0)
        else:
            k_triad = k
            v_triad = v
        kv_cache[0] = k_triad[None]
        kv_cache[1] = v_triad[None]
    else:
        k_triad = k
        v_triad = v

    n_rep = n_heads // n_kv_heads
    if n_rep > 1:
        k_triad = k_triad.repeat_interleave(n_rep, dim=1)
        v_triad = v_triad.repeat_interleave(n_rep, dim=1)

    q_t = q.transpose(0, 1)[None]
    k_t = k_triad.transpose(0, 1)[None]
    v_t = v_triad.transpose(0, 1)[None]

    total_len = k_triad.shape[0]
    if kv_cache is not None:
        mask = torch.zeros(1, 1, seq_len, total_len, dtype=x.dtype, device=x.device)
        start = total_len - seq_len
        for i in range(seq_len):
            mask[0, 0, i, start + i + 1:] = -1e9
    else:
        causal = torch.triu(torch.triad((seq_len, seq_len), -1e9, dtype=x.dtype, device=x.device), diagonal=1)
        mask = causal[None, None]

    scale = head_dim ** -0.5
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale + mask
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v_t)

    out = out[0].transpose(0, 1).reshape(seq_len, -1)
    attn_out = out @ o_w.T
    return x + attn_out, kv_cache

def forward_ffn(h: torch.Tensor, layer_idx: int,
                weights: dict[str, torch.Tensor],
                prefix: str = "model") -> torch.Tensor:
    p = f"{prefix}.layers.{layer_idx}"
    norm_w = weights[f"{p}.post_attention_layernorm.weight"]
    h_normed = rms_norm(h, norm_w)

    gate_w = weights[f"{p}.mlp.gate_proj.weight"]
    up_w = weights[f"{p}.mlp.up_proj.weight"]
    down_w = weights[f"{p}.mlp.down_proj.weight"]

    gate = h_normed @ gate_w.T
    up = h_normed @ up_w.T
    return h + (silu(gate) * up) @ down_w.T

def forward_transformer(model, token_ids: np.ndarray,
                        kv_caches: list | None = None) -> np.ndarray:
    dev = _get_device()
    w_gpu = {}
    if not hasattr(model, '_gpu_weights') or model._gpu_weights is None:
        for k, v in model.weights.items():
            w_gpu[k] = torch.as_tensor(v, device=dev)
        model._gpu_weights = w_gpu
    else:
        w_gpu = model._gpu_weights

    cfg = model.config
    n_layers = model.n_layers
    prefix = "model"

    embed_w = w_gpu[f"{prefix}.embed_tokens.weight"]
    ids_t = torch.as_tensor(token_ids, dtype=torch.long, device=dev)
    x = embed_w[ids_t]

    if kv_caches is not None and len(kv_caches) == 0:
        for _ in range(n_layers):
            kv_caches.append([])

    seq_len = x.shape[0]
    if kv_caches is not None and len(kv_caches) > 0 and len(kv_caches[0]) >= 2:
        cached_len = kv_caches[0][0].shape[1]
        start_pos = cached_len
    else:
        start_pos = 0
    positions = torch.arange(start_pos, start_pos + seq_len, dtype=torch.float32, device=dev)

    for i in range(n_layers):
        layer_kv = kv_caches[i] if kv_caches is not None else None
        x, updated_kv = forward_attention_layer(x, i, w_gpu, cfg, positions, layer_kv, prefix)
        if kv_caches is not None:
            kv_caches[i] = updated_kv
        x = forward_ffn(x, i, w_gpu, prefix)

    norm_w = w_gpu[f"{prefix}.norm.weight"]
    x = rms_norm(x, norm_w)

    lm_head_w = w_gpu.get("lm_head.weight", embed_w)
    logits = x @ lm_head_w.T
    return logits.cpu().numpy()

def forward_ffn_qwen35(h: torch.Tensor, layer_idx: int,
                       weights: dict[str, torch.Tensor],
                       prefix: str = "model") -> torch.Tensor:
    p = f"{prefix}.layers.{layer_idx}"
    norm_w = weights[f"{p}.post_attention_layernorm.weight"]
    h_normed = rms_norm_qwen35(h, norm_w)
    gate_w = weights[f"{p}.mlp.gate_proj.weight"]
    up_w = weights[f"{p}.mlp.up_proj.weight"]
    down_w = weights[f"{p}.mlp.down_proj.weight"]
    gate = h_normed @ gate_w.T
    up = h_normed @ up_w.T
    return h + (silu(gate) * up) @ down_w.T

def forward_transformer_qwen35(model, token_ids: np.ndarray,
                                kv_caches: list | None = None) -> np.ndarray:
    from runtime.ml.model_loader import TriadModelGGUF
    dev = _get_device()

    if isinstance(model, TriadModelGGUF):
        return _forward_qwen35_gguf_streaming(model, token_ids, kv_caches, dev)

    if not hasattr(model, '_gpu_weights') or model._gpu_weights is None:
        w_gpu = {}
        for k, v in model.weights.items():
            w_gpu[k] = torch.as_tensor(v, device=dev)
        model._gpu_weights = w_gpu
    else:
        w_gpu = model._gpu_weights

    cfg = model.config
    n_layers = model.n_layers
    layer_types = cfg.get("layer_types", ["triad_attention"] * n_layers)
    prefix = "model.language_model"

    embed_w = w_gpu[f"{prefix}.embed_tokens.weight"]
    ids_t = torch.as_tensor(token_ids, dtype=torch.long, device=dev)
    x = embed_w[ids_t]

    if kv_caches is not None and len(kv_caches) == 0:
        for i in range(n_layers):
            if layer_types[i] == "triad_attention":
                kv_caches.append({})
            else:
                kv_caches.append([])

    seq_len = x.shape[0]
    triad_attn_indices = [i for i in range(n_layers) if layer_types[i] == "triad_attention"]
    if kv_caches is not None and triad_attn_indices:
        first_triad = kv_caches[triad_attn_indices[0]]
        if isinstance(first_triad, list) and len(first_triad) >= 2:
            start_pos = first_triad[0].shape[1]
        else:
            start_pos = 0
    else:
        start_pos = 0
    positions = torch.arange(start_pos, start_pos + seq_len, dtype=torch.float32, device=dev)

    for i in range(n_layers):
        layer_cache = kv_caches[i] if kv_caches is not None else None
        if layer_types[i] == "triad_attention":
            x, _ = forward_triad_attn_layer(x, i, w_gpu, cfg, prefix, layer_cache)
            if kv_caches is not None:
                kv_caches[i] = layer_cache
        else:
            x, updated_kv = forward_attention_layer_qwen35(x, i, w_gpu, cfg, positions, layer_cache, prefix)
            if kv_caches is not None:
                kv_caches[i] = updated_kv
        x = forward_ffn_qwen35(x, i, w_gpu, prefix)

    norm_w = w_gpu[f"{prefix}.norm.weight"]
    x = rms_norm_qwen35(x, norm_w)

    lm_head_w = w_gpu.get("lm_head.weight", embed_w)
    logits = x @ lm_head_w.T
    return logits.cpu().numpy()

def _forward_qwen35_gguf_streaming(model, token_ids, kv_caches, dev):
    import torch.cuda
    cfg = model.config
    n_layers = model.n_layers
    layer_types = cfg.get("layer_types", ["triad_attention"] * n_layers)

    embed_w = model.get_embed_weight()
    ids_t = torch.as_tensor(token_ids, dtype=torch.long, device=dev)
    x = embed_w[ids_t]
    del embed_w

    if kv_caches is not None and len(kv_caches) == 0:
        for i in range(n_layers):
            if layer_types[i] == "triad_attention":
                kv_caches.append({})
            else:
                kv_caches.append([])

    seq_len = x.shape[0]
    triad_attn_indices = [i for i in range(n_layers) if layer_types[i] == "triad_attention"]
    if kv_caches is not None and triad_attn_indices:
        first_triad = kv_caches[triad_attn_indices[0]]
        if isinstance(first_triad, list) and len(first_triad) >= 2:
            start_pos = first_triad[0].shape[1]
        else:
            start_pos = 0
    else:
        start_pos = 0
    positions = torch.arange(start_pos, start_pos + seq_len, dtype=torch.float32, device=dev)

    for i in range(n_layers):
        lw = model.get_layer_weights(i)
        layer_cache = kv_caches[i] if kv_caches is not None else None

        if layer_types[i] == "triad_attention":
            x, _ = _forward_triad_attn_gguf(x, i, lw, cfg, layer_cache)
            if kv_caches is not None:
                kv_caches[i] = layer_cache
        else:
            x, updated_kv = _forward_triad_attn_gguf(x, i, lw, cfg, positions, layer_cache)
            if kv_caches is not None:
                kv_caches[i] = updated_kv
        x = _forward_ffn_gguf(x, lw)
        del lw
        torch.cuda.empty_cache()

    norm_w = model.get_norm_weight()
    x = rms_norm_qwen35(x, norm_w)
    del norm_w

    lm_head_w = model.get_lm_head_weight()
    logits = x @ lm_head_w.T
    del lm_head_w
    return logits.cpu().numpy()

def _forward_triad_attn_gguf(x, layer_idx, lw, cfg, positions, kv_cache):
    n_heads = cfg.get("num_attention_heads", 16)
    n_kv_heads = cfg.get("num_key_value_heads", 4)
    head_dim = cfg.get("head_dim", 256)
    rope_params = cfg.get("rope_parameters", {})
    rope_base = rope_params.get("rope_theta", 10000.0)
    partial_rotary = rope_params.get("partial_rotary_factor", 1.0)
    eps = cfg.get("rms_norm_eps", 1e-6)
    seq_len = x.shape[0]

    h = rms_norm_qwen35(x, lw["input_layernorm.weight"])
    q_gated = (h @ lw["q_proj.weight"].T).reshape(seq_len, n_heads, head_dim * 2)
    q, gate = torch.chunk(q_gated, 2, dim=-1)
    gate = gate.reshape(seq_len, -1)
    k = (h @ lw["k_proj.weight"].T).reshape(seq_len, n_kv_heads, head_dim)
    v = (h @ lw["v_proj.weight"].T).reshape(seq_len, n_kv_heads, head_dim)
    q = rms_norm_qwen35(q.reshape(-1, head_dim), lw["q_norm.weight"], eps).reshape(seq_len, n_heads, head_dim)
    k = rms_norm_qwen35(k.reshape(-1, head_dim), lw["k_norm.weight"], eps).reshape(seq_len, n_kv_heads, head_dim)
    q = rope(q.reshape(1, seq_len, n_heads, head_dim), positions, head_dim, rope_base, partial_rotary)[0]
    k = rope(k.reshape(1, seq_len, n_kv_heads, head_dim), positions, head_dim, rope_base, partial_rotary)[0]

    if kv_cache is not None:
        if len(kv_cache) == 0:
            kv_cache.append(torch.zeros(1, n_kv_heads, 0, head_dim, dtype=k.dtype, device=k.device))
            kv_cache.append(torch.zeros(1, n_kv_heads, 0, head_dim, dtype=v.dtype, device=v.device))
        cached_k, cached_v = kv_cache[0], kv_cache[1]
        if cached_k.shape[-2] > 0:
            k_triad = torch.cat([cached_k[0], k], dim=0)
            v_triad = torch.cat([cached_v[0], v], dim=0)
        else:
            k_triad = k
            v_triad = v
        kv_cache[0] = k_triad[None]
        kv_cache[1] = v_triad[None]
    else:
        k_triad = k
        v_triad = v

    n_rep = n_heads // n_kv_heads
    if n_rep > 1:
        k_triad = k_triad.repeat_interleave(n_rep, dim=1)
        v_triad = v_triad.repeat_interleave(n_rep, dim=1)

    q_t = q.transpose(0, 1)[None]
    k_t = k_triad.transpose(0, 1)[None]
    v_t = v_triad.transpose(0, 1)[None]
    total_len = k_triad.shape[0]
    if kv_cache is not None:
        mask = torch.zeros(1, 1, seq_len, total_len, dtype=x.dtype, device=x.device)
        start = total_len - seq_len
        for i in range(seq_len):
            mask[0, 0, i, start + i + 1:] = -1e9
    else:
        causal = torch.triu(torch.triad((seq_len, seq_len), -1e9, dtype=x.dtype, device=x.device), diagonal=1)
        mask = causal[None, None]
    scale = head_dim ** -0.5
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale + mask
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v_t)
    out = out[0].transpose(0, 1).reshape(seq_len, -1)
    out = out * torch.sigmoid(gate)
    attn_out = out @ lw["o_proj.weight"].T
    return x + attn_out, kv_cache

def _forward_triad_attn_gguf(x, layer_idx, lw, cfg, layer_state):
    n_kv = cfg.get("triad_num_value_heads", 32)
    n_k = cfg.get("triad_num_key_heads", 16)
    head_k_dim = cfg.get("triad_key_head_dim", 128)
    head_v_dim = cfg.get("triad_value_head_dim", 128)
    conv_kernel = cfg.get("triad_conv_kernel_dim", 4)
    key_dim = n_k * head_k_dim
    value_dim = n_kv * head_v_dim
    seq_len = x.shape[0]

    h = rms_norm_qwen35(x, lw["input_layernorm.weight"])
    mixed_qkv = h @ lw["in_proj_qkv.weight"].T
    mixed_qkv = mixed_qkv.reshape(1, seq_len, key_dim * 2 + value_dim).transpose(1, 2)
    z = (h @ lw["in_proj_z.weight"].T).reshape(1, seq_len, n_kv, head_v_dim)
    b = h @ lw["in_proj_b.weight"].T
    a = h @ lw["in_proj_a.weight"].T

    conv_w = lw["conv1d.weight"]
    conv_w = conv_w.permute(1, 0) if conv_w.shape[0] == conv_kernel else conv_w

    has_state = layer_state is not None and "conv_state" in layer_state
    if has_state and seq_len == 1:
        mixed_qkv = _causal_conv1d_update(mixed_qkv, layer_state["conv_state"], conv_w, None)
    else:
        if has_state:
            mixed_qkv = torch.cat([layer_state["conv_state"], mixed_qkv], dim=-1)
        new_conv_state_raw = torch.nn.functional.pad(
            mixed_qkv, (conv_kernel - mixed_qkv.shape[-1], 0)
        )
        if layer_state is not None:
            layer_state["conv_state"] = new_conv_state_raw[:, :, -conv_kernel:].clone()
        mixed_qkv = _causal_conv1d_forward(mixed_qkv, conv_w, None)
        if has_state:
            mixed_qkv = mixed_qkv[:, :, -seq_len:]

    mixed_qkv = mixed_qkv.transpose(1, 2)
    q, k, v = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
    q = q.reshape(1, seq_len, n_k, head_k_dim)
    k = k.reshape(1, seq_len, n_k, head_k_dim)
    v = v.reshape(1, seq_len, n_kv, head_v_dim)
    beta = b.reshape(1, seq_len, n_kv).sigmoid()
    A_log = lw["A_log"]
    dt_bias = lw["dt_bias"]
    g = -A_log.float().exp() * torch.nn.functional.softplus(a.float() + dt_bias.float())
    g = g.reshape(1, seq_len, n_kv)
    if n_kv // n_k > 1:
        q = q.repeat_interleave(n_kv // n_k, dim=2)
        k = k.repeat_interleave(n_kv // n_k, dim=2)
    has_recurrent = layer_state is not None and "recurrent_state" in layer_state
    init_state = layer_state.get("recurrent_state") if has_recurrent else None
    if has_recurrent and seq_len == 1:
        core_out, last_state = _torch_recurrent_gated_delta_rule(
            q, k, v, g, beta, init_state, output_final_state=layer_state is not None
        )
    else:
        core_out, last_state = _torch_chunk_gated_delta_rule(
            q, k, v, g, beta, chunk_size=64,
            initial_state=init_state, output_final_state=layer_state is not None
        )
    if layer_state is not None and last_state is not None:
        layer_state["recurrent_state"] = last_state
    norm_w = lw["norm.weight"]
    core_out_2d = core_out.reshape(-1, head_v_dim)
    z_2d = z.reshape(-1, head_v_dim)
    core_out_2d = _rms_norm_gated(core_out_2d, z_2d, norm_w)
    core_out = core_out_2d.reshape(1, seq_len, -1)
    out = core_out @ lw["out_proj.weight"].T
    return x + out[0], layer_state

def _forward_ffn_gguf(h, lw):
    h_normed = rms_norm_qwen35(h, lw["post_attention_layernorm.weight"])
    gate = h_normed @ lw["gate_proj.weight"].T
    up = h_normed @ lw["up_proj.weight"].T
    return h + (silu(gate) * up) @ lw["down_proj.weight"].T

def forward(model, token_ids: list[int] | np.ndarray,
            kv_caches: list | None = None) -> np.ndarray:
    if isinstance(token_ids, list):
        token_ids = np.array(token_ids, dtype=np.int64)
    arch = model.arch
    if arch == "qwen3_5":
        return forward_transformer_qwen35(model, token_ids, kv_caches)
    if arch in ("llama", "qwen2", "qwen3", "mistral", "gemma"):
        return forward_transformer(model, token_ids, kv_caches)
    if arch == "gemma2":
        return forward_transformer_gemma2(model, token_ids, kv_caches)
    if arch == "mixtral":
        return forward_transformer_mixtral(model, token_ids, kv_caches)
    raise NotImplementedError(f"forward pass not implemented for arch: {arch}")

def forward_attention_layer_gemma2(x: torch.Tensor, layer_idx: int,
                                    weights: dict[str, torch.Tensor],
                                    config: dict,
                                    positions: torch.Tensor,
                                    kv_cache: list | None = None,
                                    prefix: str = "model") -> tuple[torch.Tensor, list | None]:
    p = f"{prefix}.layers.{layer_idx}.self_attn"
    n_heads = config.get("num_attention_heads", 16)
    n_kv_heads = config.get("num_key_value_heads", 4)
    head_dim = config.get("head_dim", 256)
    rope_base = config.get("rope_theta", 10000.0)
    eps = config.get("rms_norm_eps", 1e-6)
    sliding_window = config.get("sliding_window", None)
    logits_softcap = config.get("attn_logit_softcapping", None)
    seq_len = x.shape[0]

    inp_norm_w = weights[f"{prefix}.layers.{layer_idx}.input_layernorm.weight"]
    h = rms_norm(x, inp_norm_w)

    q_w = weights[f"{p}.q_proj.weight"]
    k_w = weights[f"{p}.k_proj.weight"]
    v_w = weights[f"{p}.v_proj.weight"]
    o_w = weights[f"{p}.o_proj.weight"]

    q = (h @ q_w.T).reshape(seq_len, n_heads, head_dim)
    k = (h @ k_w.T).reshape(seq_len, n_kv_heads, head_dim)
    v = (h @ v_w.T).reshape(seq_len, n_kv_heads, head_dim)

    q = rope(q.reshape(1, seq_len, n_heads, head_dim), positions, head_dim, rope_base)[0]
    k = rope(k.reshape(1, seq_len, n_kv_heads, head_dim), positions, head_dim, rope_base)[0]

    if kv_cache is not None:
        if len(kv_cache) == 0:
            kv_cache.append(torch.zeros(1, n_kv_heads, 0, head_dim, dtype=k.dtype, device=k.device))
            kv_cache.append(torch.zeros(1, n_kv_heads, 0, head_dim, dtype=v.dtype, device=v.device))
        cached_k, cached_v = kv_cache[0], kv_cache[1]
        if cached_k.shape[-2] > 0:
            k_triad = torch.cat([cached_k[0], k], dim=0)
            v_triad = torch.cat([cached_v[0], v], dim=0)
        else:
            k_triad = k
            v_triad = v
        kv_cache[0] = k_triad[None]
        kv_cache[1] = v_triad[None]
    else:
        k_triad = k
        v_triad = v

    n_rep = n_heads // n_kv_heads
    if n_rep > 1:
        k_triad = k_triad.repeat_interleave(n_rep, dim=1)
        v_triad = v_triad.repeat_interleave(n_rep, dim=1)

    q_t = q.transpose(0, 1)[None]
    k_t = k_triad.transpose(0, 1)[None]
    v_t = v_triad.transpose(0, 1)[None]

    total_len = k_triad.shape[0]
    scale = head_dim ** -0.5
    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale

    if logits_softcap is not None and logits_softcap > 0:
        scores = logits_softcap * torch.tanh(scores / logits_softcap)

    if kv_cache is not None:
        mask = torch.zeros(1, 1, seq_len, total_len, dtype=x.dtype, device=x.device)
        start = total_len - seq_len
        for i in range(seq_len):
            if sliding_window is not None:
                ws = max(0, start + i - sliding_window + 1)
                mask[0, 0, i, :ws] = -1e9
            mask[0, 0, i, start + i + 1:] = -1e9
    else:
        causal = torch.triu(torch.triad((seq_len, seq_len), -1e9, dtype=x.dtype, device=x.device), diagonal=1)
        mask = causal[None, None]

    scores = scores + mask
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v_t)

    out = out[0].transpose(0, 1).reshape(seq_len, -1)
    attn_out = out @ o_w.T
    return x + attn_out, kv_cache

def forward_ffn_gemma2(h: torch.Tensor, layer_idx: int,
                       weights: dict[str, torch.Tensor],
                       prefix: str = "model") -> torch.Tensor:
    p = f"{prefix}.layers.{layer_idx}"
    norm_w = weights[f"{p}.post_attention_layernorm.weight"]
    h_normed = rms_norm(h, norm_w)
    gate_w = weights[f"{p}.mlp.gate_proj.weight"]
    up_w = weights[f"{p}.mlp.up_proj.weight"]
    down_w = weights[f"{p}.mlp.down_proj.weight"]
    gate = torch.nn.functional.gelu(h_normed @ gate_w.T, approximate='tanh')
    up = h_normed @ up_w.T
    return h + (gate * up) @ down_w.T

def forward_transformer_gemma2(model, token_ids: np.ndarray,
                                kv_caches: list | None = None) -> np.ndarray:
    dev = _get_device()
    if not hasattr(model, '_gpu_weights') or model._gpu_weights is None:
        w_gpu = {}
        for k, v in model.weights.items():
            w_gpu[k] = torch.as_tensor(v, device=dev)
        model._gpu_weights = w_gpu
    else:
        w_gpu = model._gpu_weights

    cfg = model.config
    n_layers = model.n_layers
    prefix = "model"

    embed_w = w_gpu[f"{prefix}.embed_tokens.weight"]
    ids_t = torch.as_tensor(token_ids, dtype=torch.long, device=dev)
    x = embed_w[ids_t] * np.sqrt(cfg.get("head_dim", 256))

    if kv_caches is not None and len(kv_caches) == 0:
        for _ in range(n_layers):
            kv_caches.append([])

    seq_len = x.shape[0]
    if kv_caches is not None and len(kv_caches) > 0 and len(kv_caches[0]) >= 2:
        cached_len = kv_caches[0][0].shape[1]
        start_pos = cached_len
    else:
        start_pos = 0
    positions = torch.arange(start_pos, start_pos + seq_len, dtype=torch.float32, device=dev)

    for i in range(n_layers):
        layer_kv = kv_caches[i] if kv_caches is not None else None
        x, updated_kv = forward_attention_layer_gemma2(x, i, w_gpu, cfg, positions, layer_kv, prefix)
        if kv_caches is not None:
            kv_caches[i] = updated_kv
        x = forward_ffn_gemma2(x, i, w_gpu, prefix)

    norm_w = w_gpu[f"{prefix}.norm.weight"]
    x = rms_norm(x, norm_w)

    lm_head_w = w_gpu.get("lm_head.weight", embed_w)
    logits = x @ lm_head_w.T

    softcap = cfg.get("final_logit_softcapping", None)
    if softcap is not None and softcap > 0:
        logits = softcap * torch.tanh(logits / softcap)

    return logits.cpu().numpy()

def forward_transformer_mixtral(model, token_ids: np.ndarray,
                                 kv_caches: list | None = None) -> np.ndarray:
    dev = _get_device()
    if not hasattr(model, '_gpu_weights') or model._gpu_weights is None:
        w_gpu = {}
        for k, v in model.weights.items():
            w_gpu[k] = torch.as_tensor(v, device=dev)
        model._gpu_weights = w_gpu
    else:
        w_gpu = model._gpu_weights

    cfg = model.config
    n_layers = model.n_layers
    n_experts = cfg.get("num_local_experts", 8)
    top_k = cfg.get("num_experts_per_tok", 2)
    prefix = "model"

    embed_w = w_gpu[f"{prefix}.embed_tokens.weight"]
    ids_t = torch.as_tensor(token_ids, dtype=torch.long, device=dev)
    x = embed_w[ids_t]

    if kv_caches is not None and len(kv_caches) == 0:
        for _ in range(n_layers):
            kv_caches.append([])

    seq_len = x.shape[0]
    if kv_caches is not None and len(kv_caches) > 0 and len(kv_caches[0]) >= 2:
        cached_len = kv_caches[0][0].shape[1]
        start_pos = cached_len
    else:
        start_pos = 0
    positions = torch.arange(start_pos, start_pos + seq_len, dtype=torch.float32, device=dev)

    for i in range(n_layers):
        layer_kv = kv_caches[i] if kv_caches is not None else None
        x, updated_kv = forward_attention_layer(x, i, w_gpu, cfg, positions, layer_kv, prefix)
        if kv_caches is not None:
            kv_caches[i] = updated_kv

        norm_w = w_gpu[f"{prefix}.layers.{i}.post_attention_layernorm.weight"]
        h = rms_norm(x, norm_w)

        gate_w = w_gpu[f"{prefix}.layers.{i}.block_sparse_moe.gate.weight"]
        router_logits = h @ gate_w.T
        router_probs = torch.softmax(router_logits.float(), dim=-1)
        topk_vals, topk_indices = torch.topk(router_probs, top_k, dim=-1)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

        moe_out = torch.zeros_like(h)
        for tok in range(seq_len):
            for ek in range(top_k):
                expert_idx = int(topk_indices[tok, ek])
                weight = float(topk_vals[tok, ek])
                eg_w = w_gpu[f"{prefix}.layers.{i}.block_sparse_moe.experts.{expert_idx}.w1.weight"]
                eu_w = w_gpu[f"{prefix}.layers.{i}.block_sparse_moe.experts.{expert_idx}.w2.weight"]
                ev_w = w_gpu[f"{prefix}.layers.{i}.block_sparse_moe.experts.{expert_idx}.w3.weight"]
                h_tok = h[tok]
                gate_act = torch.nn.functional.silu(h_tok @ eg_w.T)
                up_act = h_tok @ ev_w.T
                expert_out = (gate_act * up_act) @ eu_w.T
                moe_out[tok] += weight * expert_out

        x = x + moe_out

    norm_w = w_gpu[f"{prefix}.norm.weight"]
    x = rms_norm(x, norm_w)

    lm_head_w = w_gpu.get("lm_head.weight", embed_w)
    logits = x @ lm_head_w.T
    return logits.cpu().numpy()

def speculative_decode(target_model, draft_model, prompt_ids: list[int],
                       max_new_tokens: int = 128, speculative_length: int = 5,
                       temperature: float = 0.6, top_k: int = 50, top_p: float = 0.95,
                       seed: int | None = None) -> GenerationResult:
    import time

    from runtime.ml.generate import IM_END, PAD, GenerationResult, _softmax, sample_token

    rng = np.random.default_rng(seed)
    token_ids = list(prompt_ids)
    all_tokens = []
    t_start = time.time()
    target_kv = []
    draft_kv = []

    stop_ids = {IM_END, PAD}
    if hasattr(target_model, 'tokenizer') and target_model.tokenizer.eos_id is not None:
        stop_ids.add(target_model.tokenizer.eos_id)

    logits = forward(target_model, token_ids, target_kv)
    draft_kv_fresh = []
    _ = forward(draft_model, token_ids, draft_kv_fresh)

    while len(all_tokens) < max_new_tokens:
        draft_tokens = []
        draft_kv_running = [list(cache) if isinstance(cache, list) else cache
                           for cache in draft_kv_fresh]
        current_id = token_ids[-1]

        for _ in range(speculative_length):
            d_logits = forward(draft_model, [current_id], draft_kv_running)
            d_next = sample_token(d_logits[-1], temperature, top_k, top_p,
                                  1.0, 64, token_ids, rng)
            draft_tokens.append(d_next)
            current_id = d_next
            if d_next in stop_ids:
                break

        n_draft = len(draft_tokens)
        if n_draft == 0:
            next_id = sample_token(logits[-1], temperature, top_k, top_p,
                                   1.0, 64, token_ids, rng)
            token_ids.append(next_id)
            all_tokens.append(next_id)
            logits = forward(target_model, [next_id], target_kv)
            if next_id in stop_ids:
                break
            continue

        verify_ids = draft_tokens
        verify_logits = forward(target_model, verify_ids, target_kv)

        n_accepted = 0
        for j in range(n_draft):
            target_prob = _softmax(verify_logits[j].astype(np.float64))
            draft_prob = np.zeros_like(target_prob)
            accepted = False
            if temperature <= 0:
                target_choice = int(np.argmax(target_prob))
                accepted = (target_choice == draft_tokens[j])
            else:
                r = rng.random()
                p_draft = 1.0
                p_target = target_prob[draft_tokens[j]]
                if r < min(1.0, p_target / max(p_draft, 1e-30)):
                    accepted = True

            if accepted:
                n_accepted = j + 1
            else:
                corrected = sample_token(verify_logits[j], temperature, top_k, top_p,
                                         1.0, 64, token_ids, rng)
                draft_tokens[j] = corrected
                n_accepted = j + 1
                break

        accepted_tokens = draft_tokens[:n_accepted]
        token_ids.extend(accepted_tokens)
        all_tokens.extend(accepted_tokens)

        for t_id in accepted_tokens:
            if t_id in stop_ids:
                elapsed = time.time() - t_start
                tps = len(all_tokens) / elapsed if elapsed > 0 else 0.0
                answer_text = target_model.tokenizer.decode(all_tokens, skip_special=True) if hasattr(target_model, 'tokenizer') else ''
                return GenerationResult(text=answer_text, all_tokens=all_tokens,
                                       tokens_per_second=tps,
                                       prompt_tokens=len(prompt_ids),
                                       total_tokens=len(prompt_ids) + len(all_tokens))

        last_logits = verify_logits[n_accepted - 1] if n_accepted > 0 else logits[-1]
        logits = last_logits.reshape(1, -1)
        logits = np.broadcast_to(logits, (1, logits.shape[-1]))

        draft_kv_fresh = draft_kv_running

    elapsed = time.time() - t_start
    tps = len(all_tokens) / elapsed if elapsed > 0 else 0.0
    answer_text = target_model.tokenizer.decode(all_tokens, skip_special=True) if hasattr(target_model, 'tokenizer') else ''
    return GenerationResult(text=answer_text, all_tokens=all_tokens,
                           tokens_per_second=tps,
                           prompt_tokens=len(prompt_ids),
                           total_tokens=len(prompt_ids) + len(all_tokens))
