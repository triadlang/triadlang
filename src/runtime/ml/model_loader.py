from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from triad import ntri as np

if TYPE_CHECKING:
    import torch


class TriadTokenizer:
    def __init__(self, tokenizer):
        self._tok = tokenizer
        self.vocab_size = len(tokenizer)

    @property
    def eos_id(self) -> int:
        eos = getattr(self._tok, "eos_token_id", None)
        return eos if eos is not None else 0

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        return self._tok.encode(text, add_special_tokens=add_special)

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special)

    def apply_chat_template(self, messages: list[dict], add_generation_prompt: bool = True,
                            think: bool = False) -> list[int]:

        result = self._tok.apply_chat_template(messages, tokenize=True,
                                                 add_generation_prompt=add_generation_prompt,
                                                 return_dict=False, think=think,
                                                 enable_thinking=think)
        if hasattr(result, 'input_ids'):
            return result['input_ids']
        if hasattr(result, '__getitem__') and len(result) > 0 and hasattr(result[0], 'ids'):
            return result[0].ids
        return list(result)

    def __repr__(self):
        return f"TriadTokenizer(vocab_size={self.vocab_size})"

class TriadModel:
    def __init__(self, config: dict, weights: dict[str, np.ndarray],
                 tokenizer: TriadTokenizer | None = None):
        self.config = config
        self.weights = weights
        self.tokenizer = tokenizer
        self._arch = _detect_arch(config)
        self._gpu_weights = None

    @property
    def arch(self) -> str:
        return self._arch

    @property
    def n_layers(self) -> int:
        return self.config.get("num_hidden_layers",
                               self.config.get("n_layer", 0))

    @property
    def d_model(self) -> int:
        return self.config.get("hidden_size",
                               self.config.get("d_model", 0))

    @property
    def n_heads(self) -> int:
        return self.config.get("num_attention_heads",
                               self.config.get("n_head", 0))

    @property
    def n_kv_heads(self) -> int:
        return self.config.get("num_key_value_heads", self.n_heads)

    @property
    def vocab_size(self) -> int:
        return self.config.get("vocab_size", 0)

    @property
    def max_pos(self) -> int:
        return self.config.get("max_position_embeddings", 2048)

    @property
    def rope_base(self) -> float:
        return self.config.get("rope_theta", 10000.0)

    def w(self, name: str) -> np.ndarray:
        return self.weights[name]

    def has(self, name: str) -> bool:
        return name in self.weights

    def __repr__(self):
        nw = len(self.weights)
        total = sum(w.nbytes for w in self.weights.values())
        return (f"TriadModel(arch={self._arch}, layers={self.n_layers}, "
                f"d_model={self.d_model}, vocab={self.vocab_size}, "
                f"weights={nw}, size={total/1e9:.2f}GB)")

class TriadModelGGUF(TriadModel):
    def __init__(self, config: dict, gguf_reader, tokenizer: TriadTokenizer | None = None):
        super().__init__(config, {}, tokenizer)
        self._reader = gguf_reader
        self._tensor_map = {}
        self._gguf_config(gguf_reader, config)
        for t in gguf_reader.tensors:
            self._tensor_map[t.name] = t

    @staticmethod
    def _gguf_config(reader, config):
        if "num_hidden_layers" in config:
            return
        n_layers = 0
        for t in reader.tensors:
            parts = t.name.split(".")
            if len(parts) >= 2 and parts[0] == "blk":
                try:
                    idx = int(parts[1])
                    n_layers = max(n_layers, idx + 1)
                except ValueError:
                    pass
        config["num_hidden_layers"] = n_layers

    def _to_gpu(self, name: str) -> torch.Tensor:
        import torch
        cache = getattr(self, '_gpu_cache', None)
        if cache is None:
            cache = {}
            self._gpu_cache = cache
        if name in cache:
            return cache[name]
        t = self._tensor_map.get(name)
        if t is None:
            raise KeyError(f"tensor not found in GGUF: {name}")
        arr = t.data.astype("float32")
        out = torch.from_numpy(arr).half().to(torch.device("cuda"))
        cache[name] = out
        return out

    def get_layer_weights(self, layer_idx: int) -> dict[str, torch.Tensor]:
        layer_types = self.config.get("layer_types", [])
        is_triad = layer_types[layer_idx] == "triad_attention" if layer_idx < len(layer_types) else False
        w = {}
        prefix = f"blk.{layer_idx}"
        if is_triad:
            w["input_layernorm.weight"] = self._to_gpu(f"{prefix}.attn_norm.weight")
            w["in_proj_qkv.weight"] = self._to_gpu(f"{prefix}.attn_qkv.weight")
            w["in_proj_z.weight"] = self._to_gpu(f"{prefix}.attn_gate.weight")
            w["in_proj_b.weight"] = self._to_gpu(f"{prefix}.ssm_beta.weight")
            w["in_proj_a.weight"] = self._to_gpu(f"{prefix}.ssm_alpha.weight")
            w["conv1d.weight"] = self._to_gpu(f"{prefix}.ssm_conv1d.weight")
            w["A_log"] = self._to_gpu(f"{prefix}.ssm_a")
            w["dt_bias"] = self._to_gpu(f"{prefix}.ssm_dt.bias")
            w["norm.weight"] = self._to_gpu(f"{prefix}.ssm_norm.weight")
            w["out_proj.weight"] = self._to_gpu(f"{prefix}.ssm_out.weight")
        else:
            w["input_layernorm.weight"] = self._to_gpu(f"{prefix}.attn_norm.weight")
            w["q_proj.weight"] = self._to_gpu(f"{prefix}.attn_q.weight")
            w["k_proj.weight"] = self._to_gpu(f"{prefix}.attn_k.weight")
            w["v_proj.weight"] = self._to_gpu(f"{prefix}.attn_v.weight")
            w["o_proj.weight"] = self._to_gpu(f"{prefix}.attn_output.weight")
            w["q_norm.weight"] = self._to_gpu(f"{prefix}.attn_q_norm.weight")
            w["k_norm.weight"] = self._to_gpu(f"{prefix}.attn_k_norm.weight")
        w["post_attention_layernorm.weight"] = self._to_gpu(f"{prefix}.post_attention_norm.weight")
        w["gate_proj.weight"] = self._to_gpu(f"{prefix}.ffn_gate.weight")
        w["up_proj.weight"] = self._to_gpu(f"{prefix}.ffn_up.weight")
        w["down_proj.weight"] = self._to_gpu(f"{prefix}.ffn_down.weight")
        return w

    def get_embed_weight(self) -> torch.Tensor:
        return self._to_gpu("token_embd.weight")

    def get_norm_weight(self) -> torch.Tensor:
        return self._to_gpu("output_norm.weight")

    def get_lm_head_weight(self) -> torch.Tensor:
        if "output.weight" in self._tensor_map:
            return self._to_gpu("output.weight")
        return self.get_embed_weight()

    @property
    def weights(self):
        return {t.name: t.data for t in self._tensor_map.values()}

    def has(self, name: str) -> bool:
        return name in self._tensor_map

    def __repr__(self):
        return (f"TriadModelGGUF(arch={self._arch}, layers={self.n_layers}, "
                f"d_model={self.d_model}, vocab={self.vocab_size}, "
                f"streaming GPU forward)")

def _detect_arch(config: dict) -> str:
    mt = config.get("model_type", "").lower()
    if "qwen3_5" in mt or "qwen3.5" in mt:
        return "qwen3_5"
    if "moe" in mt:
        if "qwen3" in mt:
            return "qwen3moe"
        if "qwen2" in mt:
            return "qwen2moe"
        return "unknown"
    if "qwen3" in mt:
        return "qwen3"
    if "qwen2" in mt:
        return "qwen2"
    if "llama" in mt:
        return "llama"
    if "mistral" in mt:
        return "mistral"
    if "gemma" in mt:
        return "gemma"
    if "falcon" in mt:
        return "falcon"
    if "gpt2" in mt:
        return "gpt2"
    if "mamba" in mt:
        return "mamba"
    return mt or "unknown"

def _hf_dir(name_or_path: str) -> str:
    if os.path.isdir(name_or_path):
        return name_or_path
    try:
        from huggingface_hub import snapshot_download

        from runtime.security import get_policy
        policy = get_policy()
        if policy.safe and not policy.capabilities.ml_download_remote:
            raise PermissionError('remote model download disabled by policy')
        return snapshot_download(name_or_path)
    except (ImportError, OSError, ValueError):
        raise FileNotFoundError(f"cannot find model: {name_or_path}")

def _load_config(model_dir: str) -> dict:
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"no config.json in {model_dir}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    if "text_config" in cfg:
        tc = cfg.pop("text_config")
        tc["model_type"] = cfg.get("model_type", tc.get("model_type", ""))
        for k in ("architectures", "tie_word_embeddings"):
            if k in cfg and k not in tc:
                tc[k] = cfg[k]
        cfg = tc
    return cfg

def _load_safetensors(model_dir: str, dtype=np.float32,
                      prefix_filter: str | None = None) -> dict[str, np.ndarray]:
    from safetensors import safe_open
    weights = {}
    has_torch = False
    try:
        import torch
        has_torch = True
    except ImportError:
        pass
    for fn in sorted(os.listdir(model_dir)):
        if fn.endswith(".safetensors"):
            fp = os.path.join(model_dir, fn)
            if has_torch:
                with safe_open(fp, framework="pt") as sf:
                    for key in sf.keys():
                        if prefix_filter and not any(key.startswith(p) for p in prefix_filter.split(",")):
                            continue
                        arr = sf.get_tensor(key).cpu().float().numpy()
                        if arr.dtype != dtype:
                            arr = arr.astype(dtype)
                        weights[key] = arr
            else:
                with safe_open(fp, framework="numpy") as sf:
                    for key in sf.keys():
                        if prefix_filter and not any(key.startswith(p) for p in prefix_filter.split(",")):
                            continue
                        arr = sf.get_tensor(key)
                        if arr.dtype != dtype:
                            arr = arr.astype(dtype)
                        weights[key] = arr
    if not weights:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")
    return weights

def _load_gguf_weights(gguf_path: str) -> tuple[dict, dict[str, np.ndarray]]:
    import gguf
    reader = gguf.GGUFReader(gguf_path)
    config = {}
    weights = {}
    for field in reader.fields.values():
        if field.name in ("general.architecture",):
            config["model_type"] = _gguf_str(field)
    for field in reader.fields.values():
        if hasattr(field, 'types') and str(field.types) == 'GGUFValueType.STRING':
            try:
                config[field.name] = _gguf_str(field)
            except Exception as e:
                import warnings
                warnings.warn(f'GGUF string field {field.name}: {type(e).__name__}')
        elif hasattr(field, 'types'):
            try:
                val = field.parts[field.data[0]]
                if hasattr(val, 'item'):
                    val = val.item()
                config[field.name] = val
            except Exception as e:
                import warnings
                warnings.warn(f'GGUF field {field.name}: {type(e).__name__}')
    from runtime.ml.gguf import _HF_BLK, _HF_MAP
    top_map = dict(_HF_MAP)
    blk_map = dict(_HF_BLK)

    def _to_hf(name: str) -> str:

        if name in top_map:
            return top_map[name]
        if name.startswith('blk.'):
            _, idx, rest = name.split('.', 2)
            if rest in blk_map:
                return f'model.layers.{idx}.{blk_map[rest]}'
        return name

    from gguf import quants as _gq
    from gguf.constants import GGMLQuantizationType as _GQT
    for tensor in reader.tensors:
        arr = tensor.data
        ttype = tensor.tensor_type
        if ttype not in (_GQT.F32, _GQT.F16, _GQT.F64):

            arr = _gq.dequantize(arr, ttype)
        elif hasattr(arr, 'copy'):
            arr = arr.copy()
        if arr.dtype != np.float32 and arr.dtype != np.float16:
            arr = arr.astype(np.float32)
        weights[_to_hf(tensor.name)] = arr
    return config, weights

def load_tokenizer(name_or_path: str, gguf_file: str | None = None) -> TriadTokenizer:
    from transformers import AutoTokenizer
    if gguf_file:
        tok = AutoTokenizer.from_pretrained(name_or_path, gguf_file=gguf_file)
    else:
        tok = AutoTokenizer.from_pretrained(name_or_path)
    return TriadTokenizer(tok)

def load_model(name_or_path: str,
               gguf_file: str | None = None,
               dtype: str = "float32") -> TriadModel:

    if gguf_file:
        import gguf
        if os.path.isfile(gguf_file):
            gguf_path = gguf_file
        else:
            model_dir = _hf_dir(name_or_path)
            gguf_path = os.path.join(model_dir, gguf_file)
        if not os.path.isfile(gguf_path):
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")
        reader = gguf.GGUFReader(gguf_path)
        config = _gguf_extract_config(reader)
        try:
            model_dir = _hf_dir(name_or_path)
        except (FileNotFoundError, OSError):
            model_dir = ""
        cfg_file = os.path.join(model_dir, "config.json") if model_dir else ""
        if cfg_file and os.path.isfile(cfg_file):
            with open(cfg_file) as f:
                hf_config = json.load(f)
            if "text_config" in hf_config:
                tc = hf_config["text_config"]
                for k, v in tc.items():
                    if k not in config:
                        config[k] = v
                if "model_type" not in config:
                    config["model_type"] = hf_config.get("model_type", "")
            else:
                for k, v in hf_config.items():
                    if k not in config:
                        config[k] = v
        try:
            tok = load_tokenizer(name_or_path)
        except (OSError, FileNotFoundError):
            tok = load_tokenizer(name_or_path, gguf_file=os.path.basename(gguf_path))
        arch = _detect_arch(config)
        if arch == "qwen3_5":
            return TriadModelGGUF(config, reader, tok)
        config_final, weights = _load_gguf_weights(gguf_path)
        for k, v in config.items():
            if k not in config_final:
                config_final[k] = v
        return TriadModel(config_final, weights, tok)

    model_dir = _hf_dir(name_or_path)
    config = _load_config(model_dir)
    pf = None
    if _detect_arch(config) == "qwen3_5":
        pf = "model.language_model,lm_head"
    weights = _load_safetensors(model_dir, prefix_filter=pf)
    tok = load_tokenizer(name_or_path)
    np_dtype = np.float16 if dtype == "float16" else np.float32
    if np_dtype == np.float32:
        converted = {}
        for k, v in weights.items():
            if v.dtype != np.float32:
                converted[k] = v.astype(np.float32)
            else:
                converted[k] = v
        weights = converted
    return TriadModel(config, weights, tok)

def _gguf_str(field) -> str:
    raw = field.parts[field.data[0]]
    try:
        return bytes(raw).decode('utf-8')
    except (TypeError, ValueError, UnicodeDecodeError):
        return str(raw)

def _gguf_extract_config(reader) -> dict:
    config = {}
    for field in reader.fields.values():
        if field.name in ("general.architecture",):
            config["model_type"] = _gguf_str(field)
    for field in reader.fields.values():
        if hasattr(field, 'types') and str(field.types) == 'GGUFValueType.STRING':
            try:
                config[field.name] = _gguf_str(field)
            except Exception as e:
                import warnings
                warnings.warn(f'GGUF string field {field.name}: {type(e).__name__}')
        elif hasattr(field, 'types'):
            try:
                val = field.parts[field.data[0]]
                if hasattr(val, 'item'):
                    val = val.item()
                config[field.name] = val
            except Exception as e:
                import warnings
                warnings.warn(f'GGUF field {field.name}: {type(e).__name__}')
    n_layers = 0
    for t in reader.tensors:
        parts = t.name.split(".")
        if len(parts) >= 2 and parts[0] == "blk":
            try:
                n_layers = max(n_layers, int(parts[1]) + 1)
            except ValueError:
                pass
    config["num_hidden_layers"] = n_layers

    a = config.get("model_type", "")
    def _kv(suffix, default=None):
        return config.get(f"{a}.{suffix}", default)
    n_heads = _kv("attention.head_count")
    hidden = _kv("embedding_length")
    derived = {
        "num_attention_heads": n_heads,
        "num_key_value_heads": _kv("attention.head_count_kv", n_heads),
        "hidden_size": hidden,
        "head_dim": _kv("attention.key_length",
                        (hidden // n_heads) if (hidden and n_heads) else None),
        "rms_norm_eps": _kv("attention.layer_norm_rms_epsilon"),
        "rope_theta": _kv("rope.freq_base"),
        "intermediate_size": _kv("feed_forward_length"),
        "vocab_size": _kv("vocab_size"),
        "max_position_embeddings": _kv("context_length"),
    }
    for k, v in derived.items():
        if v is not None and config.get(k) is None:
            config[k] = v
    if config.get("tie_word_embeddings") is None:
        tensor_names = {t.name for t in reader.tensors}
        config["tie_word_embeddings"] = "output.weight" not in tensor_names
    return config
