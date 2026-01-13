# backend/app/inference.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

TORCH_AVAILABLE = False
torch = None
nn = None

try:
    import torch as _torch
    import torch.nn as _nn
    TORCH_AVAILABLE = True
    torch = _torch
    nn = _nn
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


@dataclass
class ModelBundle:
    version: str
    meta: Dict[str, Any]
    model: Any  # nn.Module | None
    device: str = "cpu"


_GRU_STATE: Dict[str, Any] = {}
ADAPTER_READY = False


def _project_root() -> Path:
    # backend/app/inference.py -> backend/app -> backend -> tutor_system_reference
    return Path(__file__).resolve().parents[2]


def _default_artifact_paths() -> Tuple[str, str]:
    repo_root = _project_root().parents[0]  # tutor_system_reference
    root = repo_root.parent                # project_root
    model_path = root / "topcer_pipeline" / "models" / "state_seq_model.pt"
    meta_path = root / "topcer_pipeline" / "data" / "sequences" / "seq_dataset_meta.json"
    return str(model_path), str(meta_path)


def _load_meta(meta_path: str) -> Dict[str, Any]:
    if meta_path and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"feature_names": []}


def normalize_features(meta: Dict[str, Any], features: Dict[str, float]) -> Dict[str, float]:
    fns = meta.get("feature_names", []) or []
    return {fn: float(features.get(fn, 0.0)) for fn in fns}


def _sigmoid(z: float) -> float:
    import math
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _state_key(student_id: str, session_id: Optional[str]) -> str:
    return f"{session_id or 'default'}::{student_id}"


def reset_state(student_id: str, session_id: Optional[str] = None) -> None:
    _GRU_STATE.pop(_state_key(student_id, session_id), None)


def resolve_kc_index(meta: Dict[str, Any], kc_id_raw: Any) -> int:
    """
    Converts kc_id (e.g., "KC_12") into integer embedding index.
    Priority:
      1) meta vocab map if exists
      2) numeric string -> int
      3) stable hash fallback bounded by kc_vocab_size
    """
    # 1) vocab mapping
    for key in ("kc_to_idx", "kc_vocab", "kc_vocab_map", "kc_index"):
        m = meta.get(key)
        if isinstance(m, dict) and str(kc_id_raw) in m:
            try:
                return int(m[str(kc_id_raw)])
            except Exception:
                pass

    # 2) numeric string
    s = str(kc_id_raw)
    if s.isdigit():
        return int(s)

    # 3) hash fallback bounded
    vocab_size = None
    cfg = meta.get("config")
    if isinstance(cfg, dict) and "kc_vocab_size" in cfg:
        try:
            vocab_size = int(cfg["kc_vocab_size"])
        except Exception:
            vocab_size = None

    base = vocab_size if (vocab_size and vocab_size > 0) else 10000
    return abs(hash(s)) % base


# Define model only if torch available
if TORCH_AVAILABLE:
    class GRUMultiTask(nn.Module):
        def __init__(
            self,
            input_dim: int,
            kc_vocab_size: int,
            kc_emb_dim: int,
            rnn_hidden: int,
            rnn_layers: int,
            dropout: float,
            num_states: int,
        ):
            super().__init__()
            self.kc_emb = nn.Embedding(kc_vocab_size, kc_emb_dim)
            self.gru = nn.GRU(
                input_size=input_dim + kc_emb_dim,
                hidden_size=rnn_hidden,
                num_layers=rnn_layers,
                batch_first=True,
                dropout=dropout if rnn_layers > 1 else 0.0,
            )
            self.head_correct = nn.Linear(rnn_hidden, 1)

            # These heads exist in ckpt but are NOT used by infer_step.
            # Keep them with safe shapes; we will drop incompatible weights during load if needed.
            self.head_hint = nn.Linear(rnn_hidden, 1)
            self.head_time = nn.Linear(rnn_hidden, 1)   # may mismatch; handled during load
            self.head_state = nn.Linear(rnn_hidden, num_states)

        def forward(self, x_feat: torch.Tensor, kc_ids: torch.Tensor, h: Optional[torch.Tensor] = None):
            e = self.kc_emb(kc_ids)
            x = torch.cat([x_feat, e], dim=-1)
            out, h_next = self.gru(x, h)
            logits_correct = self.head_correct(out)
            return logits_correct, h_next


def load_model_bundle() -> ModelBundle:
    default_model_path, default_meta_path = _default_artifact_paths()
    model_path = os.getenv("TOPCER_MODEL_PATH", default_model_path)
    meta_path = os.getenv("TOPCER_META_PATH", default_meta_path)
    version = os.getenv("TOPCER_MODEL_VERSION", "topcer-gru-multitask:v1")
    device = os.getenv("TOPCER_DEVICE", "cpu")

    meta = _load_meta(meta_path)

    if not TORCH_AVAILABLE:
        print("[MODEL] torch not available -> heuristic mode")
        return ModelBundle(version=version, meta=meta, model=None, device="cpu")

    if (not model_path) or (not os.path.exists(model_path)):
        print(f"[MODEL] not found: {model_path} -> heuristic mode")
        return ModelBundle(version=version, meta=meta, model=None, device="cpu")

    ckpt = torch.load(model_path, map_location=device)

    if (not meta.get("feature_names")) and isinstance(ckpt, dict) and ("meta" in ckpt):
        meta = ckpt["meta"]

    if not isinstance(ckpt, dict) or "model_state" not in ckpt or "config" not in ckpt:
        print("[MODEL] unexpected ckpt format -> heuristic mode")
        return ModelBundle(version=version, meta=meta, model=None, device="cpu")

    cfg = ckpt["config"]

    # Put kc_vocab_size into meta config to support hash fallback
    meta.setdefault("config", {})
    if isinstance(meta["config"], dict):
        meta["config"]["kc_vocab_size"] = int(cfg.get("kc_vocab_size", 10000))

    model = GRUMultiTask(
        input_dim=int(cfg["input_dim"]),
        kc_vocab_size=int(cfg["kc_vocab_size"]),
        kc_emb_dim=int(cfg["kc_emb_dim"]),
        rnn_hidden=int(cfg["rnn_hidden"]),
        rnn_layers=int(cfg["rnn_layers"]),
        dropout=float(cfg["dropout"]),
        num_states=int(cfg["num_states"]),
    )

    # Drop incompatible head weights (we don't need them for infer_step)
    state = dict(ckpt["model_state"])
    for k in list(state.keys()):
        if k.startswith("head_time."):
            state.pop(k, None)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[MODEL] missing keys:", missing)
    if unexpected:
        print("[MODEL] unexpected keys:", unexpected)

    model.eval()
    model.to(device)

    global ADAPTER_READY
    ADAPTER_READY = True
    print("[MODEL] loaded OK -> gru_wired")

    return ModelBundle(version=version, meta=meta, model=model, device=device)


def _no_grad(fn):
    if TORCH_AVAILABLE:
        return torch.no_grad()(fn)
    return fn


@_no_grad
def infer_step(
    bundle: ModelBundle,
    features: Dict[str, float],
    *,
    student_id: Optional[str] = None,
    session_id: Optional[str] = None,
    kc_id: Optional[int] = None,
) -> Tuple[float, Optional[List[float]], List[str]]:
    warnings: List[str] = []
    f = normalize_features(bundle.meta, features)

    # heuristic fallback
    if bundle.model is None:
        warnings.append("MODEL_HEURISTIC_MODE")
        mastery = float(f.get("mastery_mean", 0.5))
        incorrects = float(f.get("incorrects", 0.0))
        duration = float(f.get("duration", 20.0))
        hints = float(f.get("hints", 0.0))
        rapid_wrong = float(f.get("rapid_wrong", 0.0))

        z = 2.5 * (mastery - 0.5) - 0.9 * incorrects - 0.03 * (duration - 20.0) - 0.2 * hints - 0.6 * rapid_wrong
        return _sigmoid(z), None, warnings

    # GRU forward
    if kc_id is None:
        kc_id = int(features.get("kc_id", 0))
        warnings.append("KC_ID_MISSING_DEFAULT_0")

    fns = bundle.meta.get("feature_names", []) or []
    if not fns:
        warnings.append("META_FEATURE_NAMES_EMPTY")
        x_feat = torch.zeros((1, 1, 11), dtype=torch.float32, device=bundle.device)
    else:
        x_feat = torch.tensor([f[name] for name in fns], dtype=torch.float32, device=bundle.device).view(1, 1, -1)

    kc = torch.tensor([[int(kc_id)]], dtype=torch.long, device=bundle.device)

    h_prev = None
    if student_id is not None:
        key = _state_key(student_id, session_id)
        if key in _GRU_STATE:
            h_prev = _GRU_STATE[key].to(bundle.device)

    logits, h_next = bundle.model(x_feat, kc, h_prev)
    p = float(torch.sigmoid(logits[0, -1, 0]).item())

    if student_id is not None and h_next is not None:
        _GRU_STATE[_state_key(student_id, session_id)] = h_next.detach().to("cpu")

    return p, None, warnings
