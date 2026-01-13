# backend/app/inference.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


# -----------------------------
# Model Definition (matches ckpt)
# -----------------------------
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

        # Heads present in ckpt; not required for /infer_step right now
        self.head_hint = nn.Linear(rnn_hidden, 1)
        self.head_time = nn.Linear(rnn_hidden, 1)
        self.head_state = nn.Linear(rnn_hidden, num_states)

    def forward(self, x_feat: torch.Tensor, kc_ids: torch.Tensor, h: Optional[torch.Tensor] = None):
        """
        x_feat: (B,T,F=input_dim)
        kc_ids: (B,T)
        h: (num_layers,B,H) or None
        """
        e = self.kc_emb(kc_ids)             # (B,T,E)
        x = torch.cat([x_feat, e], dim=-1)  # (B,T,F+E)
        out, h_next = self.gru(x, h)        # out: (B,T,H)
        logits_correct = self.head_correct(out)  # (B,T,1)
        return logits_correct, h_next


# -----------------------------
# Bundle & State
# -----------------------------
@dataclass
class ModelBundle:
    version: str
    meta: Dict[str, Any]
    model: Any  # nn.Module | None
    device: str = "cpu"


# Hidden state per session+student (simple in-memory)
_GRU_STATE: Dict[str, Any] = {}
ADAPTER_READY = False


def _project_root() -> Path:
    # backend/app/inference.py -> backend/app -> backend -> tutor_system_reference
    return Path(__file__).resolve().parents[2]


def _default_artifact_paths() -> Tuple[str, str]:
    # Default assumes repo layout:
    # project_root/
    #   tutor_system_reference/backend/app/...
    #   topcer_pipeline/models/state_seq_model.pt
    #   topcer_pipeline/data/sequences/seq_dataset_meta.json
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


def load_model_bundle() -> ModelBundle:
    """
    Loads ckpt dict with keys: model_state, config, meta (as per your uploaded .pt).
    Env overrides:
      TOPCER_MODEL_PATH, TOPCER_META_PATH, TOPCER_MODEL_VERSION, TOPCER_DEVICE
    """
    default_model_path, default_meta_path = _default_artifact_paths()

    model_path = os.getenv("TOPCER_MODEL_PATH", default_model_path)
    meta_path = os.getenv("TOPCER_META_PATH", default_meta_path)
    version = os.getenv("TOPCER_MODEL_VERSION", "topcer-gru-multitask:v1")
    device = os.getenv("TOPCER_DEVICE", "cpu")

    meta = _load_meta(meta_path)

    if (not TORCH_AVAILABLE) or (not model_path) or (not os.path.exists(model_path)):
        # model not available; heuristic mode
        return ModelBundle(version=version, meta=meta, model=None, device="cpu")

    ckpt = torch.load(model_path, map_location=device)

    # If meta file missing, fallback to ckpt meta
    if (not meta.get("feature_names")) and isinstance(ckpt, dict) and ("meta" in ckpt):
        meta = ckpt["meta"]

    if not isinstance(ckpt, dict) or "model_state" not in ckpt or "config" not in ckpt:
        # unexpected format
        return ModelBundle(version=version, meta=meta, model=None, device="cpu")

    cfg = ckpt["config"]
    model = GRUMultiTask(
        input_dim=int(cfg["input_dim"]),
        kc_vocab_size=int(cfg["kc_vocab_size"]),
        kc_emb_dim=int(cfg["kc_emb_dim"]),
        rnn_hidden=int(cfg["rnn_hidden"]),
        rnn_layers=int(cfg["rnn_layers"]),
        dropout=float(cfg["dropout"]),
        num_states=int(cfg["num_states"]),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    model.to(device)

    global ADAPTER_READY
    ADAPTER_READY = True

    return ModelBundle(version=version, meta=meta, model=model, device=device)


def normalize_features(meta: Dict[str, Any], features: Dict[str, float]) -> Dict[str, float]:
    """
    Ensures all feature_names exist; missing -> 0.0
    """
    fns = meta.get("feature_names", []) or []
    out = {fn: float(features.get(fn, 0.0)) for fn in fns}
    return out


def _sigmoid(z: float) -> float:
    import math
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def _state_key(student_id: str, session_id: Optional[str]) -> str:
    return f"{session_id or 'default'}::{student_id}"


def reset_state(student_id: str, session_id: Optional[str] = None) -> None:
    _GRU_STATE.pop(_state_key(student_id, session_id), None)


@torch.no_grad() if TORCH_AVAILABLE else (lambda f: f)
def infer_step(
    bundle: ModelBundle,
    features: Dict[str, float],
    *,
    student_id: Optional[str] = None,
    session_id: Optional[str] = None,
    kc_id: Optional[int] = None,
) -> Tuple[float, Optional[List[float]], List[str]]:
    """
    Returns: (p_correct, latent(optional), warnings)
    latent can be None for now (or you can expose h_t later).
    """
    warnings: List[str] = []

    # normalize to meta schema
    f = normalize_features(bundle.meta, features)

    # Heuristic fallback if model missing
    if bundle.model is None:
        warnings.append("MODEL_NOT_WIRED: using heuristic predictor.")

        mastery = float(f.get("mastery_mean", 0.5))
        incorrects = float(f.get("incorrects", 0.0))
        duration = float(f.get("duration", 20.0))
        hints = float(f.get("hints", 0.0))
        rapid_wrong = float(f.get("rapid_wrong", 0.0))

        z = 2.5 * (mastery - 0.5) - 0.9 * incorrects - 0.03 * (duration - 20.0) - 0.2 * hints - 0.6 * rapid_wrong
        p = _sigmoid(z)
        return p, None, warnings

    # GRU forward
    if kc_id is None:
        # default 0 if missing; better: require kc_id in request
        kc_id = int(features.get("kc_id", 0))
        warnings.append("KC_ID_MISSING: defaulting kc_id=0")

    # Build tensors
    fns = bundle.meta.get("feature_names", []) or []
    if not fns:
        warnings.append("META_FEATURE_NAMES_EMPTY: using zeros")
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
        # store on CPU to reduce memory pressure
        _GRU_STATE[_state_key(student_id, session_id)] = h_next.detach().to("cpu")

    return p, None, warnings
