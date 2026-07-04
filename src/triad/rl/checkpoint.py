"""The ONLY save/load path for weights (CLAUDE.md §7.1).

Release checkpoints are plain dicts of CPU tensors + primitive metadata,
loaded with map_location="cpu" and weights_only=True — never pickled Modules.
Cloud-GPU-trained weights therefore load on any machine. Optimizer state for
resumable training lives in a separate *_trainstate.pt file, never shipped.
"""
from __future__ import annotations

import datetime
import subprocess
from pathlib import Path

import torch

from triad.engine.orders import VOCAB_SIZE
from triad.rl.models import TriadPolicy

OBS_SPEC_VERSION = 1


def _git_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def resolve_device(device: str = "auto") -> torch.device:
    """--device {auto,cpu,cuda,mps}: auto = cuda if available else cpu."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_checkpoint(
    model: TriadPolicy,
    path: str | Path,
    *,
    train_config: dict | None = None,
    seed: int | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": dict(model.config),
        "obs_spec_version": OBS_SPEC_VERSION,
        "order_vocab_size": VOCAB_SIZE,
        "train_config": dict(train_config or {}),
        "seed": seed,
        "git_sha": _git_sha(),
        "torch_version": str(torch.__version__),  # TorchVersion isn't weights_only-safe
        "created": datetime.date.today().isoformat(),
    }
    torch.save(payload, path)
    return path


def load_policy(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[TriadPolicy, dict]:
    """Load on CPU (weights_only=True), rebuild from stored config, move to
    device. Returns (model in eval mode, full payload)."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["order_vocab_size"] == VOCAB_SIZE, "vocab mismatch"
    assert payload["obs_spec_version"] == OBS_SPEC_VERSION, "obs spec mismatch"
    model = TriadPolicy(payload["model_config"])
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    model.to(torch.device(device) if isinstance(device, str) else device)
    return model, payload
