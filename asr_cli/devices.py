import logging

import torch


def detect_mps() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def detect_cuda() -> bool:
    return torch.cuda.is_available()


def resolve_device(requested: str, label: str = "component") -> str:
    """
    Resolve requested device name against what's available.

    Rules:
    - auto: prefer cuda, then mps, else cpu
    - cuda/mps: warn and fall back to cpu if unavailable
    - cpu: always cpu
    """
    req = (requested or "auto").lower()
    has_cuda = detect_cuda()
    has_mps = detect_mps()

    if req == "auto":
        if has_cuda:
            device = "cuda"
        elif has_mps:
            device = "mps"
        else:
            device = "cpu"
        logging.info("%s device auto-resolved to '%s'.", label, device)
        return device

    if req == "cuda":
        if has_cuda:
            return "cuda"
        logging.warning("CUDA requested for %s but not available; falling back to CPU.", label)
        return "cpu"

    if req == "mps":
        if has_mps:
            return "mps"
        logging.warning("MPS requested for %s but not available; falling back to CPU.", label)
        return "cpu"

    # Fallback to CPU for any unknown value (should not happen with CLI choices)
    if req != "cpu":
        logging.warning("Unknown device '%s' for %s; using CPU.", req, label)
    return "cpu"
