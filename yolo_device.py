"""Runtime device selection for Ultralytics inference.

Kept separate from ``yolo_detector`` so that the camera loop, the training
scripts and the health check can all ask the same question without importing
Ultralytics. Every entry point in this project fails back to CPU when torch is
missing or CUDA is unavailable; it never raises.
"""
import config

_STATE = None  # cached so the torch import happens once


def describe_device():
    """Return a dict describing what would be used for inference right now."""
    global _STATE
    if _STATE is not None:
        return dict(_STATE)

    requested = config.YOLO_DEVICE
    if config.YOLO_FORCE_CPU:
        requested = "cpu"

    info = {
        "requested": requested,
        "torch_version": None,
        "cuda_available": False,
        "device": "cpu",
        "reason": "",
        "cuda_hint": "",
    }

    if isinstance(requested, str) and requested.strip().lower() == "cpu":
        info["reason"] = "config requests CPU"
        _STATE = info
        return dict(info)
    if isinstance(requested, str) and requested.strip().lower() != "auto":
        info["device"] = requested.strip()
        info["reason"] = "config pins a device explicitly; CUDA not verified"
        _STATE = info
        return dict(info)

    try:
        import torch
    except Exception as exc:
        info["reason"] = f"torch not importable ({type(exc).__name__})"
        _STATE = info
        return dict(info)

    info["torch_version"] = getattr(torch, "__version__", "unknown")
    probe_failed = False
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        cuda_available = False
        probe_failed = True
        info["reason"] = f"torch.cuda.is_available() failed: {exc}"
    info["cuda_available"] = cuda_available

    if cuda_available:
        try:
            info["device"] = "0"
            info["reason"] = f"CUDA available on {torch.cuda.get_device_name(0)}"
        except Exception:
            info["device"] = "0"
            info["reason"] = "CUDA available; using device 0"
    elif not probe_failed:
        # No CUDA and the probe worked: name the specific reason for the CPU fallback.
        info["reason"] = "torch.cuda.is_available() is False"
        if "+cpu" in str(info["torch_version"]):
            info["reason"] = (
                f"torch {info['torch_version']} is the CPU-only build; "
                "install a CUDA build to use the GPU"
            )
    _STATE = info
    return dict(info)


def resolve_device():
    """Return the value to hand to Ultralytics ``predict(device=...)``."""
    return describe_device()["device"]


def cuda_install_hint():
    """Exact command for the user to install a CUDA build of torch. Nothing is installed here."""
    try:
        import torch
        version = getattr(torch, "__version__", "unknown")
    except Exception:
        version = "unknown"
    return (
        "Current torch: "
        f"{version} (torch.cuda.is_available() = False).\n"
        "To use an NVIDIA GPU, install the matching CUDA wheel yourself:\n"
        "    python -m pip uninstall -y torch torchvision\n"
        "    python -m pip install torch torchvision --index-url "
        "https://download.pytorch.org/whl/cu124\n"
        "Then run: python health_check.py   (it reports the device that will be used)"
    )