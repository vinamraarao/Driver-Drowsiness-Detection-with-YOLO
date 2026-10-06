"""Tests for CUDA detection and the auto/cpu/pinned device policy.

No pytest fixtures are used: config values are saved and restored with try/finally
so the file also runs under a plain-python harness.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import yolo_device


class _FakeCuda:
    def __init__(self, available, broken=False):
        self._available = available
        self._broken = broken

    def is_available(self):
        if self._broken:
            raise RuntimeError("driver mismatch")
        return self._available

    def get_device_name(self, idx):
        return "FakeGPU"


class _FakeTorch:
    """Real torch module: __getattr__ delegates to a per-instance stub."""

    def __init__(self, stub):
        object.__setattr__(self, "_stub", stub)

    def __getattr__(self, name):
        # Only reached for attributes not found normally (e.g. __version__).
        return getattr(object.__getattribute__(self, "_stub"), name)


class _BrokenCuda:
    """Mismatched driver / broken install: importing torch.cuda succeeds, is_available() raises."""

    @staticmethod
    def is_available():
        raise RuntimeError("CUDA driver mismatch")

    @staticmethod
    def get_device_name(idx):
        raise RuntimeError("CUDA driver mismatch")


class _CudaAvailableStub:
    cuda = _FakeCuda(True)
    __version__ = "2.14.0+cu124"


class _BrokenTorchStub:
    cuda = _BrokenCuda()
    __version__ = "2.14.0+cu124"


def _resolve(requested, force_cpu, stub):
    """describe_device() with config and sys.modules['torch'] temporarily replaced."""
    yolo_device._STATE = None
    saved = (config.YOLO_DEVICE, config.YOLO_FORCE_CPU)
    try:
        config.YOLO_DEVICE = requested
        config.YOLO_FORCE_CPU = force_cpu
        if stub is None:
            sys.modules["torch"] = None
        else:
            sys.modules["torch"] = _FakeTorch(stub)
        return yolo_device.describe_device()
    finally:
        config.YOLO_DEVICE, config.YOLO_FORCE_CPU = saved
        yolo_device._STATE = None


def test_config_defaults_to_auto():
    assert config.YOLO_DEVICE == "auto"
    assert config.YOLO_FORCE_CPU is False


def test_auto_on_cuda_box_returns_zero():
    info = _resolve("auto", False, _CudaAvailableStub)
    assert info["cuda_available"] is True
    assert info["device"] == "0"
    assert "CUDA available" in info["reason"]


class _CpuTorchStub:
    cuda = _FakeCuda(False)
    __version__ = "2.14.0+cpu"


class _CudaBuildNoDeviceStub:
    cuda = _FakeCuda(False)
    __version__ = "2.14.0+cu124"


def test_auto_on_cpu_box_returns_cpu():
    info = _resolve("auto", False, _CpuTorchStub)
    assert info["cuda_available"] is False
    assert info["device"] == "cpu"
    assert info["torch_version"] == "2.14.0+cpu", "the module wrapper must expose __version__ from the stub"
    assert "CPU-only build" in info["reason"]


def test_auto_on_cuda_capable_build_without_device_returns_cpu():
    info = _resolve("auto", False, _CudaBuildNoDeviceStub)
    assert info["device"] == "cpu"
    assert info["reason"] == "torch.cuda.is_available() is False"


def test_force_cpu_overrides_cuda():
    info = _resolve("auto", True, _CudaAvailableStub)
    assert info["device"] == "cpu"
    assert info["reason"] == "config requests CPU"


def test_explicit_cpu_is_pinned():
    info = _resolve("cpu", False, _CudaAvailableStub)
    assert info["device"] == "cpu"
    assert info["reason"] == "config requests CPU"


def test_explicit_device_is_pinned():
    info = _resolve("1", False, _CudaBuildNoDeviceStub)
    assert info["device"] == "1"
    assert "pins a device" in info["reason"]


def test_missing_torch_falls_back_to_cpu():
    info = _resolve("auto", False, None)   # sys.modules["torch"] = None -> ImportError
    assert info["device"] == "cpu"
    assert "torch not importable" in info["reason"]


def test_cuda_is_available_raising_falls_back_to_cpu():
    # torch imports, torch.cuda is_available() raises (mismatched driver)
    info = _resolve("auto", False, _BrokenTorchStub)
    assert info["device"] == "cpu"
    assert "torch.cuda.is_available() failed" in info["reason"]


def test_state_is_cached():
    yolo_device._STATE = None
    saved = (config.YOLO_DEVICE, config.YOLO_FORCE_CPU)
    try:
        config.YOLO_DEVICE, config.YOLO_FORCE_CPU = "auto", False
        first = yolo_device.describe_device()
        config.YOLO_DEVICE = "cpu"
        assert yolo_device.describe_device() == first, "resolution must be cached"
    finally:
        config.YOLO_DEVICE, config.YOLO_FORCE_CPU = saved
        yolo_device._STATE = None


def test_real_environment_resolves_without_raising():
    yolo_device._STATE = None
    try:
        info = yolo_device.describe_device()
        assert info["device"] == "cpu" or str(info["device"]).isdigit()
        assert isinstance(yolo_device.cuda_install_hint(), str)
        assert "pip install" in yolo_device.cuda_install_hint()
    finally:
        yolo_device._STATE = None