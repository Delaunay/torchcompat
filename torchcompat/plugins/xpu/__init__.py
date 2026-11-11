"""Intel XPU support for pytorch"""

import torch

from torchcompat.core.errors import NotAvailable

if not hasattr(torch, "xpu"):
    try:
        import intel_extension_for_pytorch as ipex
    except ImportError as err:
        raise NotAvailable("Could not import intel_extension_for_pytorch") from err


if not torch.xpu.is_available():
    raise NotAvailable("torch.xpu is not available")


impl = torch.xpu


def set_enable_tf32(enable=True):
    try:
        import intel_extension_for_pytorch as ipex

        mode = ipex.FP32MathMode.TF32 if enable else ipex.FP32MathMode.FP32
        ipex.set_fp32_math_mode(device="xpu", mode=mode)
    except ImportError:
        # torch >= 2.9 ships XPU without IPEX
        try:
            torch.xpu.set_float32_matmul_precision("high" if enable else "highest")
        except AttributeError:
            pass


# Intel XPU has two possible torch.distributed backends:
#   * "xccl"  -> native oneCCL backend shipped in torch (>= 2.4), always present
#   * "ccl"   -> registered by the OPTIONAL intel/torch-ccl package
# Pick whichever torch actually registers so init_process_group() gets a valid
# backend on every XPU setup. See https://github.com/intel/torch-ccl
def _select_ccl_backend() -> str:
    names = []
    try:
        import torch.distributed as _dist

        bl = _dist.Backend.backend_list
        names = list(bl.keys()) if isinstance(bl, dict) else list(bl)
    except Exception:
        pass
    if "xccl" in names:
        return "xccl"
    if "ccl" in names:
        return "ccl"
    try:  # optional binding registers the "ccl" backend
        import oneccl_bindings_for_pytorch  # noqa: F401

        return "ccl"
    except Exception:
        return "gloo"


ccl = _select_ccl_backend()


#
# XPU amp now lives in the unified torch.amp (torch >= 2.4). Expose it under
# impl.amp with the device pre-bound, so existing call sites keep working:
#   amp.autocast(dtype=...)
#   amp.autocast(dtype=..., enabled=...)
#   amp.GradScaler(enabled=...)
#
import functools


def _make_amp():
    import types

    amp = types.ModuleType("torch.xpu.amp")
    amp.autocast = functools.partial(torch.amp.autocast, device_type="xpu")
    try:
        torch.amp.GradScaler(device="xpu", enabled=False)
    except (TypeError, RuntimeError, ValueError):
        # torch too old for device-generic GradScaler: fall back to no-op
        class NoScale:
            def __init__(self, enabled=True) -> None:
                pass

            def scale(self, loss):
                return loss

            def step(self, optimizer):
                optimizer.step()

            def update(self):
                pass

        amp.GradScaler = NoScale
    else:
        amp.GradScaler = functools.partial(torch.amp.GradScaler, device="xpu")
    return amp


impl.amp = _make_amp()

setattr(impl, "device_type", "xpu")
setattr(impl, "set_enable_tf32", set_enable_tf32)
setattr(impl, "ccl", ccl)
